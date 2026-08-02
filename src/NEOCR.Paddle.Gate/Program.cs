using System.Buffers.Binary;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

return args switch
{
    ["validate-bundle", var bundlePath] => ValidateBundle(bundlePath),
    ["inspect", var modelPath] => Inspect(modelPath),
    _ => Usage(),
};

static int ValidateBundle(string bundlePath)
{
    var bundleRoot = Path.GetFullPath(bundlePath);
    var indexPath = ContainedPath(bundleRoot, "gate-index.json");
    var index = Deserialize<GateIndex>(indexPath);
    if (index.SchemaVersion != 1 || index.Cases.Count != 5)
    {
        throw new InvalidDataException("A gate bundle must contain schema version 1 and exactly five cases.");
    }

    foreach (var relativeCasePath in index.Cases)
    {
        var casePath = ContainedPath(bundleRoot, relativeCasePath);
        ValidateCase(
            Path.GetDirectoryName(casePath)!,
            Deserialize<GateCase>(casePath),
            index.AbsoluteTolerance,
            index.RelativeTolerance);
    }

    Console.WriteLine($"gate passed: {index.ModelPackageId}@{index.ModelPackageVersion} ({index.Cases.Count} models)");
    return 0;
}

static void ValidateCase(
    string caseRoot,
    GateCase testCase,
    float absoluteTolerance,
    float relativeTolerance)
{
    using var session = new InferenceSession(ContainedPath(caseRoot, testCase.ModelPath));
    if (!session.InputMetadata.Keys.Order().SequenceEqual(
        testCase.Inputs.Select(input => input.Name).Order(),
        StringComparer.Ordinal))
    {
        throw new InvalidDataException($"{testCase.Name}: ONNX input names do not match the gate case.");
    }

    var inputs = testCase.Inputs.Select(input =>
    {
        var values = ReadFloat32(ContainedPath(caseRoot, input.Path), input.Shape);
        return NamedOnnxValue.CreateFromTensor(
            input.Name,
            new DenseTensor<float>(values, input.Shape));
    }).ToArray();
    using var results = session.Run(inputs);
    var actualByName = results.ToDictionary(result => result.Name, StringComparer.Ordinal);
    if (!actualByName.Keys.Order().SequenceEqual(
        testCase.ExpectedOutputs.Select(output => output.Name).Order(),
        StringComparer.Ordinal))
    {
        throw new InvalidDataException($"{testCase.Name}: ONNX output names do not match Paddle.");
    }

    foreach (var expected in testCase.ExpectedOutputs)
    {
        var expectedValues = ReadFloat32(ContainedPath(caseRoot, expected.Path), expected.Shape);
        var actual = actualByName[expected.Name].AsTensor<float>();
        if (!actual.Dimensions.SequenceEqual(expected.Shape))
        {
            throw new InvalidDataException(
                $"{testCase.Name}/{expected.Name}: shape [{string.Join(',', actual.Dimensions.ToArray())}] " +
                $"does not match Paddle [{string.Join(',', expected.Shape)}].");
        }

        var actualValues = actual.ToArray();
        var maxAbsoluteError = 0f;
        for (var index = 0; index < actualValues.Length; index++)
        {
            var actualValue = actualValues[index];
            var expectedValue = expectedValues[index];
            if (!float.IsFinite(actualValue))
            {
                throw new InvalidDataException(
                    $"{testCase.Name}/{expected.Name}: ONNX output contains a non-finite value at {index}.");
            }

            var absoluteError = MathF.Abs(actualValue - expectedValue);
            maxAbsoluteError = MathF.Max(maxAbsoluteError, absoluteError);
            if (absoluteError > absoluteTolerance + (relativeTolerance * MathF.Abs(expectedValue)))
            {
                throw new InvalidDataException(
                    $"{testCase.Name}/{expected.Name}: parity failed at {index}; " +
                    $"Paddle={expectedValue:R}, ONNX={actualValue:R}, abs={absoluteError:R}.");
            }
        }

        Console.WriteLine(
            $"pass {testCase.Name}/{expected.Name} shape=[{string.Join(',', expected.Shape)}] " +
            $"maxAbs={maxAbsoluteError:R}");
    }
}

static int Inspect(string modelPath)
{
    using var session = new InferenceSession(Path.GetFullPath(modelPath));
    Console.WriteLine("inputs:");
    foreach (var input in session.InputMetadata)
    {
        Console.WriteLine($"  {input.Key}: {input.Value.ElementType.Name}[{string.Join(',', input.Value.Dimensions)}]");
    }

    Console.WriteLine("outputs:");
    foreach (var output in session.OutputMetadata)
    {
        Console.WriteLine($"  {output.Key}: {output.Value.ElementType.Name}[{string.Join(',', output.Value.Dimensions)}]");
    }

    return 0;
}

static float[] ReadFloat32(string path, IReadOnlyList<int> shape)
{
    var elementCount = shape.Aggregate(1L, (count, dimension) => checked(count * dimension));
    var expectedBytes = checked(elementCount * sizeof(float));
    var bytes = File.ReadAllBytes(path);
    if (bytes.LongLength != expectedBytes)
    {
        throw new InvalidDataException(
            $"Tensor '{path}' contains {bytes.LongLength} bytes; expected {expectedBytes}.");
    }

    var values = new float[checked((int)elementCount)];
    for (var index = 0; index < values.Length; index++)
    {
        values[index] = BitConverter.Int32BitsToSingle(
            BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(index * sizeof(float), sizeof(float))));
    }

    return values;
}

static T Deserialize<T>(string path)
{
    using var stream = File.OpenRead(path);
    return JsonSerializer.Deserialize<T>(stream, GateJson.Options)
        ?? throw new InvalidDataException($"'{path}' is not a valid {typeof(T).Name} document.");
}

static string ContainedPath(string root, string relativePath)
{
    if (Path.IsPathRooted(relativePath))
    {
        throw new InvalidDataException($"Gate bundle path must be relative: {relativePath}");
    }

    var fullRoot = Path.GetFullPath(root);
    var fullPath = Path.GetFullPath(Path.Combine(fullRoot, relativePath));
    var comparison = OperatingSystem.IsWindows()
        ? StringComparison.OrdinalIgnoreCase
        : StringComparison.Ordinal;
    if (!fullPath.StartsWith(fullRoot + Path.DirectorySeparatorChar, comparison))
    {
        throw new InvalidDataException($"Gate bundle path escapes its root: {relativePath}");
    }

    return fullPath;
}

static int Usage()
{
    Console.Error.WriteLine("Usage:");
    Console.Error.WriteLine("  NEOCR.Paddle.Gate validate-bundle <directory>");
    Console.Error.WriteLine("  NEOCR.Paddle.Gate inspect <model.onnx>");
    return 2;
}

sealed record GateIndex(
    int SchemaVersion,
    string ModelPackageId,
    string ModelPackageVersion,
    float AbsoluteTolerance,
    float RelativeTolerance,
    IReadOnlyList<string> Cases);

sealed record GateCase(
    string Name,
    string ModelPath,
    IReadOnlyList<TensorFile> Inputs,
    IReadOnlyList<TensorFile> ExpectedOutputs);

sealed record TensorFile(string Name, string Path, int[] Shape);

static class GateJson
{
    public static JsonSerializerOptions Options { get; } = new(JsonSerializerDefaults.Web);
}
