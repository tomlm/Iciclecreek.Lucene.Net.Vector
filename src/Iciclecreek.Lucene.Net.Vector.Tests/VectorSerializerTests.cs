using Iciclecreek.Lucene.Net.Vector;

namespace Iciclecreek.Lucene.Net.Vector.Tests;

public class VectorSerializerTests
{
    [Test]
    public void RoundTrip_FloatArray_ToBytes()
    {
        var original = new float[] { 1.0f, -2.5f, 3.14159f, 0f, float.MaxValue };
        var bytes = VectorSerializer.ToBytes(original);
        var result = VectorSerializer.FromBytes(bytes);

        Assert.That(result, Is.EqualTo(original));
    }

    [Test]
    public void RoundTrip_FloatArray_ToBytesRef()
    {
        var original = new float[] { 0.1f, 0.2f, 0.3f, 0.4f };
        var bytesRef = VectorSerializer.ToBytesRef(original);
        var result = VectorSerializer.FromBytesRef(bytesRef);

        Assert.That(result, Is.EqualTo(original));
    }

    [Test]
    public void RoundTrip_ReadOnlyMemory_ToBytesRef()
    {
        var original = new float[] { 1.0f, 2.0f, 3.0f };
        ReadOnlyMemory<float> memory = original.AsMemory();
        var bytesRef = VectorSerializer.ToBytesRef(memory);
        var result = VectorSerializer.FromBytesRefAsMemory(bytesRef);

        Assert.That(result.ToArray(), Is.EqualTo(original));
    }

    [Test]
    public void RoundTrip_SlicedReadOnlyMemory_ToBytesRef()
    {
        var source = new float[] { -1.0f, 1.0f, 2.0f, 3.0f, -2.0f };
        ReadOnlyMemory<float> memory = source.AsMemory(1, 3);
        var bytesRef = VectorSerializer.ToBytesRef(memory);
        var result = VectorSerializer.FromBytesRefAsMemory(bytesRef);

        Assert.That(result.ToArray(), Is.EqualTo(new float[] { 1.0f, 2.0f, 3.0f }));
    }

    [Test]
    public void EmptyVector_RoundTrips()
    {
        var original = Array.Empty<float>();
        var bytes = VectorSerializer.ToBytes(original);
        var result = VectorSerializer.FromBytes(bytes);

        Assert.That(result, Is.Empty);
    }

    [Test]
    public void SingleElement_RoundTrips()
    {
        var original = new float[] { 42.0f };
        var bytesRef = VectorSerializer.ToBytesRef(original);
        var result = VectorSerializer.FromBytesRef(bytesRef);

        Assert.That(result, Is.EqualTo(original));
    }

    [Test]
    public void LargeVector_RoundTrips()
    {
        var random = new Random(42);
        var original = Enumerable.Range(0, 1536).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        var bytesRef = VectorSerializer.ToBytesRef(original);
        var result = VectorSerializer.FromBytesRef(bytesRef);

        Assert.That(result, Is.EqualTo(original));
    }

    [Test]
    public void BytesRef_WithOffset_DeserializesCorrectly()
    {
        var original = new float[] { 1.0f, 2.0f, 3.0f };
        var bytes = VectorSerializer.ToBytes(original);

        // Simulate a BytesRef with offset (as Lucene may return)
        var paddedBytes = new byte[bytes.Length + 8];
        Array.Copy(bytes, 0, paddedBytes, 4, bytes.Length);
        var bytesRef = new global::Lucene.Net.Util.BytesRef(paddedBytes, 4, bytes.Length);

        var result = VectorSerializer.FromBytesRef(bytesRef);
        Assert.That(result, Is.EqualTo(original));
    }
}
