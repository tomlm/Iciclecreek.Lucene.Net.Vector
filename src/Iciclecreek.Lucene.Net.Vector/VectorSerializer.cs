using global::Lucene.Net.Util;

namespace Iciclecreek.Lucene.Net.Vector;

/// <summary>
/// Converts between float[] vectors and byte[]/BytesRef for Lucene BinaryDocValuesField storage.
/// Uses little-endian format, 4 bytes per float.
/// </summary>
public static class VectorSerializer
{
    public static byte[] ToBytes(float[] vector)
    {
        var bytes = new byte[vector.Length * sizeof(float)];
        Buffer.BlockCopy(vector, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    public static float[] FromBytes(byte[] bytes)
    {
        var vector = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, vector, 0, bytes.Length);
        return vector;
    }

    public static BytesRef ToBytesRef(float[] vector)
    {
        return new BytesRef(ToBytes(vector));
    }

    public static float[] FromBytesRef(BytesRef bytesRef)
    {
        var length = bytesRef.Length / sizeof(float);
        var vector = new float[length];
        Buffer.BlockCopy(bytesRef.Bytes, bytesRef.Offset, vector, 0, bytesRef.Length);
        return vector;
    }

    public static BytesRef ToBytesRef(ReadOnlyMemory<float> vector)
    {
        var bytes = new byte[vector.Length * sizeof(float)];
        var span = vector.Span;
        for (int i = 0; i < span.Length; i++)
        {
            BitConverter.TryWriteBytes(bytes.AsSpan(i * sizeof(float)), span[i]);
        }
        return new BytesRef(bytes);
    }

    public static ReadOnlyMemory<float> FromBytesRefAsMemory(BytesRef bytesRef)
    {
        return FromBytesRef(bytesRef).AsMemory();
    }
}
