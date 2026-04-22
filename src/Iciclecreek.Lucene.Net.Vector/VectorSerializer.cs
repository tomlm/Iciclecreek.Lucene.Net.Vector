using System.Runtime.InteropServices;
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
        if (vector.Length == 0)
            return Array.Empty<byte>();

        var bytes = AllocateByteArray(vector.Length * sizeof(float));
        Buffer.BlockCopy(vector, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    public static float[] FromBytes(byte[] bytes)
    {
        if (bytes.Length == 0)
            return Array.Empty<float>();

        var vector = AllocateFloatArray(bytes.Length / sizeof(float));
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
        if (length == 0)
            return Array.Empty<float>();

        var vector = AllocateFloatArray(length);
        Buffer.BlockCopy(bytesRef.Bytes, bytesRef.Offset, vector, 0, bytesRef.Length);
        return vector;
    }

    public static BytesRef ToBytesRef(ReadOnlyMemory<float> vector)
    {
        var sourceBytes = MemoryMarshal.AsBytes(vector.Span);
        if (sourceBytes.Length == 0)
            return new BytesRef(Array.Empty<byte>());

        var bytes = AllocateByteArray(sourceBytes.Length);
        sourceBytes.CopyTo(bytes);
        return new BytesRef(bytes);
    }

    public static ReadOnlyMemory<float> FromBytesRefAsMemory(BytesRef bytesRef)
    {
        return FromBytesRef(bytesRef).AsMemory();
    }

    private static byte[] AllocateByteArray(int length)
    {
#if NETSTANDARD2_0
        return new byte[length];
#else
        return GC.AllocateUninitializedArray<byte>(length);
#endif
    }

    private static float[] AllocateFloatArray(int length)
    {
#if NETSTANDARD2_0
        return new float[length];
#else
        return GC.AllocateUninitializedArray<float>(length);
#endif
    }
}
