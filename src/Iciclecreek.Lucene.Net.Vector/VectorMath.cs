using System;
using System.Numerics;

namespace Iciclecreek.Lucene.Net.Vector;

/// <summary>
/// SIMD-accelerated vector math utilities for cosine similarity.
/// Uses <see cref="Vector{T}"/> when hardware acceleration is available,
/// with a scalar fallback.
/// </summary>
public static class VectorMath
{
    /// <summary>
    /// Computes cosine similarity between two vectors.
    /// Returns a value in [-1, 1] where 1 means identical direction.
    /// </summary>
    /// <param name="a">First vector.</param>
    /// <param name="b">Second vector (must be same length as <paramref name="a"/>).</param>
    /// <param name="aNorm">Pre-computed L2 norm of <paramref name="a"/> (use <see cref="Norm"/> to compute).</param>
    public static float CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b, float aNorm)
    {
        if (a.Length != b.Length) return 0f;

        float dot, bNormSq;

#if NETSTANDARD2_0
        DotAndNormScalar(a, b, out dot, out bNormSq);
#else
        if (System.Numerics.Vector.IsHardwareAccelerated && a.Length >= Vector<float>.Count)
        {
            DotAndNormSimd(a, b, out dot, out bNormSq);
        }
        else
        {
            DotAndNormScalar(a, b, out dot, out bNormSq);
        }
#endif

        var bNorm = (float)Math.Sqrt(bNormSq);
        if (aNorm == 0f || bNorm == 0f) return 0f;
        return dot / (aNorm * bNorm);
    }

    /// <summary>
    /// Computes the L2 (Euclidean) norm of a vector.
    /// </summary>
    public static float Norm(ReadOnlySpan<float> v)
    {
        float sum;

#if NETSTANDARD2_0
        sum = 0f;
        for (int i = 0; i < v.Length; i++)
            sum += v[i] * v[i];
#else
        if (System.Numerics.Vector.IsHardwareAccelerated && v.Length >= Vector<float>.Count)
        {
            var vSum = Vector<float>.Zero;
            int simdLength = Vector<float>.Count;
            int i = 0;

            for (; i <= v.Length - simdLength; i += simdLength)
            {
                var vv = new Vector<float>(v.Slice(i));
                vSum += vv * vv;
            }

            sum = 0f;
            for (int j = 0; j < simdLength; j++)
                sum += vSum[j];

            for (; i < v.Length; i++)
                sum += v[i] * v[i];
        }
        else
        {
            sum = 0f;
            for (int i = 0; i < v.Length; i++)
                sum += v[i] * v[i];
        }
#endif

        return (float)Math.Sqrt(sum);
    }

#if !NETSTANDARD2_0
    private static void DotAndNormSimd(ReadOnlySpan<float> a, ReadOnlySpan<float> b, out float dot, out float bNormSq)
    {
        var vDot = Vector<float>.Zero;
        var vBNorm = Vector<float>.Zero;
        int simdLength = Vector<float>.Count;
        int i = 0;

        for (; i <= a.Length - simdLength; i += simdLength)
        {
            var va = new Vector<float>(a.Slice(i));
            var vb = new Vector<float>(b.Slice(i));
            vDot += va * vb;
            vBNorm += vb * vb;
        }

        dot = 0f;
        bNormSq = 0f;
        for (int j = 0; j < simdLength; j++)
        {
            dot += vDot[j];
            bNormSq += vBNorm[j];
        }

        for (; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            bNormSq += b[i] * b[i];
        }
    }
#endif

    private static void DotAndNormScalar(ReadOnlySpan<float> a, ReadOnlySpan<float> b, out float dot, out float bNormSq)
    {
        dot = 0f;
        bNormSq = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            bNormSq += b[i] * b[i];
        }
    }
}
