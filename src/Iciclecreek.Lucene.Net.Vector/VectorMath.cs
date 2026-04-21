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
    public static float CosineSimilarity(float[] a, float[] b, float aNorm)
    {
        if (a.Length != b.Length) return 0f;

        float dot, bNormSq;

        if (System.Numerics.Vector.IsHardwareAccelerated && a.Length >= Vector<float>.Count)
        {
            DotAndNormSimd(a, b, out dot, out bNormSq);
        }
        else
        {
            DotAndNormScalar(a, b, out dot, out bNormSq);
        }

        var bNorm = (float)Math.Sqrt(bNormSq);
        if (aNorm == 0f || bNorm == 0f) return 0f;
        return dot / (aNorm * bNorm);
    }

    /// <summary>
    /// Computes the L2 (Euclidean) norm of a vector.
    /// </summary>
    public static float Norm(float[] v)
    {
        float sum;

        if (System.Numerics.Vector.IsHardwareAccelerated && v.Length >= Vector<float>.Count)
        {
            var vSum = Vector<float>.Zero;
            int simdLength = Vector<float>.Count;
            int i = 0;

            for (; i <= v.Length - simdLength; i += simdLength)
            {
                var vv = new Vector<float>(v, i);
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

        return (float)Math.Sqrt(sum);
    }

    private static void DotAndNormSimd(float[] a, float[] b, out float dot, out float bNormSq)
    {
        var vDot = Vector<float>.Zero;
        var vBNorm = Vector<float>.Zero;
        int simdLength = Vector<float>.Count;
        int i = 0;

        for (; i <= a.Length - simdLength; i += simdLength)
        {
            var va = new Vector<float>(a, i);
            var vb = new Vector<float>(b, i);
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

    private static void DotAndNormScalar(float[] a, float[] b, out float dot, out float bNormSq)
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
