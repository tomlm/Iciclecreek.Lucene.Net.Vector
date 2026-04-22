using System;
using System.Linq;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using Lucene.Net.Util;
using Microsoft.VSDiagnostics;

namespace Iciclecreek.Lucene.Net.Vector;
[CPUUsageDiagnoser]
[MemoryDiagnoser]
[SimpleJob(launchCount: 1, warmupCount: 1, iterationCount: 3)]
public class VectorSerializerBenchmarks
{
    private float[] _vector = null !;
    private byte[] _bytes = null !;
    private BytesRef _bytesRef = null !;
    private float _queryNorm;
    [Params(384)]
    public int Dimensions { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);
        _vector = Enumerable.Range(0, Dimensions).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        _bytes = VectorSerializer.ToBytes(_vector);
        _bytesRef = new BytesRef(_bytes);
        _queryNorm = VectorMath.Norm(_vector);
    }

    [Benchmark]
    public float[] FromBytes()
    {
        return VectorSerializer.FromBytes(_bytes);
    }

    [Benchmark]
    public int FromBytesRef()
    {
        return VectorSerializer.FromBytesRef(_bytesRef).Length;
    }

    [Benchmark]
    public float CosineSimilarityAfterFromBytesRef()
    {
        var docVector = VectorSerializer.FromBytesRef(_bytesRef);
        return VectorMath.CosineSimilarity(_vector, docVector, _queryNorm);
    }
}