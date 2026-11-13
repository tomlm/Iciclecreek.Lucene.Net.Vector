using System;
using BenchmarkDotNet.Attributes;
using Lucene.Net.Documents;
using Lucene.Net.Index;
using Lucene.Net.Search;
using Lucene.Net.Store;
using Lucene.Net.Util;

namespace Iciclecreek.Lucene.Net.Vector;

/// <summary>
/// Re-scoring the documents a filter selects: the cost per selected document
/// of <see cref="VectorScoreQuery"/> and of the brute-force <see cref="CosineVectorQuery"/>.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(launchCount: 1, warmupCount: 3, iterationCount: 10)]
public class VectorScoreQueryBenchmarks
{
    private const int Dimensions = 384;

    private RAMDirectory _directory = null!;
    private DirectoryReader _reader = null!;
    private IndexSearcher _searcher = null!;
    private float[] _query = null!;

    [Params(20_000)]
    public int Documents { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(42);
        _directory = new RAMDirectory();

        using (var writer = new IndexWriter(_directory, new IndexWriterConfig(LuceneVersion.LUCENE_48, null)))
        {
            for (var i = 0; i < Documents; i++)
            {
                var doc = new Document
                {
                    // A tenth of the documents are "rare".
                    new StringField("kind", i % 10 == 0 ? "rare" : "common", Field.Store.NO),
                    new BinaryDocValuesField("v", VectorSerializer.ToBytesRef(RandomVector(random))),
                };
                writer.AddDocument(doc);
            }
            writer.Commit();
        }

        _reader = DirectoryReader.Open(_directory);
        _searcher = new IndexSearcher(_reader);
        _query = RandomVector(random);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _reader.Dispose();
        _directory.Dispose();
    }

    private static float[] RandomVector(Random random)
    {
        var vector = new float[Dimensions];
        for (var i = 0; i < vector.Length; i++) vector[i] = (float)(random.NextDouble() * 2 - 1);
        return vector;
    }

    [Benchmark(Description = "VectorScoreQuery over every document")]
    public int ScoreAll()
    {
        return _searcher.Search(new VectorScoreQuery(new MatchAllDocsQuery(), "v", _query), 10).TotalHits;
    }

    [Benchmark(Description = "VectorScoreQuery over a tenth of the documents")]
    public int ScoreTenth()
    {
        return _searcher.Search(new VectorScoreQuery(new TermQuery(new Term("kind", "rare")), "v", _query), 10).TotalHits;
    }

    [Benchmark(Description = "CosineVectorQuery (brute force, top 10)")]
    public int BruteForceTopTen()
    {
        return _searcher.Search(new CosineVectorQuery("v", _query, 10, _reader), 10).TotalHits;
    }
}
