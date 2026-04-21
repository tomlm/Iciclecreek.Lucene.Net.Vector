#if NET8_0_OR_GREATER
using ElBruno.LocalEmbeddings;
using ElBruno.LocalEmbeddings.Options;
#endif
using Iciclecreek.Lucene.Net.Vector;
using Lucene.Net.Analysis.Standard;
using Lucene.Net.Documents;
using Lucene.Net.Index;
using Lucene.Net.Search;
using Lucene.Net.Store;
using Lucene.Net.Util;
using Microsoft.Extensions.AI;
using System.Reflection;

namespace Iciclecreek.Lucene.Net.Vector.Tests;

/// <summary>
/// Shared test infrastructure for vector query tests.
/// </summary>
public abstract class VectorQueryTestBase
{
    protected RAMDirectory _directory = null!;
    protected const LuceneVersion Version = LuceneVersion.LUCENE_48;
    private IEmbeddingGenerator<string, Embedding<float>> _generator = null!;

    [OneTimeSetUp]
    public void OneTimeSetUp()
    {
        _generator = 
#if NET8_0_OR_GREATER
        new LocalEmbeddingGenerator(new LocalEmbeddingsOptions
        {
            ModelName = "SmartComponents/bge-micro-v2",
            PreferQuantized = true
        });
#else
        null;
#endif
    }

    [OneTimeTearDown]
    public void OneTimeTearDown()
    {
        _generator?.Dispose();
    }

    [SetUp]
    public void Setup()
    {
        _directory = new RAMDirectory();
    }

    [TearDown]
    public void TearDown()
    {
        _directory?.Dispose();
    }

    protected void IndexDocuments(params (string id, float[] vector, string? category)[] docs)
    {
        using var analyzer = new StandardAnalyzer(Version);
        var config = new IndexWriterConfig(Version, analyzer);
        using var writer = new IndexWriter(_directory, config);

        foreach (var (id, vector, category) in docs)
        {
            var doc = new Document
            {
                new StringField("id", id, Field.Store.YES),
                new BinaryDocValuesField("embedding", VectorSerializer.ToBytesRef(vector)),
            };
            if (category != null)
                doc.Add(new StringField("category", category, Field.Store.YES));
            writer.AddDocument(doc);
        }
        writer.Commit();
    }

    protected static float[] Normalize(float[] v)
    {
        var mag = (float)Math.Sqrt(v.Sum(x => x * x));
        return v.Select(x => x / mag).ToArray();
    }
}

/// <summary>
/// Shared utility tests.
/// </summary>
public class SearchResultTests
{
    [Test]
    public void SearchResult_Score_IsInverseOfDistance()
    {
        var result = new SearchResult(0, 0.5f);
        Assert.That(result.Score, Is.EqualTo(1f / 1.5f).Within(0.0001f));

        var perfect = new SearchResult(0, 0f);
        Assert.That(perfect.Score, Is.EqualTo(1f));
    }
}
