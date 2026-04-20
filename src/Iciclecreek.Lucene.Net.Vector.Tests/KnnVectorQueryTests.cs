using Iciclecreek.Lucene.Net.Vector;
using Lucene.Net.Analysis.Standard;
using Lucene.Net.Documents;
using Lucene.Net.Index;
using Lucene.Net.Search;
using Lucene.Net.Store;
using Lucene.Net.Util;

namespace Iciclecreek.Lucene.Net.Vector.Tests;

public class KnnVectorQueryTests
{
    private RAMDirectory _directory = null!;
    private const LuceneVersion Version = LuceneVersion.LUCENE_48;

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

    private void IndexDocuments(params (string id, string category, float[] vector)[] docs)
    {
        using var analyzer = new StandardAnalyzer(Version);
        var config = new IndexWriterConfig(Version, analyzer);
        using var writer = new IndexWriter(_directory, config);

        foreach (var (id, category, vector) in docs)
        {
            var doc = new Document
            {
                new StringField("id", id, Field.Store.YES),
                new StringField("category", category, Field.Store.YES),
                new BinaryDocValuesField("embedding", VectorSerializer.ToBytesRef(vector)),
            };
            writer.AddDocument(doc);
        }
        writer.Commit();
    }

    [Test]
    public void KnnVectorQuery_ReturnsResultsViaSearcher()
    {
        IndexDocuments(
            ("1", "a", new float[] { 1f, 0f, 0f, 0f }),
            ("2", "a", new float[] { 0.9f, 0.1f, 0f, 0f }),
            ("3", "b", new float[] { 0f, 0f, 0f, 1f }));

        using var reader = DirectoryReader.Open(_directory);
        using var index = new LuceneVectorIndex("embedding");
        index.BuildIndex(reader);

        var searcher = new IndexSearcher(reader);
        var query = new KnnVectorQuery("embedding", new float[] { 1f, 0f, 0f, 0f }, 3, index);

        var topDocs = searcher.Search(query, 3);

        Assert.That(topDocs.TotalHits, Is.EqualTo(3));
        Assert.That(topDocs.ScoreDocs[0].Score, Is.GreaterThan(0));
    }

    [Test]
    public void KnnVectorQuery_OrdersByRelevance()
    {
        IndexDocuments(
            ("close", "a", new float[] { 0.95f, 0.05f, 0f, 0f }),
            ("far", "b", new float[] { 0f, 0f, 0f, 1f }),
            ("medium", "a", new float[] { 0.5f, 0.5f, 0f, 0f }));

        using var reader = DirectoryReader.Open(_directory);
        using var index = new LuceneVectorIndex("embedding");
        index.BuildIndex(reader);

        var searcher = new IndexSearcher(reader);
        var query = new KnnVectorQuery("embedding", new float[] { 1f, 0f, 0f, 0f }, 3, index);

        var topDocs = searcher.Search(query, 3);

        // Scores should be in descending order (Lucene sorts by score)
        Assert.That(topDocs.ScoreDocs[0].Score, Is.GreaterThan(topDocs.ScoreDocs[1].Score));
        Assert.That(topDocs.ScoreDocs[1].Score, Is.GreaterThan(topDocs.ScoreDocs[2].Score));
    }

    [Test]
    public void KnnVectorQuery_ComposesWithBooleanQuery_Filter()
    {
        IndexDocuments(
            ("1", "tech", new float[] { 1f, 0f, 0f, 0f }),
            ("2", "tech", new float[] { 0f, 0f, 0f, 1f }),
            ("3", "sports", new float[] { 0.99f, 0.01f, 0f, 0f }));

        using var reader = DirectoryReader.Open(_directory);
        using var index = new LuceneVectorIndex("embedding");
        index.BuildIndex(reader);

        var searcher = new IndexSearcher(reader);
        var knnQuery = new KnnVectorQuery("embedding", new float[] { 1f, 0f, 0f, 0f }, 3, index);

        // Filter to only "tech" category using MUST (no Occur.FILTER in Lucene.Net 4.8)
        var boolQuery = new BooleanQuery
        {
            { new TermQuery(new Term("category", "tech")), Occur.MUST },
            { knnQuery, Occur.MUST }
        };

        var topDocs = searcher.Search(boolQuery, 10);

        // Should only return tech documents
        Assert.That(topDocs.TotalHits, Is.EqualTo(2));
        foreach (var scoreDoc in topDocs.ScoreDocs)
        {
            var doc = searcher.Doc(scoreDoc.Doc);
            Assert.That(doc.Get("category"), Is.EqualTo("tech"));
        }
    }

    [Test]
    public void KnnVectorQuery_WithBoost_ScalesScores()
    {
        IndexDocuments(
            ("1", "a", new float[] { 1f, 0f, 0f, 0f }));

        using var reader = DirectoryReader.Open(_directory);
        using var index = new LuceneVectorIndex("embedding");
        index.BuildIndex(reader);

        var searcher = new IndexSearcher(reader);

        var query1 = new KnnVectorQuery("embedding", new float[] { 1f, 0f, 0f, 0f }, 1, index);
        var topDocs1 = searcher.Search(query1, 1);

        var query2 = new KnnVectorQuery("embedding", new float[] { 1f, 0f, 0f, 0f }, 1, index) { Boost = 2.0f };
        var topDocs2 = searcher.Search(query2, 1);

        Assert.That(topDocs2.ScoreDocs[0].Score,
            Is.EqualTo(topDocs1.ScoreDocs[0].Score * 2.0f).Within(0.001f));
    }

    [Test]
    public void KnnVectorQuery_ToString_ReturnsDescription()
    {
        using var index = new LuceneVectorIndex("embedding");
        var query = new KnnVectorQuery("embedding", new float[] { 1f, 0f }, 10, index);

        var str = query.ToString();
        Assert.That(str, Does.Contain("KnnVectorQuery"));
        Assert.That(str, Does.Contain("embedding"));
        Assert.That(str, Does.Contain("10"));
    }

    [Test]
    public void KnnVectorQuery_Equality()
    {
        using var index = new LuceneVectorIndex("embedding");
        var vec = new float[] { 1f, 0f };
        var q1 = new KnnVectorQuery("embedding", vec, 10, index);
        var q2 = new KnnVectorQuery("embedding", vec, 10, index);
        var q3 = new KnnVectorQuery("embedding", vec, 5, index);

        Assert.That(q1, Is.EqualTo(q2));
        Assert.That(q1, Is.Not.EqualTo(q3));
        Assert.That(q1.GetHashCode(), Is.EqualTo(q2.GetHashCode()));
    }
}
