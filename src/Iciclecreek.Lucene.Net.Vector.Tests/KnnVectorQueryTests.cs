#if NET10_0_OR_GREATER
using Lucene.Net.Analysis.Standard;
using Lucene.Net.Documents;
using Lucene.Net.Index;
using Lucene.Net.Search;

namespace Iciclecreek.Lucene.Net.Vector.Tests;

/// <summary>
/// Tests for KnnVectorQuery (HNSW-based, .NET 10+ only).
/// </summary>
public class KnnVectorQueryTests : VectorQueryTestBase
{
    [Test]
    public void KnnVectorQuery_ReturnsResults()
    {
        IndexDocuments(
            ("1", new float[] { 1f, 0f, 0f, 0f }, null),
            ("2", new float[] { 0.9f, 0.1f, 0f, 0f }, null),
            ("3", new float[] { 0f, 0f, 0f, 1f }, null));

        using var reader = DirectoryReader.Open(_directory);
        var searcher = new IndexSearcher(reader);
        var query = new KnnVectorQuery("embedding", new float[] { 1f, 0f, 0f, 0f }, 3, reader);

        var topDocs = searcher.Search(query, 3);

        Assert.That(topDocs.TotalHits, Is.EqualTo(3));
        Assert.That(topDocs.ScoreDocs[0].Score, Is.GreaterThan(0));
    }

    [Test]
    public void KnnVectorQuery_OrdersByRelevance()
    {
        IndexDocuments(
            ("close", Normalize(new float[] { 0.95f, 0.05f, 0f, 0f }), null),
            ("far", Normalize(new float[] { 0f, 0f, 0f, 1f }), null),
            ("medium", Normalize(new float[] { 0.5f, 0.5f, 0f, 0f }), null));

        using var reader = DirectoryReader.Open(_directory);
        var searcher = new IndexSearcher(reader);
        var query = new KnnVectorQuery("embedding",
            Normalize(new float[] { 1f, 0f, 0f, 0f }), 3, reader,
            new VectorIndexOptions { Distance = VectorDistanceFunction.CosineForUnits });

        var topDocs = searcher.Search(query, 3);

        Assert.That(topDocs.ScoreDocs[0].Score, Is.GreaterThan(topDocs.ScoreDocs[1].Score));
        Assert.That(topDocs.ScoreDocs[1].Score, Is.GreaterThan(topDocs.ScoreDocs[2].Score));
    }

    [Test]
    public void KnnVectorQuery_ComposesWithBooleanQuery()
    {
        IndexDocuments(
            ("1", new float[] { 1f, 0f, 0f, 0f }, "tech"),
            ("2", new float[] { 0f, 0f, 0f, 1f }, "tech"),
            ("3", new float[] { 0.99f, 0.01f, 0f, 0f }, "sports"));

        using var reader = DirectoryReader.Open(_directory);
        var searcher = new IndexSearcher(reader);
        var knnQuery = new KnnVectorQuery("embedding", new float[] { 1f, 0f, 0f, 0f }, 3, reader);

        var boolQuery = new BooleanQuery
        {
            { new TermQuery(new Term("category", "tech")), Occur.MUST },
            { knnQuery, Occur.MUST }
        };

        var topDocs = searcher.Search(boolQuery, 10);

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
        IndexDocuments(("1", new float[] { 1f, 0f, 0f, 0f }, null));

        using var reader = DirectoryReader.Open(_directory);
        var searcher = new IndexSearcher(reader);

        var query1 = new KnnVectorQuery("embedding", new float[] { 1f, 0f, 0f, 0f }, 1, reader);
        var topDocs1 = searcher.Search(query1, 1);

        var query2 = new KnnVectorQuery("embedding", new float[] { 1f, 0f, 0f, 0f }, 1, reader) { Boost = 2.0f };
        var topDocs2 = searcher.Search(query2, 1);

        Assert.That(topDocs2.ScoreDocs[0].Score,
            Is.EqualTo(topDocs1.ScoreDocs[0].Score * 2.0f).Within(0.001f));
    }

    [Test]
    public void KnnVectorQuery_EmptyIndex_ReturnsNoResults()
    {
        using var analyzer = new StandardAnalyzer(Version);
        var config = new IndexWriterConfig(Version, analyzer);
        using var writer = new IndexWriter(_directory, config);
        writer.AddDocument(new Document { new StringField("id", "1", Field.Store.YES) });
        writer.Commit();
        writer.Dispose();

        using var reader = DirectoryReader.Open(_directory);
        var searcher = new IndexSearcher(reader);
        var query = new KnnVectorQuery("embedding", new float[] { 1f, 0f }, 5, reader);

        var topDocs = searcher.Search(query, 5);
        Assert.That(topDocs.TotalHits, Is.EqualTo(0));
    }

    [Test]
    public void KnnVectorQuery_DeletedDocs_Excluded()
    {
        IndexDocuments(
            ("1", new float[] { 1f, 0f, 0f, 0f }, null),
            ("2", new float[] { 0.9f, 0.1f, 0f, 0f }, null),
            ("3", new float[] { 0f, 0f, 0f, 1f }, null));

        using var analyzer = new StandardAnalyzer(Version);
        var config = new IndexWriterConfig(Version, analyzer);
        using (var writer = new IndexWriter(_directory, config))
        {
            writer.DeleteDocuments(new Term("id", "1"));
            writer.Commit();
        }

        using var reader = DirectoryReader.Open(_directory);
        var searcher = new IndexSearcher(reader);
        var query = new KnnVectorQuery("embedding", new float[] { 1f, 0f, 0f, 0f }, 3, reader);

        var topDocs = searcher.Search(query, 10);

        Assert.That(topDocs.TotalHits, Is.EqualTo(2));
        foreach (var scoreDoc in topDocs.ScoreDocs)
        {
            var doc = searcher.Doc(scoreDoc.Doc);
            Assert.That(doc.Get("id"), Is.Not.EqualTo("1"));
        }
    }

    [Test]
    public void KnnVectorQuery_DimensionMismatch_Throws()
    {
        IndexDocuments(
            ("1", new float[] { 1f, 0f, 0f, 0f }, null),
            ("2", new float[] { 0f, 1f, 0f, 0f }, null));

        using var reader = DirectoryReader.Open(_directory);
        var searcher = new IndexSearcher(reader);
        var query = new KnnVectorQuery("embedding", new float[] { 1f, 0f, 0f }, 2, reader);
        var ex = Assert.Throws<ArgumentException>(() => searcher.Search(query, 2));
        Assert.That(ex!.Message, Does.Contain("3 dimensions").And.Contain("4"));
    }

    [Test]
    public void KnnVectorQuery_Warmup_PreBuildsIndex()
    {
        IndexDocuments(
            ("1", new float[] { 1f, 0f, 0f, 0f }, null),
            ("2", new float[] { 0f, 1f, 0f, 0f }, null));

        using var reader = DirectoryReader.Open(_directory);
        KnnVectorQuery.Warmup(reader, "embedding");

        var searcher = new IndexSearcher(reader);
        var query = new KnnVectorQuery("embedding", new float[] { 1f, 0f, 0f, 0f }, 2, reader);
        var topDocs = searcher.Search(query, 2);

        Assert.That(topDocs.TotalHits, Is.EqualTo(2));
    }
}
#endif
