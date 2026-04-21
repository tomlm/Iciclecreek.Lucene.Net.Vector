using Lucene.Net.Analysis.Standard;
using Lucene.Net.Documents;
using Lucene.Net.Index;
using Lucene.Net.Search;

namespace Iciclecreek.Lucene.Net.Vector.Tests;

/// <summary>
/// Tests for VectorQuery (smart wrapper, all TFMs).
/// </summary>
public class VectorQueryTests : VectorQueryTestBase
{
    [Test]
    public void VectorQuery_ReturnsResults()
    {
        IndexDocuments(
            ("1", new float[] { 1f, 0f, 0f, 0f }, null),
            ("2", new float[] { 0.9f, 0.1f, 0f, 0f }, null),
            ("3", new float[] { 0f, 0f, 0f, 1f }, null));

        using var reader = DirectoryReader.Open(_directory);
        var searcher = new IndexSearcher(reader);
        var query = new VectorQuery("embedding", new float[] { 1f, 0f, 0f, 0f }, 3, reader);

        var topDocs = searcher.Search(query, 3);

        Assert.That(topDocs.TotalHits, Is.EqualTo(3));
        Assert.That(topDocs.ScoreDocs[0].Score, Is.GreaterThan(0));
    }

    [Test]
    public void VectorQuery_OrdersByRelevance()
    {
        IndexDocuments(
            ("close", Normalize(new float[] { 0.95f, 0.05f, 0f, 0f }), null),
            ("far", Normalize(new float[] { 0f, 0f, 0f, 1f }), null),
            ("medium", Normalize(new float[] { 0.5f, 0.5f, 0f, 0f }), null));

        using var reader = DirectoryReader.Open(_directory);
        var searcher = new IndexSearcher(reader);
        var query = new VectorQuery("embedding",
            Normalize(new float[] { 1f, 0f, 0f, 0f }), 3, reader);

        var topDocs = searcher.Search(query, 3);

        var ids = topDocs.ScoreDocs.Select(sd => searcher.Doc(sd.Doc).Get("id")).ToList();
        Assert.That(ids[0], Is.EqualTo("close"));
        Assert.That(ids[2], Is.EqualTo("far"));
    }

    [Test]
    public void VectorQuery_ComposesWithBooleanQuery()
    {
        IndexDocuments(
            ("1", new float[] { 1f, 0f, 0f, 0f }, "tech"),
            ("2", new float[] { 0f, 0f, 0f, 1f }, "tech"),
            ("3", new float[] { 0.99f, 0.01f, 0f, 0f }, "sports"));

        using var reader = DirectoryReader.Open(_directory);
        var searcher = new IndexSearcher(reader);
        var vectorQuery = new VectorQuery("embedding", new float[] { 1f, 0f, 0f, 0f }, 3, reader);

        var boolQuery = new BooleanQuery
        {
            { new TermQuery(new Term("category", "tech")), Occur.MUST },
            { vectorQuery, Occur.MUST }
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
    public void VectorQuery_EmptyIndex_ReturnsNoResults()
    {
        using var analyzer = new StandardAnalyzer(Version);
        var config = new IndexWriterConfig(Version, analyzer);
        using var writer = new IndexWriter(_directory, config);
        writer.AddDocument(new Document { new StringField("id", "1", Field.Store.YES) });
        writer.Commit();
        writer.Dispose();

        using var reader = DirectoryReader.Open(_directory);
        var searcher = new IndexSearcher(reader);
        var query = new VectorQuery("embedding", new float[] { 1f, 0f }, 5, reader);

        var topDocs = searcher.Search(query, 5);
        Assert.That(topDocs.TotalHits, Is.EqualTo(0));
    }

    [Test]
    public void VectorQuery_DeletedDocs_Excluded()
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
        var query = new VectorQuery("embedding", new float[] { 1f, 0f, 0f, 0f }, 3, reader);

        var topDocs = searcher.Search(query, 10);

        Assert.That(topDocs.TotalHits, Is.EqualTo(2));
        foreach (var scoreDoc in topDocs.ScoreDocs)
        {
            var doc = searcher.Doc(scoreDoc.Doc);
            Assert.That(doc.Get("id"), Is.Not.EqualTo("1"));
        }
    }
}
