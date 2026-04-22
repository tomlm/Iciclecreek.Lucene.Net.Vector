using Lucene.Net.Analysis.Standard;
using Lucene.Net.Documents;
using Lucene.Net.Index;
using Lucene.Net.Search;

namespace Iciclecreek.Lucene.Net.Vector.Tests;

/// <summary>
/// Tests for VectorScoreQuery (CustomScoreQuery-based cosine re-scoring).
/// </summary>
public class VectorScoreQueryTests : VectorQueryTestBase
{
    [Test]
    public void VectorScoreQuery_ReturnsResults()
    {
        IndexDocuments(
            ("1", new float[] { 1f, 0f, 0f, 0f }, null),
            ("2", new float[] { 0.9f, 0.1f, 0f, 0f }, null),
            ("3", new float[] { 0f, 0f, 0f, 1f }, null));

        using var reader = DirectoryReader.Open(_directory);
        var searcher = new IndexSearcher(reader);
        var query = new VectorScoreQuery(
            new MatchAllDocsQuery(), "embedding", new float[] { 1f, 0f, 0f, 0f });

        var topDocs = searcher.Search(query, 10);

        Assert.That(topDocs.TotalHits, Is.EqualTo(3));
        Assert.That(topDocs.ScoreDocs[0].Score, Is.GreaterThan(0));
    }

    [Test]
    public void VectorScoreQuery_OrdersByCosineSimilarity()
    {
        IndexDocuments(
            ("close", Normalize(new float[] { 0.95f, 0.05f, 0f, 0f }), null),
            ("far", Normalize(new float[] { 0f, 0f, 0f, 1f }), null),
            ("medium", Normalize(new float[] { 0.5f, 0.5f, 0f, 0f }), null));

        using var reader = DirectoryReader.Open(_directory);
        var searcher = new IndexSearcher(reader);
        var query = new VectorScoreQuery(
            new MatchAllDocsQuery(), "embedding", Normalize(new float[] { 1f, 0f, 0f, 0f }));

        var topDocs = searcher.Search(query, 10);

        var ids = topDocs.ScoreDocs.Select(sd => searcher.Doc(sd.Doc).Get("id")).ToList();
        Assert.That(ids[0], Is.EqualTo("close"));
        Assert.That(ids[2], Is.EqualTo("far"));
    }

    [Test]
    public void VectorScoreQuery_ScoresInZeroToOneRange()
    {
        IndexDocuments(
            ("identical", Normalize(new float[] { 1f, 0f, 0f, 0f }), null),
            ("opposite", Normalize(new float[] { -1f, 0f, 0f, 0f }), null),
            ("orthogonal", Normalize(new float[] { 0f, 1f, 0f, 0f }), null));

        using var reader = DirectoryReader.Open(_directory);
        var searcher = new IndexSearcher(reader);
        var query = new VectorScoreQuery(
            new MatchAllDocsQuery(), "embedding", Normalize(new float[] { 1f, 0f, 0f, 0f }));

        var topDocs = searcher.Search(query, 10);

        foreach (var scoreDoc in topDocs.ScoreDocs)
        {
            Assert.That(scoreDoc.Score, Is.GreaterThanOrEqualTo(0f));
            Assert.That(scoreDoc.Score, Is.LessThanOrEqualTo(1f));
        }

        // Identical vector => score 1.0, opposite => 0.0, orthogonal => 0.5
        var scores = topDocs.ScoreDocs
            .Select(sd => (Id: searcher.Doc(sd.Doc).Get("id"), Score: sd.Score))
            .ToDictionary(x => x.Id, x => x.Score);

        Assert.That(scores["identical"], Is.EqualTo(1f).Within(0.01f));
        Assert.That(scores["opposite"], Is.EqualTo(0f).Within(0.01f));
        Assert.That(scores["orthogonal"], Is.EqualTo(0.5f).Within(0.01f));
    }

    [Test]
    public void VectorScoreQuery_FilterQueryLimitsResults()
    {
        IndexDocuments(
            ("1", new float[] { 1f, 0f, 0f, 0f }, "tech"),
            ("2", new float[] { 0f, 0f, 0f, 1f }, "tech"),
            ("3", new float[] { 0.99f, 0.01f, 0f, 0f }, "sports"));

        using var reader = DirectoryReader.Open(_directory);
        var searcher = new IndexSearcher(reader);
        var query = new VectorScoreQuery(
            new TermQuery(new Term("category", "tech")),
            "embedding",
            new float[] { 1f, 0f, 0f, 0f });

        var topDocs = searcher.Search(query, 10);

        Assert.That(topDocs.TotalHits, Is.EqualTo(2));
        foreach (var scoreDoc in topDocs.ScoreDocs)
        {
            var doc = searcher.Doc(scoreDoc.Doc);
            Assert.That(doc.Get("category"), Is.EqualTo("tech"));
        }
    }

    [Test]
    public void VectorScoreQuery_NoVectorField_ReturnsZeroScore()
    {
        // Index docs without a vector field
        using var analyzer = new StandardAnalyzer(Version);
        var config = new IndexWriterConfig(Version, analyzer);
        using var writer = new IndexWriter(_directory, config);
        writer.AddDocument(new Document { new StringField("id", "1", Field.Store.YES) });
        writer.Commit();
        writer.Dispose();

        using var reader = DirectoryReader.Open(_directory);
        var searcher = new IndexSearcher(reader);
        var query = new VectorScoreQuery(
            new MatchAllDocsQuery(), "embedding", new float[] { 1f, 0f });

        var topDocs = searcher.Search(query, 10);

        // Doc matches MatchAllDocsQuery but gets score 0 from missing vector
        Assert.That(topDocs.TotalHits, Is.EqualTo(1));
        Assert.That(topDocs.ScoreDocs[0].Score, Is.EqualTo(0f));
    }

    [Test]
    public void VectorScoreQuery_ComposesWithBooleanQuery()
    {
        IndexDocuments(
            ("1", Normalize(new float[] { 1f, 0f, 0f, 0f }), "tech"),
            ("2", Normalize(new float[] { 0f, 0f, 0f, 1f }), "tech"),
            ("3", Normalize(new float[] { 0.99f, 0.01f, 0f, 0f }), "sports"));

        using var reader = DirectoryReader.Open(_directory);
        var searcher = new IndexSearcher(reader);
        var vectorScore = new VectorScoreQuery(
            new TermQuery(new Term("category", "tech")),
            "embedding",
            Normalize(new float[] { 1f, 0f, 0f, 0f }));

        var boolQuery = new BooleanQuery
        {
            { vectorScore, Occur.MUST }
        };

        var topDocs = searcher.Search(boolQuery, 10);

        Assert.That(topDocs.TotalHits, Is.EqualTo(2));
        // Most similar tech doc should be first
        var firstId = searcher.Doc(topDocs.ScoreDocs[0].Doc).Get("id");
        Assert.That(firstId, Is.EqualTo("1"));
    }

    [Test]
    public void VectorScoreQuery_DeletedDocs_Excluded()
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
        var query = new VectorScoreQuery(
            new MatchAllDocsQuery(), "embedding", new float[] { 1f, 0f, 0f, 0f });

        var topDocs = searcher.Search(query, 10);

        Assert.That(topDocs.TotalHits, Is.EqualTo(2));
        foreach (var scoreDoc in topDocs.ScoreDocs)
        {
            var doc = searcher.Doc(scoreDoc.Doc);
            Assert.That(doc.Get("id"), Is.Not.EqualTo("1"));
        }
    }

    [Test]
    public void VectorScoreQuery_ToString_ContainsFieldAndSubQuery()
    {
        var query = new VectorScoreQuery(
            new MatchAllDocsQuery(), "embedding", new float[] { 1f, 0f });

        var result = query.ToString("embedding");

        Assert.That(result, Does.Contain("VectorScoreQuery"));
        Assert.That(result, Does.Contain("embedding"));
    }
}
