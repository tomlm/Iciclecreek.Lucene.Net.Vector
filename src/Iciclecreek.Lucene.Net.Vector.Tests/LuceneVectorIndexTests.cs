using Iciclecreek.Lucene.Net.Vector;
using Lucene.Net.Analysis.Standard;
using Lucene.Net.Documents;
using Lucene.Net.Index;
using Lucene.Net.Search;
using Lucene.Net.Store;
using Lucene.Net.Util;

namespace Iciclecreek.Lucene.Net.Vector.Tests;

public class LuceneVectorIndexTests
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

    private void IndexDocuments(params (string id, float[] vector, string? text)[] docs)
    {
        using var analyzer = new StandardAnalyzer(Version);
        var config = new IndexWriterConfig(Version, analyzer);
        using var writer = new IndexWriter(_directory, config);

        foreach (var (id, vector, text) in docs)
        {
            var doc = new Document
            {
                new StringField("id", id, Field.Store.YES),
                new BinaryDocValuesField("embedding", VectorSerializer.ToBytesRef(vector)),
            };
            if (text != null)
                doc.Add(new TextField("text", text, Field.Store.YES));
            writer.AddDocument(doc);
        }
        writer.Commit();
    }

    private static float[] MakeVector(int dims, float seed)
    {
        var vec = new float[dims];
        for (int i = 0; i < dims; i++)
            vec[i] = MathF.Sin(seed + i);
        return vec;
    }

    [Test]
    public void BuildIndex_LoadsVectorsFromLucene()
    {
        IndexDocuments(
            ("1", MakeVector(4, 1.0f), null),
            ("2", MakeVector(4, 2.0f), null),
            ("3", MakeVector(4, 3.0f), null));

        using var reader = DirectoryReader.Open(_directory);
        using var index = new LuceneVectorIndex("embedding");
        index.BuildIndex(reader);

        Assert.That(index.Count, Is.EqualTo(3));
    }

    [Test]
    public void Search_ReturnsNearestNeighbors()
    {
        // Use unit vectors for predictable cosine distance
        var queryVector = Normalize(new float[] { 1f, 0f, 0f, 0f });
        IndexDocuments(
            ("close", Normalize(new float[] { 0.9f, 0.1f, 0f, 0f }), null),
            ("medium", Normalize(new float[] { 0.5f, 0.5f, 0f, 0f }), null),
            ("far", Normalize(new float[] { 0f, 0f, 0f, 1f }), null));

        using var reader = DirectoryReader.Open(_directory);
        using var index = new LuceneVectorIndex("embedding",
            new VectorIndexOptions { Distance = VectorDistanceFunction.CosineForUnits });
        index.BuildIndex(reader);

        var results = index.Search(queryVector, 3);

        Assert.That(results.Count, Is.EqualTo(3));
        // The closest vector should be first (lowest distance)
        Assert.That(results[0].Distance, Is.LessThan(results[2].Distance));
    }

    private static float[] Normalize(float[] v)
    {
        var mag = MathF.Sqrt(v.Sum(x => x * x));
        return v.Select(x => x / mag).ToArray();
    }

    [Test]
    public void Search_WithDocIdFilter_RestrictsResults()
    {
        IndexDocuments(
            ("1", new float[] { 1f, 0f, 0f, 0f }, null),
            ("2", new float[] { 0.9f, 0.1f, 0f, 0f }, null),
            ("3", new float[] { 0f, 1f, 0f, 0f }, null));

        using var reader = DirectoryReader.Open(_directory);
        using var index = new LuceneVectorIndex("embedding");
        index.BuildIndex(reader);

        // Only allow doc IDs 1 and 2 (skip doc 0 which is closest)
        var accepted = new HashSet<int> { 1, 2 };
        var results = index.Search(new float[] { 1f, 0f, 0f, 0f }, 3, accepted);

        Assert.That(results.All(r => accepted.Contains(r.DocId)), Is.True);
    }

    [Test]
    public void AddVector_IncrementallyUpdatesIndex()
    {
        IndexDocuments(
            ("1", new float[] { 1f, 0f, 0f, 0f }, null));

        using var reader = DirectoryReader.Open(_directory);
        using var index = new LuceneVectorIndex("embedding");
        index.BuildIndex(reader);

        Assert.That(index.Count, Is.EqualTo(1));

        // Add a new vector incrementally
        index.AddVector(99, new float[] { 0f, 1f, 0f, 0f });

        Assert.That(index.Count, Is.EqualTo(2));

        var results = index.Search(new float[] { 0f, 1f, 0f, 0f }, 1);
        Assert.That(results[0].DocId, Is.EqualTo(99));
    }

    [Test]
    public void RemoveVector_ExcludesFromSearch()
    {
        IndexDocuments(
            ("1", new float[] { 1f, 0f, 0f, 0f }, null),
            ("2", new float[] { 0f, 1f, 0f, 0f }, null));

        using var reader = DirectoryReader.Open(_directory);
        using var index = new LuceneVectorIndex("embedding");
        index.BuildIndex(reader);

        // Remove doc 0 (closest to query)
        index.RemoveVector(0);

        var results = index.Search(new float[] { 1f, 0f, 0f, 0f }, 2);
        Assert.That(results.All(r => r.DocId != 0), Is.True);
    }

    [Test]
    public void Serialize_Deserialize_RoundTrips()
    {
        IndexDocuments(
            ("1", new float[] { 1f, 0f, 0f, 0f }, null),
            ("2", new float[] { 0f, 1f, 0f, 0f }, null),
            ("3", new float[] { 0f, 0f, 1f, 0f }, null));

        using var reader = DirectoryReader.Open(_directory);
        using var index = new LuceneVectorIndex("embedding");
        index.BuildIndex(reader);

        // Serialize
        using var ms = new MemoryStream();
        index.Serialize(ms);
        ms.Position = 0;

        // Deserialize
        using var restored = LuceneVectorIndex.Deserialize(ms, "embedding");

        Assert.That(restored.Count, Is.EqualTo(3));

        // Search should still work
        var results = restored.Search(new float[] { 1f, 0f, 0f, 0f }, 1);
        Assert.That(results.Count, Is.EqualTo(1));
        Assert.That(results[0].DocId, Is.EqualTo(0));
    }

    [Test]
    public void Search_EmptyIndex_ReturnsEmpty()
    {
        using var analyzer = new StandardAnalyzer(Version);
        var config = new IndexWriterConfig(Version, analyzer);
        using var writer = new IndexWriter(_directory, config);
        writer.AddDocument(new Document { new StringField("id", "1", Field.Store.YES) });
        writer.Commit();
        writer.Dispose();

        using var reader = DirectoryReader.Open(_directory);
        using var index = new LuceneVectorIndex("embedding");
        index.BuildIndex(reader);

        var results = index.Search(new float[] { 1f, 0f }, 5);
        Assert.That(results, Is.Empty);
    }

    [Test]
    public void SearchResult_Score_IsInverseOfDistance()
    {
        var result = new SearchResult(0, 0.5f);
        Assert.That(result.Score, Is.EqualTo(1f / 1.5f).Within(0.0001f));

        var perfect = new SearchResult(0, 0f);
        Assert.That(perfect.Score, Is.EqualTo(1f));
    }

    [Test]
    public void DeletedLuceneDocs_ExcludedFromVectorSearch()
    {
        // Index 3 documents
        IndexDocuments(
            ("1", new float[] { 1f, 0f, 0f, 0f }, null),
            ("2", new float[] { 0.9f, 0.1f, 0f, 0f }, null),
            ("3", new float[] { 0f, 0f, 0f, 1f }, null));

        // Delete doc "1" (the closest to our query vector) via Lucene
        using var analyzer = new StandardAnalyzer(Version);
        var config = new IndexWriterConfig(Version, analyzer);
        using (var writer = new IndexWriter(_directory, config))
        {
            writer.DeleteDocuments(new Term("id", "1"));
            writer.Commit();
        }

        // Rebuild index from the updated reader — deleted doc should be excluded
        using var reader = DirectoryReader.Open(_directory);
        using var index = new LuceneVectorIndex("embedding");
        index.BuildIndex(reader);

        Assert.That(index.Count, Is.EqualTo(2), "Deleted doc should not be in the HNSW index");

        var results = index.Search(new float[] { 1f, 0f, 0f, 0f }, 3);

        // Verify deleted doc's vector is not in results
        var searcher = new IndexSearcher(reader);
        foreach (var result in results)
        {
            var doc = searcher.Doc(result.DocId);
            Assert.That(doc.Get("id"), Is.Not.EqualTo("1"),
                "Deleted document should not appear in vector search results");
        }
    }

    [Test]
    public void DeletedLuceneDocs_ExcludedFromKnnVectorQuery()
    {
        IndexDocuments(
            ("1", new float[] { 1f, 0f, 0f, 0f }, "keep"),
            ("2", new float[] { 0.95f, 0.05f, 0f, 0f }, "delete"),
            ("3", new float[] { 0f, 0f, 0f, 1f }, "keep"));

        // Delete doc "2" which is very close to our query
        using var analyzer = new StandardAnalyzer(Version);
        var config = new IndexWriterConfig(Version, analyzer);
        using (var writer = new IndexWriter(_directory, config))
        {
            writer.DeleteDocuments(new Term("id", "2"));
            writer.Commit();
        }

        using var reader = DirectoryReader.Open(_directory);
        using var index = new LuceneVectorIndex("embedding");
        index.BuildIndex(reader);

        var searcher = new IndexSearcher(reader);
        var query = new KnnVectorQuery("embedding", new float[] { 1f, 0f, 0f, 0f }, 3, index);
        var topDocs = searcher.Search(query, 10);

        Assert.That(topDocs.TotalHits, Is.EqualTo(2));
        foreach (var scoreDoc in topDocs.ScoreDocs)
        {
            var doc = searcher.Doc(scoreDoc.Doc);
            Assert.That(doc.Get("id"), Is.Not.EqualTo("2"),
                "Deleted document should not appear in KnnVectorQuery results");
        }
    }

    [Test]
    public void UpdatedLuceneDocs_ReflectedInVectorSearch()
    {
        // Use unit vectors with CosineForUnits for predictable distances
        IndexDocuments(
            ("1", Normalize(new float[] { 1f, 0f, 0f, 0f }), null),
            ("2", Normalize(new float[] { 0f, 1f, 0f, 0f }), null),
            ("3", Normalize(new float[] { 0f, 0f, 1f, 0f }), null));

        // Update doc "1" — was near [1,0,0,0], now near [0,0,0,1]
        using var analyzer = new StandardAnalyzer(Version);
        var config = new IndexWriterConfig(Version, analyzer);
        using (var writer = new IndexWriter(_directory, config))
        {
            writer.DeleteDocuments(new Term("id", "1"));
            var doc = new Document
            {
                new StringField("id", "1", Field.Store.YES),
                new BinaryDocValuesField("embedding",
                    VectorSerializer.ToBytesRef(Normalize(new float[] { 0f, 0f, 0f, 1f }))),
            };
            writer.AddDocument(doc);
            writer.Commit();
        }

        // Rebuild — should see the updated vector
        using var reader = DirectoryReader.Open(_directory);
        using var index = new LuceneVectorIndex("embedding",
            new VectorIndexOptions { Distance = VectorDistanceFunction.CosineForUnits });
        index.BuildIndex(reader);

        // Query for the NEW vector — doc "1" should now be closest to [0,0,0,1]
        var results = index.Search(Normalize(new float[] { 0f, 0f, 0f, 1f }), 1);
        var searcher = new IndexSearcher(reader);
        var topDoc = searcher.Doc(results[0].DocId);
        Assert.That(topDoc.Get("id"), Is.EqualTo("1"),
            "Updated document should be found by its new vector");

        // Query for the OLD vector [1,0,0,0] — doc "1" should NOT be closest anymore
        var results2 = index.Search(Normalize(new float[] { 1f, 0f, 0f, 0f }), 1);
        var topDoc2 = searcher.Doc(results2[0].DocId);
        Assert.That(topDoc2.Get("id"), Is.Not.EqualTo("1"),
            "Updated document should not match its old vector");
    }
}
