using Iciclecreek.Lucene.Net.Vector;
using System.Reflection;
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

    private void IndexDocuments(params (string id, float[] vector, string? category)[] docs)
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

    private static float[] Normalize(float[] v)
    {
        var mag = MathF.Sqrt(v.Sum(x => x * x));
        return v.Select(x => x / mag).ToArray();
    }

    private static bool IsReaderCached(IndexReader reader)
    {
        var cacheType = typeof(KnnVectorQuery).Assembly.GetType("Iciclecreek.Lucene.Net.Vector.VectorIndexCache", throwOnError: true)!;
        var cacheField = cacheType.GetField("_cache", BindingFlags.NonPublic | BindingFlags.Static)!;
        var cache = cacheField.GetValue(null)!;
        var tryGetValue = cache.GetType().GetMethod("TryGetValue")!;
        var arguments = new object?[] { reader, null };

        return (bool)tryGetValue.Invoke(cache, arguments)!;
    }

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
    public void KnnVectorQuery_ToString_ReturnsDescription()
    {
        IndexDocuments(("1", new float[] { 1f, 0f }, null));
        using var reader = DirectoryReader.Open(_directory);
        var query = new KnnVectorQuery("embedding", new float[] { 1f, 0f }, 10, reader);

        var str = query.ToString();
        Assert.That(str, Does.Contain("KnnVectorQuery"));
        Assert.That(str, Does.Contain("embedding"));
        Assert.That(str, Does.Contain("10"));
    }

    [Test]
    public void KnnVectorQuery_Equality()
    {
        IndexDocuments(("1", new float[] { 1f, 0f }, null));
        using var reader = DirectoryReader.Open(_directory);
        var vec = new float[] { 1f, 0f };
        var q1 = new KnnVectorQuery("embedding", vec, 10, reader);
        var q2 = new KnnVectorQuery("embedding", vec, 10, reader);
        var q3 = new KnnVectorQuery("embedding", vec, 5, reader);

        Assert.That(q1, Is.EqualTo(q2));
        Assert.That(q1, Is.Not.EqualTo(q3));
        Assert.That(q1.GetHashCode(), Is.EqualTo(q2.GetHashCode()));
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
    public void DeletedDocs_ExcludedFromVectorSearch()
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
            Assert.That(doc.Get("id"), Is.Not.EqualTo("1"),
                "Deleted document should not appear in vector search results");
        }
    }

    [Test]
    public void UpdatedDocs_ReflectedInVectorSearch()
    {
        IndexDocuments(
            ("1", Normalize(new float[] { 1f, 0f, 0f, 0f }), null),
            ("2", Normalize(new float[] { 0f, 1f, 0f, 0f }), null),
            ("3", Normalize(new float[] { 0f, 0f, 1f, 0f }), null));

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

        using var reader = DirectoryReader.Open(_directory);
        var searcher = new IndexSearcher(reader);
        var options = new VectorIndexOptions { Distance = VectorDistanceFunction.CosineForUnits };

        // Query for the NEW vector — doc "1" should be closest
        var query1 = new KnnVectorQuery("embedding",
            Normalize(new float[] { 0f, 0f, 0f, 1f }), 1, reader, options);
        var topDocs1 = searcher.Search(query1, 1);
        Assert.That(searcher.Doc(topDocs1.ScoreDocs[0].Doc).Get("id"), Is.EqualTo("1"));

        // Query for the OLD vector — doc "1" should NOT be closest
        var query2 = new KnnVectorQuery("embedding",
            Normalize(new float[] { 1f, 0f, 0f, 0f }), 1, reader, options);
        var topDocs2 = searcher.Search(query2, 1);
        Assert.That(searcher.Doc(topDocs2.ScoreDocs[0].Doc).Get("id"), Is.Not.EqualTo("1"));
    }

    [Test]
    public void VectorsPersistInDocValues_SurviveFullRoundTrip()
    {
        var vectors = new Dictionary<string, float[]>
        {
            ["doc1"] = Normalize(new float[] { 1f, 0f, 0f, 0f }),
            ["doc2"] = Normalize(new float[] { 0f, 1f, 0f, 0f }),
            ["doc3"] = Normalize(new float[] { 0f, 0f, 1f, 0f }),
            ["doc4"] = Normalize(new float[] { 0.7f, 0.7f, 0f, 0f }),
        };

        // Phase 1: Write documents, commit, dispose writer
        {
            using var analyzer = new StandardAnalyzer(Version);
            var config = new IndexWriterConfig(Version, analyzer);
            using var writer = new IndexWriter(_directory, config);

            foreach (var (id, vector) in vectors)
            {
                var doc = new Document
                {
                    new StringField("id", id, Field.Store.YES),
                    new StringField("category", id.Contains("1") || id.Contains("4") ? "groupA" : "groupB", Field.Store.YES),
                    new BinaryDocValuesField("embedding", VectorSerializer.ToBytesRef(vector)),
                };
                writer.AddDocument(doc);
            }
            writer.Commit();
        }

        // Phase 2: Open fresh reader, verify raw DocValues round-trip
        {
            using var reader = DirectoryReader.Open(_directory);
            foreach (var leaf in reader.Leaves)
            {
                var docValues = leaf.AtomicReader.GetBinaryDocValues("embedding");
                Assert.That(docValues, Is.Not.Null);

                for (int i = 0; i < leaf.AtomicReader.MaxDoc; i++)
                {
                    var storedDoc = leaf.AtomicReader.Document(i);
                    var id = storedDoc.Get("id");

                    var bytesRef = new BytesRef();
                    docValues.Get(i, bytesRef);
                    var recovered = VectorSerializer.FromBytesRef(bytesRef);

                    Assert.That(recovered, Is.EqualTo(vectors[id]).Within(0.0001f));
                }
            }
        }

        // Phase 3: Open another fresh reader, run KNN queries
        {
            using var reader = DirectoryReader.Open(_directory);
            var searcher = new IndexSearcher(reader);
            var options = new VectorIndexOptions { Distance = VectorDistanceFunction.CosineForUnits };

            // Pure vector search
            var query1 = new KnnVectorQuery("embedding",
                Normalize(new float[] { 0f, 1f, 0f, 0f }), 4, reader, options);
            var topDocs1 = searcher.Search(query1, 4);

            Assert.That(topDocs1.TotalHits, Is.EqualTo(4));
            Assert.That(searcher.Doc(topDocs1.ScoreDocs[0].Doc).Get("id"), Is.EqualTo("doc2"));
            Assert.That(topDocs1.ScoreDocs[0].Score, Is.GreaterThan(topDocs1.ScoreDocs[3].Score));

            // Filtered KNN
            var filteredQuery = new BooleanQuery
            {
                { new TermQuery(new Term("category", "groupB")), Occur.MUST },
                { new KnnVectorQuery("embedding",
                    Normalize(new float[] { 0f, 0f, 1f, 0f }), 4, reader, options), Occur.MUST }
            };
            var filteredDocs = searcher.Search(filteredQuery, 10);

            Assert.That(filteredDocs.TotalHits, Is.EqualTo(2));
            foreach (var sd in filteredDocs.ScoreDocs)
                Assert.That(searcher.Doc(sd.Doc).Get("category"), Is.EqualTo("groupB"));
            Assert.That(searcher.Doc(filteredDocs.ScoreDocs[0].Doc).Get("id"), Is.EqualTo("doc3"));
        }
    }

    [Test]
    public void HnswGraphIsCachedPerReader()
    {
        IndexDocuments(
            ("1", new float[] { 1f, 0f, 0f, 0f }, null),
            ("2", new float[] { 0f, 1f, 0f, 0f }, null));

        using var reader = DirectoryReader.Open(_directory);
        var searcher = new IndexSearcher(reader);

        // Two queries on the same reader should both work (cache hit on second)
        var query1 = new KnnVectorQuery("embedding", new float[] { 1f, 0f, 0f, 0f }, 2, reader);
        var topDocs1 = searcher.Search(query1, 2);
        Assert.That(topDocs1.TotalHits, Is.EqualTo(2));

        var query2 = new KnnVectorQuery("embedding", new float[] { 0f, 1f, 0f, 0f }, 2, reader);
        var topDocs2 = searcher.Search(query2, 2);
        Assert.That(topDocs2.TotalHits, Is.EqualTo(2));

        // Results should differ based on query vector
        Assert.That(topDocs1.ScoreDocs[0].Doc, Is.Not.EqualTo(topDocs2.ScoreDocs[0].Doc));
    }

    [Test]
    public void DimensionMismatch_ThrowsOnSearch()
    {
        // Index 4-dimensional vectors
        IndexDocuments(
            ("1", new float[] { 1f, 0f, 0f, 0f }, null),
            ("2", new float[] { 0f, 1f, 0f, 0f }, null));

        using var reader = DirectoryReader.Open(_directory);
        var searcher = new IndexSearcher(reader);

        // Search with a 3-dimensional vector — should throw
        var query = new KnnVectorQuery("embedding", new float[] { 1f, 0f, 0f }, 2, reader);
        var ex = Assert.Throws<ArgumentException>(() => searcher.Search(query, 2));
        Assert.That(ex!.Message, Does.Contain("3 dimensions").And.Contain("4"));
    }

    [Test]
    public void Warmup_PreBuildsHnswIndex()
    {
        IndexDocuments(
            ("1", new float[] { 1f, 0f, 0f, 0f }, null),
            ("2", new float[] { 0f, 1f, 0f, 0f }, null));

        using var reader = DirectoryReader.Open(_directory);

        // Warmup builds the index before any query
        KnnVectorQuery.Warmup(reader, "embedding");

        // Subsequent query should use the cached index
        var searcher = new IndexSearcher(reader);
        var query = new KnnVectorQuery("embedding", new float[] { 1f, 0f, 0f, 0f }, 2, reader);
        var topDocs = searcher.Search(query, 2);

        Assert.That(topDocs.TotalHits, Is.EqualTo(2));
    }

    [Test]
    public void CacheEntryIsRemovedWhenReaderIsDisposed()
    {
        IndexDocuments(
            ("1", new float[] { 1f, 0f, 0f, 0f }, null),
            ("2", new float[] { 0f, 1f, 0f, 0f }, null));

        using var reader = DirectoryReader.Open(_directory);

        KnnVectorQuery.Warmup(reader, "embedding");
        Assert.That(IsReaderCached(reader), Is.True);

        reader.Dispose();

        Assert.That(IsReaderCached(reader), Is.False);
    }
}
