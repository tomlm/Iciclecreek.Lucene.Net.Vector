#if NET10_0_OR_GREATER
using Lucene.Net.Analysis.Standard;
using Lucene.Net.Documents;
using Lucene.Net.Index;
using Lucene.Net.Search;

namespace Iciclecreek.Lucene.Net.Vector.Tests;

/// <summary>
/// KnnVectorQuery keeps one vector index per index segment, so a write only
/// costs an index for the segment it adds.
/// </summary>
public class KnnSegmentTests : VectorQueryTestBase
{
    private const int Dimensions = 16;

    /// <summary>Builds a graph however small the segment.</summary>
    private static VectorIndexOptions AlwaysGraph => new() { MinimumGraphSize = 0 };

    private static float[] RandomVector(Random random)
    {
        var vector = new float[Dimensions];
        for (var i = 0; i < vector.Length; i++) vector[i] = (float)(random.NextDouble() * 2 - 1);
        return vector;
    }

    /// <summary>Adds one segment of documents named <c>{prefix}{n}</c>.</summary>
    private Dictionary<string, float[]> AddSegment(string prefix, int count, Random random, Func<int, string?>? category = null)
    {
        var docs = Enumerable.Range(0, count)
            .Select(i => (id: prefix + i, vector: RandomVector(random), category: category?.Invoke(i)))
            .ToArray();
        IndexDocuments(docs);
        return docs.ToDictionary(d => d.id, d => d.vector);
    }

    private void Delete(Term term)
    {
        using var analyzer = new StandardAnalyzer(Version);
        using var writer = new IndexWriter(_directory, new IndexWriterConfig(Version, analyzer));
        writer.DeleteDocuments(term);
        writer.Commit();
    }

    private static List<string> Ids(IndexSearcher searcher, Query query, int n)
    {
        return searcher.Search(query, n).ScoreDocs.Select(d => searcher.Doc(d.Doc).Get("id")).ToList();
    }

    /// <summary>The K nearest by comparing every vector.</summary>
    private static List<string> ExactNearest(Dictionary<string, float[]> vectors, float[] query, int k)
    {
        var norm = VectorMath.Norm(query);
        return vectors
            .OrderByDescending(v => VectorMath.CosineSimilarity(query, v.Value, norm))
            .Take(k)
            .Select(v => v.Key)
            .ToList();
    }

    [Test]
    public void SmallSegments_AreComparedOneByOne_AndGiveTheExactNearest()
    {
        var random = new Random(1);
        var all = new Dictionary<string, float[]>();
        foreach (var prefix in new[] { "a", "b", "c" })
        {
            foreach (var doc in AddSegment(prefix, 200, random)) all.Add(doc.Key, doc.Value);
        }

        using var reader = DirectoryReader.Open(_directory);
        Assert.That(reader.Leaves.Count, Is.EqualTo(3));

        var queryVector = RandomVector(random);
        var ids = Ids(new IndexSearcher(reader), new KnnVectorQuery("embedding", queryVector, 10, reader), 10);

        Assert.That(ids, Is.EqualTo(ExactNearest(all, queryVector, 10)));
        foreach (var leaf in reader.Leaves)
        {
            Assert.That(VectorIndexCache.GetOrBuild(leaf.AtomicReader, "embedding").HasGraph, Is.False);
        }
    }

    [Test]
    public void SeveralGraphs_FindTheNearestAcrossSegments()
    {
        var random = new Random(2);
        var all = new Dictionary<string, float[]>();
        foreach (var prefix in new[] { "a", "b", "c" })
        {
            foreach (var doc in AddSegment(prefix, 400, random)) all.Add(doc.Key, doc.Value);
        }

        using var reader = DirectoryReader.Open(_directory);
        KnnVectorQuery.Warmup(reader, "embedding", AlwaysGraph);

        var queryVector = RandomVector(random);
        var ids = Ids(new IndexSearcher(reader), new KnnVectorQuery("embedding", queryVector, 10, reader, AlwaysGraph), 10);

        Assert.That(ids, Has.Count.EqualTo(10));
        foreach (var leaf in reader.Leaves)
        {
            Assert.That(VectorIndexCache.GetOrBuild(leaf.AtomicReader, "embedding").HasGraph, Is.True);
        }

        // A graph search is approximate; nearly all of the true nearest should be found.
        var found = ids.Intersect(ExactNearest(all, queryVector, 10)).Count();
        Assert.That(found, Is.GreaterThanOrEqualTo(8));
    }

    [Test]
    public void AfterAWrite_TheOldSegmentKeepsItsIndex()
    {
        var random = new Random(3);
        AddSegment("a", 300, random);

        LuceneVectorIndex before;
        using (var reader = DirectoryReader.Open(_directory))
        {
            KnnVectorQuery.Warmup(reader, "embedding", AlwaysGraph);
            before = VectorIndexCache.GetOrBuild(reader.Leaves[0].AtomicReader, "embedding");

            IndexDocuments(("new", RandomVector(random), null));

            using var reopened = DirectoryReader.OpenIfChanged(reader);
            Assert.That(reopened, Is.Not.Null);
            Assert.That(reopened!.Leaves.Count, Is.EqualTo(2));

            var searcher = new IndexSearcher(reopened);
            var ids = Ids(searcher, new KnnVectorQuery("embedding", RandomVector(random), 301, reopened, AlwaysGraph), 301);

            Assert.That(ids, Has.Count.EqualTo(301));
            Assert.That(ids, Does.Contain("new"));
            Assert.That(VectorIndexCache.GetOrBuild(reopened.Leaves[0].AtomicReader, "embedding"), Is.SameAs(before));
            Assert.That(VectorIndexCache.GetOrBuild(reopened.Leaves[1].AtomicReader, "embedding").Count, Is.EqualTo(1));
        }
    }

    [Test]
    public void ADelete_ReusesTheIndex_AndLeavesTheDocumentOut()
    {
        var random = new Random(4);
        var all = AddSegment("a", 300, random);
        var queryVector = all["a7"];

        using var reader = DirectoryReader.Open(_directory);
        KnnVectorQuery.Warmup(reader, "embedding", AlwaysGraph);
        var first = Ids(new IndexSearcher(reader), new KnnVectorQuery("embedding", queryVector, 5, reader, AlwaysGraph), 5);
        Assert.That(first[0], Is.EqualTo("a7"));
        var index = VectorIndexCache.GetOrBuild(reader.Leaves[0].AtomicReader, "embedding");

        Delete(new Term("id", "a7"));

        using var reopened = DirectoryReader.OpenIfChanged(reader);
        Assert.That(reopened, Is.Not.Null);

        var second = Ids(new IndexSearcher(reopened!), new KnnVectorQuery("embedding", queryVector, 5, reopened!, AlwaysGraph), 5);

        Assert.That(second, Has.Count.EqualTo(5));
        Assert.That(second, Does.Not.Contain("a7"));
        Assert.That(VectorIndexCache.GetOrBuild(reopened!.Leaves[0].AtomicReader, "embedding"), Is.SameAs(index));

        // The reader from before the delete still sees the document.
        var again = Ids(new IndexSearcher(reader), new KnnVectorQuery("embedding", queryVector, 5, reader, AlwaysGraph), 5);
        Assert.That(again[0], Is.EqualTo("a7"));
    }

    [Test]
    public void ASegmentWithMostDocumentsDeleted_StillGivesKResults()
    {
        var random = new Random(5);
        var all = AddSegment("a", 2000, random, i => i % 20 == 0 ? "keep" : "drop");
        Delete(new Term("category", "drop"));

        using var reader = DirectoryReader.Open(_directory);
        Assert.That(reader.NumDocs, Is.EqualTo(100));
        KnnVectorQuery.Warmup(reader, "embedding", AlwaysGraph);
        Assert.That(VectorIndexCache.GetOrBuild(reader.Leaves[0].AtomicReader, "embedding").HasGraph, Is.True);

        var queryVector = RandomVector(random);
        var searcher = new IndexSearcher(reader);
        var ids = Ids(searcher, new KnnVectorQuery("embedding", queryVector, 10, reader, AlwaysGraph), 10);

        var kept = all.Where(v => int.Parse(v.Key.Substring(1)) % 20 == 0).ToDictionary(v => v.Key, v => v.Value);

        Assert.That(ids, Has.Count.EqualTo(10));
        Assert.That(ids, Is.SubsetOf(kept.Keys));
        Assert.That(ids.Intersect(ExactNearest(kept, queryVector, 10)).Count(), Is.GreaterThanOrEqualTo(8));
    }

    [Test]
    public void UntilTheGraphIsReady_SearchesCompareOneByOne()
    {
        var random = new Random(8);
        var all = AddSegment("a", 3000, random);

        using var reader = DirectoryReader.Open(_directory);
        var queryVector = RandomVector(random);

        // Whether or not the graph is ready yet, the query answers, and with real neighbors.
        var ids = Ids(new IndexSearcher(reader), new KnnVectorQuery("embedding", queryVector, 10, reader), 10);
        Assert.That(ids.Intersect(ExactNearest(all, queryVector, 10)).Count(), Is.GreaterThanOrEqualTo(8));

        var index = VectorIndexCache.GetOrBuild(reader.Leaves[0].AtomicReader, "embedding");
        index.WaitForGraph();
        Assert.That(index.HasGraph, Is.True);

        ids = Ids(new IndexSearcher(reader), new KnnVectorQuery("embedding", queryVector, 10, reader), 10);
        Assert.That(ids.Intersect(ExactNearest(all, queryVector, 10)).Count(), Is.GreaterThanOrEqualTo(8));
    }

    [Test]
    public void ABuiltGraph_DoesNotKeepItsConstructionCache()
    {
        var random = new Random(9);
        AddSegment("a", 5000, random);

        using var reader = DirectoryReader.Open(_directory);
        var before = GC.GetTotalMemory(forceFullCollection: true);

        KnnVectorQuery.Warmup(reader, "embedding");
        Assert.That(VectorIndexCache.GetOrBuild(reader.Leaves[0].AtomicReader, "embedding").HasGraph, Is.True);

        // 5,000 vectors of 16 floats and their graph are a few megabytes. The cache
        // of distances that HNSW fills while it builds is 195 MB at this size.
        var held = GC.GetTotalMemory(forceFullCollection: true) - before;
        Assert.That(held, Is.LessThan(32 * 1024 * 1024));

        var ids = Ids(new IndexSearcher(reader), new KnnVectorQuery("embedding", RandomVector(random), 10, reader), 10);
        Assert.That(ids, Has.Count.EqualTo(10));
    }

    [Test]
    public void ASegmentWithoutVectors_IsSkipped()
    {
        var random = new Random(6);
        AddSegment("a", 20, random);

        using (var analyzer = new StandardAnalyzer(Version))
        using (var writer = new IndexWriter(_directory, new IndexWriterConfig(Version, analyzer)))
        {
            writer.AddDocument(new Document { new StringField("id", "plain", Field.Store.YES) });
            writer.Commit();
        }

        using var reader = DirectoryReader.Open(_directory);
        Assert.That(reader.Leaves.Count, Is.EqualTo(2));

        var ids = Ids(new IndexSearcher(reader), new KnnVectorQuery("embedding", RandomVector(random), 30, reader), 30);

        Assert.That(ids, Has.Count.EqualTo(20));
        Assert.That(ids, Does.Not.Contain("plain"));
    }

    [Test]
    public void DisposingTheLastReader_DropsTheSegmentsIndex()
    {
        var random = new Random(7);
        AddSegment("a", 50, random);

        LuceneVectorIndex index;
        using (var reader = DirectoryReader.Open(_directory))
        {
            KnnVectorQuery.Warmup(reader, "embedding");
            index = VectorIndexCache.GetOrBuild(reader.Leaves[0].AtomicReader, "embedding");
            Assert.That(index.Count, Is.EqualTo(50));
            Assert.That(index.IsDisposed, Is.False);
        }

        Assert.That(index.IsDisposed, Is.True);
    }
}
#endif
