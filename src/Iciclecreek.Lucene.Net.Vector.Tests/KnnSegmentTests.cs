#if NET10_0_OR_GREATER
using System.Runtime.CompilerServices;
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
            Assert.That(VectorIndexCache.GetOrBuild(leaf.AtomicReader, "embedding", AlwaysGraph).HasGraph, Is.True);
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
            before = VectorIndexCache.GetOrBuild(reader.Leaves[0].AtomicReader, "embedding", AlwaysGraph);

            IndexDocuments(("new", RandomVector(random), null));

            using var reopened = DirectoryReader.OpenIfChanged(reader);
            Assert.That(reopened, Is.Not.Null);
            Assert.That(reopened!.Leaves.Count, Is.EqualTo(2));

            var searcher = new IndexSearcher(reopened);
            var ids = Ids(searcher, new KnnVectorQuery("embedding", RandomVector(random), 301, reopened, AlwaysGraph), 301);

            Assert.That(ids, Has.Count.EqualTo(301));
            Assert.That(ids, Does.Contain("new"));
            Assert.That(VectorIndexCache.GetOrBuild(reopened.Leaves[0].AtomicReader, "embedding", AlwaysGraph), Is.SameAs(before));
            Assert.That(VectorIndexCache.GetOrBuild(reopened.Leaves[1].AtomicReader, "embedding", AlwaysGraph).Count, Is.EqualTo(1));
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
        var index = VectorIndexCache.GetOrBuild(reader.Leaves[0].AtomicReader, "embedding", AlwaysGraph);

        Delete(new Term("id", "a7"));

        using var reopened = DirectoryReader.OpenIfChanged(reader);
        Assert.That(reopened, Is.Not.Null);

        var second = Ids(new IndexSearcher(reopened!), new KnnVectorQuery("embedding", queryVector, 5, reopened!, AlwaysGraph), 5);

        Assert.That(second, Has.Count.EqualTo(5));
        Assert.That(second, Does.Not.Contain("a7"));
        Assert.That(VectorIndexCache.GetOrBuild(reopened!.Leaves[0].AtomicReader, "embedding", AlwaysGraph), Is.SameAs(index));

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
        Assert.That(VectorIndexCache.GetOrBuild(reader.Leaves[0].AtomicReader, "embedding", AlwaysGraph).HasGraph, Is.True);

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
    public void EveryCaller_GetsTheOptionsItPassed()
    {
        var random = new Random(13);
        AddSegment("a", 500, random);

        using var reader = DirectoryReader.Open(_directory);
        var segment = reader.Leaves[0].AtomicReader;
        var queryVector = RandomVector(random);

        // The first caller passes no options, and 500 vectors are too few for a graph by default.
        var ids = Ids(new IndexSearcher(reader), new KnnVectorQuery("embedding", queryVector, 10, reader), 10);
        Assert.That(ids, Has.Count.EqualTo(10));
        var byDefault = VectorIndexCache.GetOrBuild(segment, "embedding");
        Assert.That(byDefault.HasGraph, Is.False);

        // A later caller that asks for a graph gets one.
        KnnVectorQuery.Warmup(reader, "embedding", AlwaysGraph);
        Assert.That(VectorIndexCache.GetOrBuild(segment, "embedding", AlwaysGraph).HasGraph, Is.True);
        Assert.That(byDefault.HasGraph, Is.False);

        // The same values share one index, whichever object carries them.
        Assert.That(VectorIndexCache.GetOrBuild(segment, "embedding", AlwaysGraph), Is.SameAs(VectorIndexCache.GetOrBuild(segment, "embedding", AlwaysGraph)));
        Assert.That(VectorIndexCache.GetOrBuild(segment, "embedding", new VectorIndexOptions()), Is.SameAs(byDefault));

        // These vectors are not of unit length, so the distance for unit vectors differs from the cosine distance.
        var forUnits = VectorIndexCache.GetOrBuild(segment, "embedding", new VectorIndexOptions { Distance = VectorDistanceFunction.CosineForUnits });
        Assert.That(forUnits, Is.Not.SameAs(byDefault));
        Assert.That(
            forUnits.Search(queryVector, 1)[0].Distance,
            Is.Not.EqualTo(byDefault.Search(queryVector, 1)[0].Distance).Within(0.01f));
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

    [Test]
    public void ADisposedSegment_IsNotKeptByItsQueuedBuild()
    {
        var random = new Random(11);
        AddSegment("a", 50, random);

        using (LuceneVectorIndex.HoldBuildQueue())
        {
            var index = QueueABuildThenDispose();

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            // The build is still waiting for its turn, and the index holds every vector of the segment.
            Assert.That(index.IsAlive, Is.False);
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private WeakReference QueueABuildThenDispose()
    {
        using var reader = DirectoryReader.Open(_directory);
        var index = VectorIndexCache.GetOrBuild(reader.Leaves[0].AtomicReader, "embedding", AlwaysGraph);
        Assert.That(index.Count, Is.EqualTo(50));
        return new WeakReference(index);
    }

    [Test]
    public void DisposingASegment_StopsItsRunningBuild()
    {
        var random = new Random(12);
        AddSegment("a", 5000, random);

        LuceneVectorIndex index;
        using (var reader = DirectoryReader.Open(_directory))
        {
            index = VectorIndexCache.GetOrBuild(reader.Leaves[0].AtomicReader, "embedding");
            Assert.That(SpinWait.SpinUntil(() => index.GraphBuildProgress > 0, TimeSpan.FromSeconds(60)), Is.True);
        }

        // Returns, rather than throws, once the build has stopped.
        index.WaitForGraph();

        Assert.That(index.IsDisposed, Is.True);
        Assert.That(index.HasGraph, Is.False);
        Assert.That(index.GraphBuildProgress, Is.LessThan(index.Count));
    }

    [Test]
    public void ADisposedSegment_LetsItsGraphBeCollected()
    {
        var random = new Random(10);
        AddSegment("a", 300, random);

        var graphItems = SearchThroughTheGraphThenDispose(RandomVector(random));

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        // If this fails after an HNSW upgrade, LuceneVectorIndex.ReleaseGraph no longer
        // finds the package's private searcher pool, and every graph of a segment that
        // is merged away or closed stays in memory until the process ends.
        Assert.That(graphItems.IsAlive, Is.False);
    }

    /// <summary>
    /// Its own method, so that nothing here still refers to the graph when the caller collects.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private WeakReference SearchThroughTheGraphThenDispose(float[] queryVector)
    {
        using var reader = DirectoryReader.Open(_directory);
        KnnVectorQuery.Warmup(reader, "embedding", AlwaysGraph);

        var index = VectorIndexCache.GetOrBuild(reader.Leaves[0].AtomicReader, "embedding", AlwaysGraph);
        var graphItems = index.GraphItems;
        Assert.That(graphItems, Is.Not.Null);

        // HNSW pools its searchers per thread, so search from two.
        List<string> Search() => Ids(new IndexSearcher(reader), new KnnVectorQuery("embedding", queryVector, 5, reader, AlwaysGraph), 5);
        Assert.That(Search(), Has.Count.EqualTo(5));
        Assert.That(Task.Run(Search).GetAwaiter().GetResult(), Has.Count.EqualTo(5));

        return new WeakReference(graphItems);
    }
}
#endif
