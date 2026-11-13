#if NET10_0_OR_GREATER
using HNSW.Net;
using global::Lucene.Net.Index;
using global::Lucene.Net.Util;

namespace Iciclecreek.Lucene.Net.Vector;

/// <summary>
/// The vectors of one field in one index segment, held in memory for nearest
/// neighbor search. A segment with at least
/// <see cref="VectorIndexOptions.MinimumGraphSize"/> vectors gets an HNSW graph;
/// a smaller one is compared one by one.
/// Built and cached automatically per (segment, fieldName) via VectorIndexCache.
/// </summary>
/// <remarks>
/// The graph is built in the background, because building one takes seconds to
/// minutes. Until it is ready, searches compare the vectors one by one, which
/// gives the exact nearest neighbors.
/// </remarks>
/// <remarks>
/// A segment never changes once written, except that documents in it get
/// deleted. So the index holds every document of the segment, deleted or not,
/// and each search passes the documents that are live for its reader.
/// Document ids are the segment's own.
/// </remarks>
internal class LuceneVectorIndex : IDisposable
{
    /// <summary>
    /// Graphs are built one at a time, each after the one before it, to leave
    /// processors for searches.
    /// </summary>
    private static Task _buildQueue = Task.CompletedTask;
    private static readonly object BuildQueueLock = new();

    private volatile SmallWorld<float[], float>? _graph;
    private Task? _graphBuild;
    private readonly Dictionary<float[], int> _vectorToDocId = new(ReferenceEqualityComparer.Instance);
    private readonly string _vectorFieldName;
    private readonly VectorIndexOptions _options;
    private readonly Func<float[], float[], float> _distanceFunc;
    private readonly List<float[]> _vectors = new();
    private readonly List<int> _docIds = new();
    private volatile bool _disposed;

    internal LuceneVectorIndex(string vectorFieldName, VectorIndexOptions? options = null)
    {
        _vectorFieldName = vectorFieldName ?? throw new ArgumentNullException(nameof(vectorFieldName));
        _options = options ?? new VectorIndexOptions();
        _distanceFunc = GetDistanceFunction(_options.Distance);
    }

    internal int Count => _vectors.Count;
    internal int Dimensions => _vectors.Count > 0 ? _vectors[0].Length : 0;

    /// <summary>True when this segment is searched through an HNSW graph.</summary>
    internal bool HasGraph => _graph != null;

    internal bool IsDisposed => _disposed;

    /// <summary>
    /// Waits for the HNSW graph, when the segment gets one, and rethrows
    /// whatever stopped it from being built.
    /// </summary>
    internal void WaitForGraph() => _graphBuild?.GetAwaiter().GetResult();

    /// <summary>
    /// Load the vectors of one segment and, when there are enough of them,
    /// start building the HNSW graph. Called once, before the index is shared.
    /// </summary>
    internal void BuildIndex(AtomicReader segment)
    {
        ArgumentNullException.ThrowIfNull(segment);

        var docValues = segment.GetBinaryDocValues(_vectorFieldName);
        if (docValues == null)
            return;

        var bytesRef = new BytesRef();
        for (int docId = 0; docId < segment.MaxDoc; docId++)
        {
            docValues.Get(docId, bytesRef);

            if (bytesRef.Length == 0)
                continue;

            var vector = VectorSerializer.FromBytesRef(bytesRef).ToArray();

            if (_vectors.Count > 0 && vector.Length != _vectors[0].Length)
                throw new InvalidOperationException(
                    $"Dimension mismatch in field '{_vectorFieldName}': document {docId} of its segment has {vector.Length} dimensions but expected {_vectors[0].Length}.");

            _vectors.Add(vector);
            _docIds.Add(docId);
            _vectorToDocId[vector] = docId;
        }

        if (_vectors.Count > 0 && _vectors.Count >= _options.MinimumGraphSize)
        {
            lock (BuildQueueLock)
            {
                // Runs whether or not the build before it succeeded.
                _graphBuild = _buildQueue.ContinueWith(
                    _ => BuildGraph(), CancellationToken.None, TaskContinuationOptions.None, TaskScheduler.Default);
                _buildQueue = _graphBuild;
            }
        }
    }

    /// <summary>
    /// Search for the top-K nearest neighbors to the query vector among the
    /// documents of this segment that <paramref name="liveDocs"/> accepts.
    /// </summary>
    /// <param name="liveDocs">The segment's live documents, or <c>null</c> when none are deleted.</param>
    /// <returns>Segment document ids with their distances, nearest first.</returns>
    internal IReadOnlyList<SearchResult> Search(float[] queryVector, int topK, IBits? liveDocs = null)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(LuceneVectorIndex));

        if (_vectors.Count == 0 || topK <= 0)
            return Array.Empty<SearchResult>();

        if (queryVector.Length != Dimensions)
            throw new ArgumentException(
                $"Query vector has {queryVector.Length} dimensions but the index contains {Dimensions}-dimensional vectors.",
                nameof(queryVector));

        var graph = _graph;
        if (graph == null)
            return SearchExact(queryVector, topK, liveDocs);

        Func<float[], bool> itemFilter = liveDocs == null
            ? _ => true
            : vector => liveDocs.Get(_vectorToDocId[vector]);

        var results = graph.KNNSearch(queryVector, topK, itemFilter, CancellationToken.None)
            .Select(r => new SearchResult(_docIds[r.Id], r.Distance))
            .OrderBy(r => r.Distance)
            .ToList();

        // The graph walk can come up short when many of the documents near the
        // query are deleted. Count what is live before settling for less.
        if (results.Count < topK && liveDocs != null && results.Count < CountLive(liveDocs, topK))
            return SearchExact(queryVector, topK, liveDocs);

        return results;
    }

    private IReadOnlyList<SearchResult> SearchExact(float[] queryVector, int topK, IBits? liveDocs)
    {
        // The farthest of the best K so far is at the top, ready to be replaced.
        var best = new PriorityQueue<SearchResult, float>(Math.Min(topK, _vectors.Count));

        for (int i = 0; i < _vectors.Count; i++)
        {
            var docId = _docIds[i];
            if (liveDocs != null && !liveDocs.Get(docId))
                continue;

            var distance = _distanceFunc(queryVector, _vectors[i]);

            if (best.Count < topK)
            {
                best.Enqueue(new SearchResult(docId, distance), -distance);
            }
            else if (best.TryPeek(out _, out var negatedFarthest) && distance < -negatedFarthest)
            {
                best.DequeueEnqueue(new SearchResult(docId, distance), -distance);
            }
        }

        var results = new List<SearchResult>(best.Count);
        while (best.TryDequeue(out var result, out _))
            results.Add(result);

        results.Sort((a, b) => a.Distance.CompareTo(b.Distance));
        return results;
    }

    /// <summary>Counts live documents that have a vector, stopping at <paramref name="limit"/>.</summary>
    private int CountLive(IBits liveDocs, int limit)
    {
        int count = 0;
        for (int i = 0; i < _docIds.Count && count < limit; i++)
        {
            if (liveDocs.Get(_docIds[i]))
                count++;
        }
        return count;
    }

    private void BuildGraph()
    {
        // The segment may be gone by the time its turn comes.
        if (_disposed)
            return;

        var parameters = new SmallWorldParameters
        {
            M = _options.M,
            ConstructionPruning = _options.ConstructionPruning,
            EfSearch = _options.EfSearch,
            EnableDistanceCacheForConstruction = true,
        };

        var graph = new SmallWorld<float[], float>(
            _distanceFunc,
            DefaultRandomGenerator.Instance,
            parameters,
            _options.ThreadSafe);

        graph.AddItems(_vectors, progressReporter: null);

        // The cache of distances only helps while the graph is built, and it is
        // far larger than the graph: 12 bytes for every pair of vectors, up to 3 GB.
        graph.DisableDistanceCache();

        if (!_disposed)
            _graph = graph;
    }

    private static Func<float[], float[], float> GetDistanceFunction(VectorDistanceFunction distance)
    {
        var simd = System.Numerics.Vector.IsHardwareAccelerated;
        return distance switch
        {
            VectorDistanceFunction.Cosine => simd ? CosineDistance.SIMD : CosineDistance.NonOptimized,
            VectorDistanceFunction.CosineForUnits => simd ? CosineDistance.SIMDForUnits : CosineDistance.ForUnits,
            _ => throw new ArgumentOutOfRangeException(nameof(distance))
        };
    }

    public void Dispose()
    {
        // The vectors stay as they are: a search or the graph build may be reading them.
        _disposed = true;
        _graph = null;
    }
}
#endif
