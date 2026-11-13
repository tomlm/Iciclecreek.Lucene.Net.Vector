#if NET10_0_OR_GREATER
using System.Runtime.CompilerServices;
using global::Lucene.Net.Index;

namespace Iciclecreek.Lucene.Net.Vector;

/// <summary>
/// Caches vector indexes per (segment, fieldName, options).
/// A segment never changes once written, so its index stays valid across
/// every reader that includes the segment: after a commit, only the segments
/// that are new have an index built.
/// </summary>
/// <remarks>
/// The key is the segment's <see cref="IndexReader.CoreCacheKey"/>, which a
/// reopened reader shares with the reader it came from for every segment
/// they have in common. Deleting documents does not change it; the deleted
/// documents are passed to each search instead. Neither does updating a
/// doc value in place (<c>IndexWriter.UpdateBinaryDocValue</c>), so a vector
/// changed that way is not seen: replace the document instead.
/// <para>
/// Callers that pass different option values for the same segment and field
/// each get an index of their own, with its own copy of the vectors. Sharing
/// one would give every caller after the first the first caller's distance
/// function and graph settings for as long as the segment lives.
/// </para>
/// </remarks>
internal static class VectorIndexCache
{
    // ConditionalWeakTable keys on the identity of the segment's core key — when
    // the key is GC'd, the cache entry is automatically cleaned up. Its HNSW graph is
    // only let go when the segment is disposed (see LuceneVectorIndex.ReleaseGraph),
    // so a reader that is dropped without being disposed leaves its graphs behind.
    private static readonly ConditionalWeakTable<object, Dictionary<IndexKey, LuceneVectorIndex>> _cache = new();
    private static readonly object _lock = new();
    private static readonly SegmentDisposedListener _disposedListener = new();

    /// <summary>
    /// Get or build the vector index for the given segment, field and options.
    /// No options is the same as the default options, and options are compared
    /// by their values, not by instance.
    /// Thread-safe: concurrent callers for the same segment will block until the
    /// first caller has loaded the vectors. Nobody waits for a graph.
    /// </summary>
    internal static LuceneVectorIndex GetOrBuild(AtomicReader segment, string fieldName, VectorIndexOptions? options = null)
    {
        var key = segment.CoreCacheKey;
        var indexKey = new IndexKey(fieldName, options ?? new VectorIndexOptions());
        Dictionary<IndexKey, LuceneVectorIndex>? fieldMap;

        lock (_lock)
        {
            if (!_cache.TryGetValue(key, out fieldMap))
            {
                fieldMap = new Dictionary<IndexKey, LuceneVectorIndex>();
                _cache.Add(key, fieldMap);

                // The core of a segment outlives any one reader of it.
                if (segment is SegmentReader segmentReader)
                    segmentReader.AddCoreDisposedListener(_disposedListener);
                else
                    segment.AddReaderDisposedListener(_disposedListener);
            }
        }

        lock (fieldMap)
        {
            if (fieldMap.TryGetValue(indexKey, out var existing))
                return existing;

            // A copy, so that the index keeps the values it is filed under
            // whatever the caller does with its options object afterwards.
            var index = new LuceneVectorIndex(fieldName, indexKey.ToOptions());
            index.BuildIndex(segment);
            fieldMap[indexKey] = index;
            return index;
        }
    }

    private static void Remove(object key)
    {
        Dictionary<IndexKey, LuceneVectorIndex>? fieldMap;

        lock (_lock)
        {
            if (!_cache.TryGetValue(key, out fieldMap))
                return;

            _cache.Remove(key);
        }

        List<LuceneVectorIndex> indexesToDispose;

        lock (fieldMap)
        {
            indexesToDispose = new List<LuceneVectorIndex>(fieldMap.Values);
            fieldMap.Clear();
        }

        foreach (var index in indexesToDispose)
            index.Dispose();
    }

    /// <summary>The field and every option value an index is built from.</summary>
    private readonly record struct IndexKey(
        string Field,
        int M,
        int ConstructionPruning,
        int EfSearch,
        VectorDistanceFunction Distance,
        bool ThreadSafe,
        int MinimumGraphSize)
    {
        public IndexKey(string field, VectorIndexOptions options)
            : this(field, options.M, options.ConstructionPruning, options.EfSearch, options.Distance, options.ThreadSafe, options.MinimumGraphSize)
        {
        }

        public VectorIndexOptions ToOptions() => new()
        {
            M = M,
            ConstructionPruning = ConstructionPruning,
            EfSearch = EfSearch,
            Distance = Distance,
            ThreadSafe = ThreadSafe,
            MinimumGraphSize = MinimumGraphSize,
        };
    }

    private sealed class SegmentDisposedListener : SegmentReader.ICoreDisposedListener, IReaderDisposedListener
    {
        public void OnDispose(object ownerCoreCacheKey) => Remove(ownerCoreCacheKey);

        public void OnDispose(IndexReader reader) => Remove(reader.CoreCacheKey);
    }
}
#endif
