#if NET10_0_OR_GREATER
using System.Runtime.CompilerServices;
using global::Lucene.Net.Index;

namespace Iciclecreek.Lucene.Net.Vector;

/// <summary>
/// Caches vector indexes per (segment, fieldName) pair.
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
/// </remarks>
internal static class VectorIndexCache
{
    // ConditionalWeakTable keys on the identity of the segment's core key — when
    // the key is GC'd, the cache entry is automatically cleaned up. Its HNSW graph is
    // only let go when the segment is disposed (see LuceneVectorIndex.ReleaseGraph),
    // so a reader that is dropped without being disposed leaves its graphs behind.
    private static readonly ConditionalWeakTable<object, Dictionary<string, LuceneVectorIndex>> _cache = new();
    private static readonly object _lock = new();
    private static readonly SegmentDisposedListener _disposedListener = new();

    /// <summary>
    /// Get or build the vector index for the given segment and field.
    /// Thread-safe: concurrent callers for the same (segment, field) will block
    /// until the first caller finishes building.
    /// </summary>
    internal static LuceneVectorIndex GetOrBuild(AtomicReader segment, string fieldName, VectorIndexOptions? options = null)
    {
        var key = segment.CoreCacheKey;
        Dictionary<string, LuceneVectorIndex>? fieldMap;

        lock (_lock)
        {
            if (!_cache.TryGetValue(key, out fieldMap))
            {
                fieldMap = new Dictionary<string, LuceneVectorIndex>();
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
            if (fieldMap.TryGetValue(fieldName, out var existing))
                return existing;

            var index = new LuceneVectorIndex(fieldName, options);
            index.BuildIndex(segment);
            fieldMap[fieldName] = index;
            return index;
        }
    }

    private static void Remove(object key)
    {
        Dictionary<string, LuceneVectorIndex>? fieldMap;

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

    private sealed class SegmentDisposedListener : SegmentReader.ICoreDisposedListener, IReaderDisposedListener
    {
        public void OnDispose(object ownerCoreCacheKey) => Remove(ownerCoreCacheKey);

        public void OnDispose(IndexReader reader) => Remove(reader.CoreCacheKey);
    }
}
#endif
