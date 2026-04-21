#if NET10_0_OR_GREATER
using System.Runtime.CompilerServices;
using global::Lucene.Net.Index;

namespace Iciclecreek.Lucene.Net.Vector;

/// <summary>
/// Caches HNSW vector indexes per (IndexReader, fieldName) pair.
/// Since IndexReader is an immutable snapshot, the cached graph
/// is valid for the reader's entire lifetime.
/// </summary>
internal static class VectorIndexCache
{
    // ConditionalWeakTable keys on IndexReader identity — when the reader is GC'd,
    // the cache entry (and its HNSW graph) is automatically cleaned up.
    private static readonly ConditionalWeakTable<IndexReader, Dictionary<string, LuceneVectorIndex>> _cache = new();
    private static readonly object _lock = new();
    private static readonly IReaderDisposedListener _readerDisposedListener = new ReaderDisposedListener();

    /// <summary>
    /// Get or build the HNSW index for the given reader and field.
    /// Thread-safe: concurrent callers for the same (reader, field) will block
    /// until the first caller finishes building.
    /// </summary>
    internal static LuceneVectorIndex GetOrBuild(IndexReader reader, string fieldName, VectorIndexOptions? options = null)
    {
        Dictionary<string, LuceneVectorIndex> fieldMap;

        lock (_lock)
        {
            fieldMap = _cache.GetOrCreateValue(reader);
            reader.AddReaderDisposedListener(_readerDisposedListener);
        }

        lock (fieldMap)
        {
            if (fieldMap.TryGetValue(fieldName, out var existing))
                return existing;

            var index = new LuceneVectorIndex(fieldName, options);
            index.BuildIndex(reader);
            fieldMap[fieldName] = index;
            return index;
        }
    }

    private sealed class ReaderDisposedListener : IReaderDisposedListener
    {
        public void OnDispose(IndexReader reader)
        {
            if (!_cache.TryGetValue(reader, out var fieldMap))
                return;

            List<LuceneVectorIndex> indexesToDispose;

            lock (fieldMap)
            {
                indexesToDispose = new List<LuceneVectorIndex>(fieldMap.Count);
                foreach (var index in fieldMap.Values)
                    indexesToDispose.Add(index);

                fieldMap.Clear();
            }

            _cache.Remove(reader);

            foreach (var index in indexesToDispose)
                index.Dispose();
        }
    }
}
#endif
