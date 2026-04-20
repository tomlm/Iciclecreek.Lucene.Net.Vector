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
}
