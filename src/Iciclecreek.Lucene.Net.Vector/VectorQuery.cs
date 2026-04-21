using System;
using global::Lucene.Net.Index;
using global::Lucene.Net.Search;

namespace Iciclecreek.Lucene.Net.Vector;

/// <summary>
/// A vector similarity search query that automatically selects the best
/// implementation for the current runtime:
/// <list type="bullet">
///   <item>.NET 10+: HNSW approximate nearest neighbor via <c>KnnVectorQuery</c></item>
///   <item>Earlier runtimes: brute-force cosine similarity via <c>CosineVectorQuery</c></item>
/// </list>
/// </summary>
public class VectorQuery : Query
{
    private readonly Query _inner;

    public VectorQuery(string field, float[] queryVector, int k, IndexReader reader, VectorIndexOptions options = null)
    {
#if NET10_0_OR_GREATER
        _inner = new KnnVectorQuery(field, queryVector, k, reader, options);
#else
        _inner = new CosineVectorQuery(field, queryVector, k, reader);
#endif
        if (Boost != 1.0f)
            _inner.Boost = Boost;
    }

    public override Weight CreateWeight(IndexSearcher searcher)
    {
        _inner.Boost = Boost;
        return _inner.CreateWeight(searcher);
    }

    public override string ToString(string field)
    {
        return _inner.ToString(field);
    }

    public override bool Equals(object obj)
    {
        if (!(obj is VectorQuery other))
            return false;
        return _inner.Equals(other._inner);
    }

    public override int GetHashCode()
    {
        return _inner.GetHashCode();
    }
}
