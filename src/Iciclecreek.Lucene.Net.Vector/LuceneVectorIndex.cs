using HNSW.Net;
using global::Lucene.Net.Index;
using global::Lucene.Net.Search;
using global::Lucene.Net.Util;

namespace Iciclecreek.Lucene.Net.Vector;

/// <summary>
/// Manages an in-memory HNSW graph for approximate nearest neighbor search
/// over vectors stored as BinaryDocValues in a Lucene index.
/// Built and cached automatically per (IndexReader, fieldName) via VectorIndexCache.
/// </summary>
internal class LuceneVectorIndex : IDisposable
{
    private SmallWorld<float[], float>? _graph;
    private readonly Dictionary<int, int> _graphIdToDocId = new();
    private readonly Dictionary<float[], int> _vectorToDocId = new(ReferenceEqualityComparer.Instance);
    private readonly string _vectorFieldName;
    private readonly VectorIndexOptions _options;
    private readonly Func<float[], float[], float> _distanceFunc;
    private List<float[]> _vectors = new();
    private bool _disposed;

    internal LuceneVectorIndex(string vectorFieldName, VectorIndexOptions? options = null)
    {
        _vectorFieldName = vectorFieldName ?? throw new ArgumentNullException(nameof(vectorFieldName));
        _options = options ?? new VectorIndexOptions();
        _distanceFunc = GetDistanceFunction(_options.Distance);
    }

    internal int Count => _vectors.Count;
    internal int Dimensions => _vectors.Count > 0 ? _vectors[0].Length : 0;

    /// <summary>
    /// Build the HNSW graph from all BinaryDocValues in the given reader.
    /// </summary>
    internal void BuildIndex(IndexReader reader)
    {
        ArgumentNullException.ThrowIfNull(reader);
        _graphIdToDocId.Clear();
        _vectorToDocId.Clear();
        _vectors.Clear();

        var leaves = reader.Leaves;
        foreach (var leaf in leaves)
        {
            var atomicReader = leaf.AtomicReader;
            var docValues = atomicReader.GetBinaryDocValues(_vectorFieldName);
            if (docValues == null)
                continue;

            var liveDocs = atomicReader.LiveDocs;
            for (int docId = 0; docId < atomicReader.MaxDoc; docId++)
            {
                if (liveDocs != null && !liveDocs.Get(docId))
                    continue;

                var bytesRef = new BytesRef();
                docValues.Get(docId, bytesRef);

                if (bytesRef.Length == 0)
                    continue;

                var vector = VectorSerializer.FromBytesRef(bytesRef);

                if (_vectors.Count > 0 && vector.Length != _vectors[0].Length)
                    throw new InvalidOperationException(
                        $"Dimension mismatch in field '{_vectorFieldName}': document {leaf.DocBase + docId} has {vector.Length} dimensions but expected {_vectors[0].Length}.");

                var globalDocId = leaf.DocBase + docId;
                var graphId = _vectors.Count;

                _vectors.Add(vector);
                _graphIdToDocId[graphId] = globalDocId;
                _vectorToDocId[vector] = globalDocId;
            }
        }

        RebuildGraph();
    }

    /// <summary>
    /// Search for the top-K nearest neighbors to the query vector.
    /// </summary>
    internal IReadOnlyList<SearchResult> Search(float[] queryVector, int topK)
    {
        return SearchCore(queryVector, topK);
    }

    private IReadOnlyList<SearchResult> SearchCore(float[] queryVector, int topK)
    {
        if (_graph == null)
            throw new InvalidOperationException("Index has not been built.");

        if (_vectors.Count == 0)
            return Array.Empty<SearchResult>();

        if (queryVector.Length != Dimensions)
            throw new ArgumentException(
                $"Query vector has {queryVector.Length} dimensions but the index contains {Dimensions}-dimensional vectors.",
                nameof(queryVector));

        Func<float[], bool> itemFilter = vector => _vectorToDocId.ContainsKey(vector);

        var results = _graph.KNNSearch(queryVector, topK, itemFilter, CancellationToken.None);

        return results
            .Where(r => _graphIdToDocId.ContainsKey(r.Id))
            .Select(r => new SearchResult(
                _graphIdToDocId[r.Id],
                r.Distance))
            .OrderBy(r => r.Distance)
            .ToList();
    }

    private void RebuildGraph()
    {
        var parameters = CreateParameters();
        _graph = new SmallWorld<float[], float>(
            _distanceFunc,
            DefaultRandomGenerator.Instance,
            parameters,
            _options.ThreadSafe);

        if (_vectors.Count > 0)
            _graph.AddItems(_vectors, progressReporter: null);
    }

    private SmallWorldParameters CreateParameters()
    {
        return new SmallWorldParameters
        {
            M = _options.M,
            ConstructionPruning = _options.ConstructionPruning,
            EfSearch = _options.EfSearch,
            EnableDistanceCacheForConstruction = true,
        };
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
        if (!_disposed)
        {
            _graph = null;
            _vectors.Clear();
            _graphIdToDocId.Clear();
            _vectorToDocId.Clear();
            _disposed = true;
        }
    }
}

/// <summary>
/// Result of a vector similarity search.
/// </summary>
public readonly record struct SearchResult(int DocId, float Distance)
{
    /// <summary>
    /// Converts the distance to a Lucene-compatible relevance score.
    /// Higher scores indicate more similar documents.
    /// </summary>
    public float Score => 1f / (1f + Distance);
}
