using HNSW.Net;
using global::Lucene.Net.Index;
using global::Lucene.Net.Search;
using global::Lucene.Net.Util;

namespace Iciclecreek.Lucene.Net.Vector;

/// <summary>
/// Manages an in-memory HNSW graph for approximate nearest neighbor search
/// over vectors stored as BinaryDocValues in a Lucene index.
/// </summary>
public class LuceneVectorIndex : IDisposable
{
    private SmallWorld<float[], float>? _graph;
    private readonly Dictionary<int, int> _docIdToGraphId = new();
    private readonly Dictionary<int, int> _graphIdToDocId = new();
    private readonly Dictionary<float[], int> _vectorToDocId = new(ReferenceEqualityComparer.Instance);
    private readonly string _vectorFieldName;
    private readonly VectorIndexOptions _options;
    private readonly Func<float[], float[], float> _distanceFunc;
    private IndexReader? _reader;
    private List<float[]> _vectors = new();
    private bool _disposed;

    public LuceneVectorIndex(string vectorFieldName, VectorIndexOptions? options = null)
    {
        _vectorFieldName = vectorFieldName ?? throw new ArgumentNullException(nameof(vectorFieldName));
        _options = options ?? new VectorIndexOptions();
        _distanceFunc = GetDistanceFunction(_options.Distance);
    }

    /// <summary>
    /// The name of the vector field in the Lucene index.
    /// </summary>
    public string VectorFieldName => _vectorFieldName;

    /// <summary>
    /// The number of vectors currently in the index.
    /// </summary>
    public int Count => _vectors.Count;

    /// <summary>
    /// Build the HNSW graph from all BinaryDocValues in the given reader.
    /// </summary>
    public void BuildIndex(IndexReader reader)
    {
        _reader = reader ?? throw new ArgumentNullException(nameof(reader));
        _docIdToGraphId.Clear();
        _graphIdToDocId.Clear();
        _vectorToDocId.Clear();
        _vectors.Clear();

        // Read all vectors from BinaryDocValues across all atomic readers
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
                var globalDocId = leaf.DocBase + docId;
                var graphId = _vectors.Count;

                _vectors.Add(vector);
                _docIdToGraphId[globalDocId] = graphId;
                _graphIdToDocId[graphId] = globalDocId;
                _vectorToDocId[vector] = globalDocId;
            }
        }

        RebuildGraph();
    }

    /// <summary>
    /// Add a vector for a document. Call after adding the document to the Lucene index.
    /// </summary>
    public void AddVector(int docId, float[] vector)
    {
        if (_graph == null)
            throw new InvalidOperationException("Index has not been built. Call BuildIndex first.");

        var graphId = _vectors.Count;
        _vectors.Add(vector);
        _docIdToGraphId[docId] = graphId;
        _graphIdToDocId[graphId] = docId;
        _vectorToDocId[vector] = docId;

        _graph.AddItems(new[] { vector }, progressReporter: null);
    }

    /// <summary>
    /// Mark a document's vector as removed. The vector remains in the graph
    /// but is excluded from search results.
    /// </summary>
    public void RemoveVector(int docId)
    {
        if (_docIdToGraphId.Remove(docId, out var graphId))
        {
            if (graphId < _vectors.Count)
                _vectorToDocId.Remove(_vectors[graphId]);
            _graphIdToDocId.Remove(graphId);
        }
    }

    /// <summary>
    /// Search for the top-K nearest neighbors to the query vector.
    /// </summary>
    public IReadOnlyList<SearchResult> Search(float[] queryVector, int topK)
    {
        return SearchCore(queryVector, topK, filterDocId: null);
    }

    /// <summary>
    /// Search for the top-K nearest neighbors, restricted to documents matching the filter.
    /// </summary>
    public IReadOnlyList<SearchResult> Search(float[] queryVector, int topK, Filter filter, IndexSearcher searcher)
    {
        if (filter == null)
            return Search(queryVector, topK);

        // Collect matching doc IDs from the filter
        var acceptedDocs = new HashSet<int>();
        foreach (var leaf in searcher.IndexReader.Leaves)
        {
            var docIdSet = filter.GetDocIdSet(leaf, leaf.AtomicReader.LiveDocs);
            if (docIdSet == null)
                continue;

            var iterator = docIdSet.GetIterator();
            if (iterator == null)
                continue;

            int doc;
            while ((doc = iterator.NextDoc()) != DocIdSetIterator.NO_MORE_DOCS)
            {
                acceptedDocs.Add(leaf.DocBase + doc);
            }
        }

        return SearchCore(queryVector, topK, docId => acceptedDocs.Contains(docId));
    }

    /// <summary>
    /// Search restricted to a specific set of document IDs.
    /// </summary>
    public IReadOnlyList<SearchResult> Search(float[] queryVector, int topK, ISet<int> acceptedDocIds)
    {
        return SearchCore(queryVector, topK, docId => acceptedDocIds.Contains(docId));
    }

    /// <summary>
    /// Serialize the HNSW graph to a stream for persistence.
    /// </summary>
    public void Serialize(Stream stream)
    {
        if (_graph == null)
            throw new InvalidOperationException("Index has not been built.");

        // Write mapping data first
        using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);

        // Write vector count and dimension
        writer.Write(_vectors.Count);
        writer.Write(_vectors.Count > 0 ? _vectors[0].Length : 0);

        // Write vectors
        foreach (var vector in _vectors)
        {
            var bytes = VectorSerializer.ToBytes(vector);
            writer.Write(bytes);
        }

        // Write doc ID mappings
        writer.Write(_docIdToGraphId.Count);
        foreach (var (docId, graphId) in _docIdToGraphId)
        {
            writer.Write(docId);
            writer.Write(graphId);
        }

        // Write HNSW graph
        _graph.SerializeGraph(stream);
    }

    /// <summary>
    /// Deserialize an HNSW graph from a stream.
    /// </summary>
    public static LuceneVectorIndex Deserialize(Stream stream, string vectorFieldName, VectorIndexOptions? options = null)
    {
        var index = new LuceneVectorIndex(vectorFieldName, options);

        using var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);

        // Read vectors
        var vectorCount = reader.ReadInt32();
        var dimensions = reader.ReadInt32();

        for (int i = 0; i < vectorCount; i++)
        {
            var bytes = reader.ReadBytes(dimensions * sizeof(float));
            index._vectors.Add(VectorSerializer.FromBytes(bytes));
        }

        // Read doc ID mappings
        var mappingCount = reader.ReadInt32();
        for (int i = 0; i < mappingCount; i++)
        {
            var docId = reader.ReadInt32();
            var graphId = reader.ReadInt32();
            index._docIdToGraphId[docId] = graphId;
            index._graphIdToDocId[graphId] = docId;
            if (graphId < index._vectors.Count)
                index._vectorToDocId[index._vectors[graphId]] = docId;
        }

        // Deserialize HNSW graph
        var parameters = index.CreateParameters();
        var (graph, _) = SmallWorld<float[], float>.DeserializeGraph(
            index._vectors,
            index._distanceFunc,
            DefaultRandomGenerator.Instance,
            stream,
            index._options.ThreadSafe);
        index._graph = graph;

        return index;
    }

    private IReadOnlyList<SearchResult> SearchCore(float[] queryVector, int topK, Func<int, bool>? filterDocId)
    {
        if (_graph == null)
            throw new InvalidOperationException("Index has not been built. Call BuildIndex first.");

        if (_vectors.Count == 0)
            return Array.Empty<SearchResult>();

        // HNSW KNNSearch filter receives the vector item (float[]).
        // We use reference equality via _vectorToDocId to map back to doc IDs.
        Func<float[], bool> itemFilter;
        if (filterDocId != null)
        {
            itemFilter = vector =>
            {
                if (!_vectorToDocId.TryGetValue(vector, out var docId))
                    return false;
                return filterDocId(docId);
            };
        }
        else
        {
            // Still need to filter out removed vectors
            itemFilter = vector => _vectorToDocId.ContainsKey(vector);
        }

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
        if (_vectors.Count == 0)
        {
            _graph = new SmallWorld<float[], float>(
                _distanceFunc,
                DefaultRandomGenerator.Instance,
                CreateParameters(),
                _options.ThreadSafe);
            return;
        }

        var parameters = CreateParameters();
        _graph = new SmallWorld<float[], float>(
            _distanceFunc,
            DefaultRandomGenerator.Instance,
            parameters,
            _options.ThreadSafe);

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
        return distance switch
        {
            VectorDistanceFunction.Cosine => CosineDistance.SIMD,
            VectorDistanceFunction.CosineForUnits => CosineDistance.SIMDForUnits,
            _ => throw new ArgumentOutOfRangeException(nameof(distance))
        };
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _graph = null;
            _vectors.Clear();
            _docIdToGraphId.Clear();
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
