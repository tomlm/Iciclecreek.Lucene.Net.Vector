using global::Lucene.Net.Index;
using global::Lucene.Net.Queries;
using global::Lucene.Net.Search;

namespace Iciclecreek.Lucene.Net.Vector;

/// <summary>
/// A query that re-scores documents matched by a filter query using cosine similarity
/// between a query vector and vectors stored as BinaryDocValuesField.
/// No top-K or HNSW — the filter sub-query controls which docs match, cosine controls ranking.
/// </summary>
public class VectorScoreQuery : CustomScoreQuery
{
    private readonly string _vectorFieldName;
    private readonly float[] _queryVector;
    private readonly Query _filterQuery;

    public VectorScoreQuery(Query filterQuery, string vectorFieldName, float[] queryVector)
        : base(filterQuery)
    {
        _filterQuery = filterQuery ?? throw new ArgumentNullException(nameof(filterQuery));
        _vectorFieldName = vectorFieldName ?? throw new ArgumentNullException(nameof(vectorFieldName));
        _queryVector = queryVector ?? throw new ArgumentNullException(nameof(queryVector));
    }

    public string VectorFieldName => _vectorFieldName;
    public float[] QueryVector => _queryVector;

    protected override CustomScoreProvider GetCustomScoreProvider(AtomicReaderContext context)
    {
        return new VectorScoreProvider(context, _vectorFieldName, _queryVector);
    }

    public override string ToString(string field)
    {
        return $"VectorScoreQuery(field={_vectorFieldName}, subQuery={_filterQuery}, boost={Boost})";
    }

    private class VectorScoreProvider : CustomScoreProvider
    {
        private readonly AtomicReaderContext _context;
        private readonly string _vectorFieldName;
        private readonly float[] _queryVector;
        private readonly float _queryNorm;

        // One provider scores one segment, on one thread. The vector column and
        // the buffer are fetched once and reused for every document, instead of
        // being looked up and allocated per document.
        private readonly global::Lucene.Net.Util.BytesRef _bytes = new global::Lucene.Net.Util.BytesRef();
        private BinaryDocValues? _vectors;
        private bool _vectorsLoaded;

        public VectorScoreProvider(AtomicReaderContext context, string vectorFieldName, float[] queryVector)
            : base(context)
        {
            _context = context;
            _vectorFieldName = vectorFieldName;
            _queryVector = queryVector;
            _queryNorm = VectorMath.Norm(queryVector);
        }

        public override float CustomScore(int doc, float subQueryScore, float valSrcScore)
        {
            if (!_vectorsLoaded)
            {
                // Loaded on first use so it happens on the thread that scores.
                _vectors = _context.AtomicReader.GetBinaryDocValues(_vectorFieldName);
                _vectorsLoaded = true;
            }

            if (_vectors == null)
                return 0f;

            _vectors.Get(doc, _bytes);

            if (_bytes.Length == 0)
                return 0f;

            var docVector = VectorSerializer.FromBytesRef(_bytes);
            var similarity = VectorMath.CosineSimilarity(_queryVector, docVector, _queryNorm);
            return (similarity + 1f) / 2f;
        }
    }
}
