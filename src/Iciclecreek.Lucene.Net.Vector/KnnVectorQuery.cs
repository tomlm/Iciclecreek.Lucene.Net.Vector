using global::Lucene.Net.Index;
using global::Lucene.Net.Search;
using global::Lucene.Net.Util;

namespace Iciclecreek.Lucene.Net.Vector;

/// <summary>
/// A Lucene Query that performs approximate nearest neighbor search using an HNSW index.
/// Composes naturally with BooleanQuery for filtered or hybrid search.
/// </summary>
public class KnnVectorQuery : Query
{
    private readonly string _field;
    private readonly float[] _queryVector;
    private readonly int _k;
    private readonly LuceneVectorIndex _vectorIndex;

    public KnnVectorQuery(string field, float[] queryVector, int k, LuceneVectorIndex vectorIndex)
    {
        _field = field ?? throw new ArgumentNullException(nameof(field));
        _queryVector = queryVector ?? throw new ArgumentNullException(nameof(queryVector));
        _k = k;
        _vectorIndex = vectorIndex ?? throw new ArgumentNullException(nameof(vectorIndex));
    }

    public string Field => _field;
    public float[] QueryVector => _queryVector;
    public int K => _k;

    public override Weight CreateWeight(IndexSearcher searcher)
    {
        return new KnnVectorWeight(this, searcher);
    }

    public override string ToString(string field)
    {
        return $"KnnVectorQuery(field={_field}, k={_k}, boost={Boost})";
    }

    public override bool Equals(object? obj)
    {
        if (obj is not KnnVectorQuery other)
            return false;
        return _field == other._field
            && _k == other._k
            && ReferenceEquals(_vectorIndex, other._vectorIndex)
            && _queryVector.AsSpan().SequenceEqual(other._queryVector)
            && Boost == other.Boost;
    }

    public override int GetHashCode()
    {
        int hash = _field.GetHashCode();
        hash = 31 * hash + _k;
        hash = 31 * hash + Boost.GetHashCode();
        if (_queryVector.Length > 0)
            hash = 31 * hash + _queryVector[0].GetHashCode();
        return hash;
    }

    private class KnnVectorWeight : Weight
    {
        private readonly KnnVectorQuery _query;
        private readonly IndexSearcher _searcher;
        private Dictionary<int, float>? _scores;

        public KnnVectorWeight(KnnVectorQuery query, IndexSearcher searcher)
            : base()
        {
            _query = query;
            _searcher = searcher;
        }

        public override Query Query => _query;

        public override Explanation Explain(AtomicReaderContext context, int doc)
        {
            var globalDocId = context.DocBase + doc;
            EnsureScoresComputed();
            if (_scores!.TryGetValue(globalDocId, out var score))
            {
                return new Explanation(score, $"KnnVectorQuery(field={_query._field}, k={_query._k}), distance-based score");
            }
            return new Explanation(0f, "no vector match");
        }

        public override float GetValueForNormalization()
        {
            return 1f;
        }

        public override void Normalize(float norm, float topLevelBoost)
        {
            // Vector scores are self-contained; no normalization needed
        }

        public override Scorer GetScorer(AtomicReaderContext context, IBits? acceptDocs)
        {
            EnsureScoresComputed();

            // Build list of (docId, score) for this segment
            var segmentResults = new List<(int Doc, float Score)>();
            int docBase = context.DocBase;
            int maxDoc = context.AtomicReader.MaxDoc;

            foreach (var (globalDocId, score) in _scores!)
            {
                int localDoc = globalDocId - docBase;
                if (localDoc < 0 || localDoc >= maxDoc)
                    continue;

                if (acceptDocs != null && !acceptDocs.Get(localDoc))
                    continue;

                segmentResults.Add((localDoc, score));
            }

            if (segmentResults.Count == 0)
                return null!;

            segmentResults.Sort((a, b) => a.Doc.CompareTo(b.Doc));
            return new KnnVectorScorer(this, segmentResults);
        }

        private void EnsureScoresComputed()
        {
            if (_scores != null)
                return;

            var results = _query._vectorIndex.Search(
                _query._queryVector,
                _query._k);

            _scores = new Dictionary<int, float>(results.Count);
            float boost = _query.Boost;
            foreach (var result in results)
            {
                _scores[result.DocId] = result.Score * boost;
            }
        }
    }

    private class KnnVectorScorer : Scorer
    {
        private readonly List<(int Doc, float Score)> _results;
        private int _currentIndex = -1;

        public KnnVectorScorer(Weight weight, List<(int Doc, float Score)> results)
            : base(weight)
        {
            _results = results;
        }

        public override int DocID =>
            _currentIndex < 0 ? -1 :
            _currentIndex >= _results.Count ? NO_MORE_DOCS :
            _results[_currentIndex].Doc;

        public override float GetScore()
        {
            return _results[_currentIndex].Score;
        }

        public override int NextDoc()
        {
            _currentIndex++;
            return _currentIndex >= _results.Count ? NO_MORE_DOCS : _results[_currentIndex].Doc;
        }

        public override int Advance(int target)
        {
            while (_currentIndex < _results.Count)
            {
                _currentIndex++;
                if (_currentIndex >= _results.Count)
                    return NO_MORE_DOCS;
                if (_results[_currentIndex].Doc >= target)
                    return _results[_currentIndex].Doc;
            }
            return NO_MORE_DOCS;
        }

        public override long GetCost()
        {
            return _results.Count;
        }

        public override int Freq => 1;
    }
}
