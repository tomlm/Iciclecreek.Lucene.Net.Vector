#if NET10_0_OR_GREATER
using global::Lucene.Net.Index;
using global::Lucene.Net.Search;
using global::Lucene.Net.Util;

namespace Iciclecreek.Lucene.Net.Vector;

/// <summary>
/// A Lucene Query that performs approximate nearest neighbor search over vectors
/// stored as BinaryDocValuesField. The HNSW index is built and cached automatically
/// per (index segment, fieldName) — no manual index management required. Segments
/// do not change once written, so a reader opened after a commit reuses the graphs
/// of the segments it shares with the reader before it and builds only the new ones.
/// Graphs are built in the background; until a segment's graph is ready its vectors
/// are compared one by one, so a query never waits for a build.
///
/// Composes naturally with BooleanQuery for filtered or hybrid search.
/// </summary>
public class KnnVectorQuery : Query
{
    private readonly string _field;
    private readonly float[] _queryVector;
    private readonly int _k;
    private readonly IndexReader _reader;
    private readonly VectorIndexOptions? _options;

    public KnnVectorQuery(string field, float[] queryVector, int k, IndexReader reader, VectorIndexOptions? options = null)
    {
        _field = field ?? throw new ArgumentNullException(nameof(field));
        _queryVector = queryVector ?? throw new ArgumentNullException(nameof(queryVector));
        _k = k;
        _reader = reader ?? throw new ArgumentNullException(nameof(reader));
        _options = options;
    }

    public string Field => _field;
    public float[] QueryVector => _queryVector;
    public int K => _k;

    /// <summary>
    /// Build the HNSW index for every segment of the given reader and field, and wait
    /// until it is ready. Without this the graphs are built in the background and
    /// queries compare vectors one by one in the meantime, which is exact but slower
    /// on a large index. Safe to call multiple times — subsequent calls are no-ops
    /// for segments that are already built.
    /// </summary>
    public static void Warmup(IndexReader reader, string field, VectorIndexOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(reader);

        // Start every segment, then wait for each.
        var indexes = new List<LuceneVectorIndex>();
        foreach (var leaf in reader.Leaves)
            indexes.Add(VectorIndexCache.GetOrBuild(leaf.AtomicReader, field, options));

        foreach (var index in indexes)
            index.WaitForGraph();
    }

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
            && ReferenceEquals(_reader, other._reader)
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
        }

        public override Scorer GetScorer(AtomicReaderContext context, IBits? acceptDocs)
        {
            EnsureScoresComputed();

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

            // The nearest K of each segment, then the nearest K of those.
            var nearest = new List<SearchResult>();
            foreach (var leaf in _query._reader.Leaves)
            {
                var segment = leaf.AtomicReader;
                var vectorIndex = VectorIndexCache.GetOrBuild(segment, _query._field, _query._options);

                foreach (var result in vectorIndex.Search(_query._queryVector, _query._k, segment.LiveDocs))
                {
                    nearest.Add(new SearchResult(leaf.DocBase + result.DocId, result.Distance));
                }
            }

            nearest.Sort((a, b) => a.Distance.CompareTo(b.Distance));

            var scores = new Dictionary<int, float>(Math.Min(nearest.Count, Math.Max(_query._k, 0)));
            float boost = _query.Boost;
            for (int i = 0; i < nearest.Count && i < _query._k; i++)
            {
                scores[nearest[i].DocId] = nearest[i].Score * boost;
            }

            _scores = scores;
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
#endif
