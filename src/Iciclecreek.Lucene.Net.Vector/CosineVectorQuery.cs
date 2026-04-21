using System;
using System.Collections.Generic;
using System.Numerics;
using global::Lucene.Net.Index;
using global::Lucene.Net.Search;
using global::Lucene.Net.Util;

namespace Iciclecreek.Lucene.Net.Vector;

/// <summary>
/// A Lucene Query that performs exact nearest neighbor search using
/// brute-force cosine similarity over vectors stored as BinaryDocValuesField.
/// Uses SIMD acceleration via <see cref="Vector{T}"/> when available.
///
/// For approximate (faster) search on .NET 10+, use <see cref="VectorQuery"/>
/// which automatically selects HNSW when available.
/// </summary>
public class CosineVectorQuery : Query
{
    private readonly string _field;
    private readonly float[] _queryVector;
    private readonly int _k;
    private readonly IndexReader _reader;

    public CosineVectorQuery(string field, float[] queryVector, int k, IndexReader reader)
    {
        _field = field ?? throw new ArgumentNullException(nameof(field));
        _queryVector = queryVector ?? throw new ArgumentNullException(nameof(queryVector));
        _k = k;
        _reader = reader ?? throw new ArgumentNullException(nameof(reader));
    }

    public string Field => _field;
    public float[] QueryVector => _queryVector;
    public int K => _k;

    public override Weight CreateWeight(IndexSearcher searcher)
    {
        return new CosineVectorWeight(this, searcher);
    }

    public override string ToString(string field)
    {
        return $"CosineVectorQuery(field={_field}, k={_k}, boost={Boost})";
    }

    public override bool Equals(object obj)
    {
        if (!(obj is CosineVectorQuery other))
            return false;
        return _field == other._field
            && _k == other._k
            && Boost == other.Boost;
    }

    public override int GetHashCode()
    {
        int hash = _field.GetHashCode();
        hash = 31 * hash + _k;
        hash = 31 * hash + Boost.GetHashCode();
        return hash;
    }

    private class CosineVectorWeight : Weight
    {
        private readonly CosineVectorQuery _query;
        private readonly IndexSearcher _searcher;
        private Dictionary<int, float> _scores;

        public CosineVectorWeight(CosineVectorQuery query, IndexSearcher searcher)
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
            if (_scores.TryGetValue(globalDocId, out var score))
            {
                return new Explanation(score, $"CosineVectorQuery(field={_query._field}, k={_query._k}), cosine similarity score");
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

        public override Scorer GetScorer(AtomicReaderContext context, IBits acceptDocs)
        {
            EnsureScoresComputed();

            var segmentResults = new List<(int Doc, float Score)>();
            int docBase = context.DocBase;
            int maxDoc = context.AtomicReader.MaxDoc;

            foreach (var kvp in _scores)
            {
                int localDoc = kvp.Key - docBase;
                if (localDoc < 0 || localDoc >= maxDoc)
                    continue;

                if (acceptDocs != null && !acceptDocs.Get(localDoc))
                    continue;

                segmentResults.Add((localDoc, kvp.Value));
            }

            if (segmentResults.Count == 0)
                return null;

            segmentResults.Sort((a, b) => a.Doc.CompareTo(b.Doc));
            return new CosineVectorScorer(this, segmentResults);
        }

        private void EnsureScoresComputed()
        {
            if (_scores != null)
                return;

            var queryVector = _query._queryVector;
            var queryNorm = VectorMath.Norm(queryVector);
            var boost = _query.Boost;

            // Collect all (docId, distance) pairs
            var allResults = new List<SearchResult>();

            var leaves = _query._reader.Leaves;
            foreach (var leaf in leaves)
            {
                var atomicReader = leaf.AtomicReader;
                var docValues = atomicReader.GetBinaryDocValues(_query._field);
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

                    var docVector = VectorSerializer.FromBytesRef(bytesRef);
                    if (docVector.Length != queryVector.Length)
                        continue;

                    var similarity = VectorMath.CosineSimilarity(queryVector, docVector, queryNorm);
                    var distance = 1f - similarity;
                    var globalDocId = leaf.DocBase + docId;

                    allResults.Add(new SearchResult(globalDocId, distance));
                }
            }

            // Sort by distance ascending (most similar first), take top K
            allResults.Sort((a, b) => a.Distance.CompareTo(b.Distance));
            var topK = Math.Min(_query._k, allResults.Count);

            _scores = new Dictionary<int, float>(topK);
            for (int i = 0; i < topK; i++)
            {
                _scores[allResults[i].DocId] = allResults[i].Score * boost;
            }
        }
    }

    private class CosineVectorScorer : Scorer
    {
        private readonly List<(int Doc, float Score)> _results;
        private int _currentIndex = -1;

        public CosineVectorScorer(Weight weight, List<(int Doc, float Score)> results)
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
