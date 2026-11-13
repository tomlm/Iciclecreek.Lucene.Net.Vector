#if NET10_0_OR_GREATER
using System.Diagnostics;
using Lucene.Net.Analysis.Standard;
using Lucene.Net.Documents;
using Lucene.Net.Index;
using Lucene.Net.Search;

namespace Iciclecreek.Lucene.Net.Vector.Tests;

/// <summary>
/// Times a KnnVectorQuery on a new index, before and after its graph is ready,
/// and right after one document is added. Run explicitly:
/// <c>dotnet test -c Release -f net10.0 --filter "FullyQualifiedName~KnnSegmentBenchmark" --logger "console;verbosity=detailed"</c>
/// </summary>
[Explicit, Category("Benchmark")]
public class KnnSegmentBenchmark : VectorQueryTestBase
{
    [TestCase(5000)]
    [TestCase(50000)]
    public void QueryAfterAWrite(int count)
    {
        var random = new Random(1);
        float[] Vector()
        {
            var vector = new float[384];
            for (var i = 0; i < vector.Length; i++) vector[i] = (float)(random.NextDouble() * 2 - 1);
            return vector;
        }

        using var analyzer = new StandardAnalyzer(Version);
        using var writer = new IndexWriter(_directory, new IndexWriterConfig(Version, analyzer));

        void Add(string id)
        {
            writer.AddDocument(new Document
            {
                new StringField("id", id, Field.Store.YES),
                new BinaryDocValuesField("embedding", VectorSerializer.ToBytesRef(Vector())),
            });
        }

        for (var i = 0; i < count; i++) Add("d" + i);
        writer.Commit();

        var queryVector = Vector();
        var reader = DirectoryReader.Open(_directory);

        double Time(DirectoryReader r)
        {
            var watch = Stopwatch.StartNew();
            var hits = new IndexSearcher(r).Search(new KnnVectorQuery("embedding", queryVector, 10, r), 10);
            Assert.That(hits.ScoreDocs, Has.Length.EqualTo(10));
            return watch.Elapsed.TotalMilliseconds;
        }

        Console.WriteLine("{0} vectors in {1} segment(s)", count, reader.Leaves.Count);
        Console.WriteLine("  first query on a new index      {0,10:F1} ms (loads the vectors)", Time(reader));
        Console.WriteLine("  repeat, graph still building    {0,10:F1} ms", Enumerable.Range(0, 15).Select(_ => Time(reader)).OrderBy(t => t).ElementAt(7));

        var warmup = Stopwatch.StartNew();
        KnnVectorQuery.Warmup(reader, "embedding");
        Console.WriteLine("  graph ready after a further     {0,10:F1} ms", warmup.Elapsed.TotalMilliseconds);
        Console.WriteLine("  repeat, through the graph       {0,10:F1} ms", Enumerable.Range(0, 15).Select(_ => Time(reader)).OrderBy(t => t).ElementAt(7));

        var afterWrite = new List<double>();
        for (var i = 0; i < 25; i++)
        {
            Add("w" + i);
            writer.Commit();

            var reopened = DirectoryReader.OpenIfChanged(reader);
            reader.Dispose();
            reader = reopened;
            afterWrite.Add(Time(reader));
        }

        afterWrite.Sort();
        Console.WriteLine("  first query after adding one    {0,10:F1} ms median, {1:F1} ms worst of 25 ({2} segments at the end)",
            afterWrite[12], afterWrite[24], reader.Leaves.Count);
        reader.Dispose();
    }
}
#endif
