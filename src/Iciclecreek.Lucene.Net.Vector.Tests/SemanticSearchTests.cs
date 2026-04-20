using Iciclecreek.Lucene.Net.Vector;
using Lucene.Net.Analysis.Standard;
using Lucene.Net.Documents;
using Lucene.Net.Index;
using Lucene.Net.Search;
using Lucene.Net.Store;
using Lucene.Net.Util;
using Microsoft.Extensions.AI;

namespace Iciclecreek.Lucene.Net.Vector.Tests;

/// <summary>
/// End-to-end tests using real embeddings via the standard IEmbeddingGenerator interface,
/// backed by SmartComponents.LocalEmbeddings (bge-micro-v2 model).
/// </summary>
public class SemanticSearchTests
{
    private RAMDirectory _directory = null!;
    private IEmbeddingGenerator<string, Embedding<float>> _generator = null!;
    private const LuceneVersion Version = LuceneVersion.LUCENE_48;

    [OneTimeSetUp]
    public void OneTimeSetUp()
    {
        _generator = new LocalEmbedderGenerator();
    }

    [OneTimeTearDown]
    public void OneTimeTearDown()
    {
        _generator?.Dispose();
    }

    [SetUp]
    public void Setup()
    {
        _directory = new RAMDirectory();
    }

    [TearDown]
    public void TearDown()
    {
        _directory?.Dispose();
    }

    private async Task<float[]> GetVectors(string text)
    {
        var result = await _generator.GenerateAsync([text]);
        return result[0].Vector.ToArray();
    }

    private async Task IndexDocumentsAsync(params (string id, string text, string? category)[] docs)
    {
        using var analyzer = new StandardAnalyzer(Version);
        var config = new IndexWriterConfig(Version, analyzer);
        using var writer = new IndexWriter(_directory, config);

        var texts = docs.Select(d => d.text).ToList();
        var embeddings = await _generator.GenerateAsync(texts);

        for (int i = 0; i < docs.Length; i++)
        {
            var (id, text, category) = docs[i];
            var vector = embeddings[i].Vector.ToArray();
            var doc = new Document
            {
                new StringField("id", id, Field.Store.YES),
                new TextField("text", text, Field.Store.YES),
                new BinaryDocValuesField("embedding", VectorSerializer.ToBytesRef(vector)),
            };
            if (category != null)
                doc.Add(new StringField("category", category, Field.Store.YES));
            writer.AddDocument(doc);
        }
        writer.Commit();
    }

    [Test]
    public async Task SemanticSearch_FindsSimilarDocuments()
    {
        await IndexDocumentsAsync(
            ("1", "The cat sat on the mat", null),
            ("2", "A dog played in the park", null),
            ("3", "Machine learning models process data", null),
            ("4", "The kitten rested on the rug", null),
            ("5", "Neural networks learn from examples", null));

        using var reader = DirectoryReader.Open(_directory);
        var searcher = new IndexSearcher(reader);
        var queryVector = await GetVectors("A small cat sleeping on a carpet");
        var query = new KnnVectorQuery("embedding", queryVector, 5, reader);

        var topDocs = searcher.Search(query, 2);
        var topTexts = topDocs.ScoreDocs
            .Select(sd => searcher.Doc(sd.Doc).Get("text"))
            .ToList();

        Assert.That(topTexts, Has.Some.Contain("cat").Or.Contain("kitten"),
            $"Expected cat/kitten docs in top results, got: {string.Join(", ", topTexts)}");
    }

    [Test]
    public async Task SemanticSearch_TechQueriesMatchTechDocs()
    {
        await IndexDocumentsAsync(
            ("1", "The weather is sunny and warm today", null),
            ("2", "Python is a popular programming language", null),
            ("3", "Fresh vegetables are good for health", null),
            ("4", "JavaScript frameworks like React build user interfaces", null),
            ("5", "Cooking pasta requires boiling water", null));

        using var reader = DirectoryReader.Open(_directory);
        var searcher = new IndexSearcher(reader);
        var queryVector = await GetVectors("software development and coding");
        var query = new KnnVectorQuery("embedding", queryVector, 2, reader);

        var topDocs = searcher.Search(query, 2);
        var topTexts = topDocs.ScoreDocs
            .Select(sd => searcher.Doc(sd.Doc).Get("text"))
            .ToList();

        Assert.That(topTexts, Has.Some.Contain("Python").Or.Contain("JavaScript"),
            $"Expected programming docs in top results, got: {string.Join(", ", topTexts)}");
    }

    [Test]
    public async Task KnnVectorQuery_WithRealEmbeddings_ReturnsSemanticResults()
    {
        await IndexDocumentsAsync(
            ("1", "The stock market crashed today", "finance"),
            ("2", "Scientists discovered a new species of frog", "science"),
            ("3", "The economy is showing signs of recovery", "finance"),
            ("4", "A new planet was found orbiting a distant star", "science"),
            ("5", "Investors are worried about inflation rates", "finance"));

        using var reader = DirectoryReader.Open(_directory);
        var searcher = new IndexSearcher(reader);
        var queryVector = await GetVectors("financial markets and economic trends");
        var query = new KnnVectorQuery("embedding", queryVector, 5, reader);

        var topDocs = searcher.Search(query, 3);
        var categories = topDocs.ScoreDocs
            .Select(sd => searcher.Doc(sd.Doc).Get("category"))
            .ToList();

        Assert.That(categories, Is.All.EqualTo("finance"),
            $"Expected all finance docs in top 3, got: {string.Join(", ", categories)}");
    }

    [Test]
    public async Task KnnVectorQuery_FilteredByCategory_WithRealEmbeddings()
    {
        await IndexDocumentsAsync(
            ("1", "The stock market crashed today", "finance"),
            ("2", "Scientists discovered a new species of frog", "science"),
            ("3", "A new planet was found orbiting a distant star", "science"),
            ("4", "DNA sequencing reveals evolutionary patterns", "science"),
            ("5", "Investors are worried about inflation rates", "finance"));

        using var reader = DirectoryReader.Open(_directory);
        var searcher = new IndexSearcher(reader);
        var queryVector = await GetVectors("recent discoveries in biology and astronomy");

        var boolQuery = new BooleanQuery
        {
            { new TermQuery(new Term("category", "science")), Occur.MUST },
            { new KnnVectorQuery("embedding", queryVector, 5, reader), Occur.MUST }
        };

        var topDocs = searcher.Search(boolQuery, 10);

        Assert.That(topDocs.TotalHits, Is.EqualTo(3));
        foreach (var scoreDoc in topDocs.ScoreDocs)
        {
            var doc = searcher.Doc(scoreDoc.Doc);
            Assert.That(doc.Get("category"), Is.EqualTo("science"));
        }
    }

    [Test]
    public async Task EmbeddingsAreConsistent_SameTextProducesSameVector()
    {
        var text = "The quick brown fox jumps over the lazy dog";
        var vec1 = await GetVectors(text);
        var vec2 = await GetVectors(text);

        Assert.That(vec1, Is.EqualTo(vec2));
    }

    [Test]
    public async Task EmbeddingsAreDifferent_ForDifferentText()
    {
        var vec1 = await GetVectors("I love programming in C#");
        var vec2 = await GetVectors("The sunset was beautiful over the ocean");

        Assert.That(vec1, Is.Not.EqualTo(vec2));
    }

    [Test]
    public async Task BatchEmbedding_ProducesOneEmbeddingPerInput()
    {
        var texts = new[] { "hello", "world", "foo" };
        var embeddings = await _generator.GenerateAsync(texts);

        Assert.That(embeddings, Has.Count.EqualTo(3));
        Assert.That(embeddings[0].Vector.Length, Is.GreaterThan(0));
    }

    [Test]
    public async Task KnnVectorQuery_RanksResultsBySimilarity()
    {
        await IndexDocumentsAsync(
            ("1", "The president signed a new trade agreement with China", "politics"),
            ("2", "Apple released a new iPhone with improved camera", "tech"),
            ("3", "Congress debated the new healthcare reform bill", "politics"),
            ("4", "Google announced advances in quantum computing", "tech"),
            ("5", "The senator gave a speech on immigration policy", "politics"));

        using var reader = DirectoryReader.Open(_directory);
        var searcher = new IndexSearcher(reader);
        var queryVector = await GetVectors("government legislation and policy");
        var query = new KnnVectorQuery("embedding", queryVector, 5, reader);

        var topDocs = searcher.Search(query, 5);

        for (int i = 0; i < topDocs.ScoreDocs.Length - 1; i++)
        {
            Assert.That(topDocs.ScoreDocs[i].Score,
                Is.GreaterThanOrEqualTo(topDocs.ScoreDocs[i + 1].Score),
                $"Results should be ranked by similarity score (index {i} vs {i + 1})");
        }

        var rankedCategories = topDocs.ScoreDocs
            .Select(sd => searcher.Doc(sd.Doc).Get("category"))
            .ToList();

        var firstTechIndex = rankedCategories.IndexOf("tech");
        var lastPoliticsIndex = rankedCategories.LastIndexOf("politics");

        Assert.That(lastPoliticsIndex, Is.LessThan(firstTechIndex),
            $"Politics docs should rank above tech docs, got: {string.Join(", ", rankedCategories)}");
    }

    [Test]
    public async Task HybridQuery_CombinesTextFilterWithSemanticRanking()
    {
        await IndexDocumentsAsync(
            ("1", "Python web frameworks like Django and Flask", "programming"),
            ("2", "Machine learning with Python and TensorFlow", "programming"),
            ("3", "JavaScript frameworks like React and Vue", "programming"),
            ("4", "The python snake is found in tropical regions", "animals"),
            ("5", "Data analysis and visualization with Python", "programming"));

        using var reader = DirectoryReader.Open(_directory);
        var searcher = new IndexSearcher(reader);
        var queryVector = await GetVectors("building AI and deep learning models");

        var hybridQuery = new BooleanQuery
        {
            { new TermQuery(new Term("category", "programming")), Occur.MUST },
            { new KnnVectorQuery("embedding", queryVector, 5, reader), Occur.MUST }
        };

        var topDocs = searcher.Search(hybridQuery, 10);

        Assert.That(topDocs.TotalHits, Is.EqualTo(4));

        var topText = searcher.Doc(topDocs.ScoreDocs[0].Doc).Get("text");
        Assert.That(topText, Does.Contain("Machine learning").Or.Contain("Data analysis"),
            $"Expected ML/data doc ranked first, got: {topText}");

        foreach (var sd in topDocs.ScoreDocs)
        {
            Assert.That(searcher.Doc(sd.Doc).Get("category"), Is.EqualTo("programming"));
        }
    }
}
