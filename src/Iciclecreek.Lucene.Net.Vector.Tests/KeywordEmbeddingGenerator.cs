using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;

namespace Iciclecreek.Lucene.Net.Vector.Tests;

/// <summary>
/// A mock embedding generator that maps keywords to vector dimensions.
/// Produces 16-dimensional vectors where each dimension corresponds to a
/// topic/keyword group. Text containing related words will produce similar
/// vectors, enabling semantic-like matching in tests without real ML models.
/// </summary>
internal class KeywordEmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>
{
    // Each entry: (dimension index, keywords that activate it)
    private static readonly (int dim, string[] keywords)[] TopicMap = new[]
    {
        (0,  new[] { "cat", "kitten", "feline", "meow", "purr" }),
        (1,  new[] { "dog", "puppy", "canine", "bark" }),
        (2,  new[] { "mat", "rug", "carpet", "blanket", "sleeping", "rested", "sat", "napping" }),
        (3,  new[] { "park", "played", "run", "walk" }),
        (4,  new[] { "machine", "learning", "neural", "network", "ai", "model", "deep", "data" }),
        (5,  new[] { "python", "programming", "code", "coding", "software", "development", "javascript", "react", "framework" }),
        (6,  new[] { "stock", "market", "economy", "financial", "investor", "inflation", "economic", "recovery", "crashed" }),
        (7,  new[] { "science", "scientist", "species", "planet", "star", "dna", "discovery", "biology", "astronomy", "frog" }),
        (8,  new[] { "president", "congress", "senator", "government", "legislation", "policy", "politics", "immigration", "healthcare", "reform", "speech", "trade", "agreement" }),
        (9,  new[] { "apple", "iphone", "google", "quantum", "computing", "camera", "tech" }),
        (10, new[] { "weather", "sunny", "warm", "rain" }),
        (11, new[] { "food", "cooking", "pasta", "vegetables", "health", "boiling" }),
        (12, new[] { "animal", "bear", "fox", "honey", "forest", "snake", "tropical" }),
        (13, new[] { "small", "little", "tiny", "cute" }),
        (14, new[] { "big", "large", "friendly", "quick", "brown", "lazy" }),
        (15, new[] { "hello", "world", "foo", "test" }),
    };

    private const int Dimensions = 16;

    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<string> values,
        EmbeddingGenerationOptions options = null,
        CancellationToken cancellationToken = default)
    {
        var results = new GeneratedEmbeddings<Embedding<float>>();
        foreach (var text in values)
        {
            results.Add(new Embedding<float>(Embed(text)));
        }
        return Task.FromResult(results);
    }

    private static float[] Embed(string text)
    {
        var vector = new float[Dimensions];
        var lower = text.ToLowerInvariant();
        var words = lower.Split(new[] { ' ', ',', '.', '!', '?', ';', ':' }, StringSplitOptions.RemoveEmptyEntries);

        foreach (var (dim, keywords) in TopicMap)
        {
            float score = 0f;
            foreach (var word in words)
            {
                foreach (var keyword in keywords)
                {
                    if (word.Contains(keyword) || keyword.Contains(word))
                    {
                        score += 1f;
                    }
                }
            }
            vector[dim] = score;
        }

        // Normalize to unit vector
        var mag = (float)Math.Sqrt(vector.Sum(x => x * x));
        if (mag > 0f)
        {
            for (int i = 0; i < vector.Length; i++)
                vector[i] /= mag;
        }

        return vector;
    }

    public EmbeddingGeneratorMetadata Metadata => new EmbeddingGeneratorMetadata("KeywordMock");

    public object GetService(Type serviceType, object key = null) => null;

    public void Dispose() { }
}
