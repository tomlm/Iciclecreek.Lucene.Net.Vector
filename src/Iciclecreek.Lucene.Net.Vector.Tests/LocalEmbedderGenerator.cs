using Microsoft.Extensions.AI;
using SmartComponents.LocalEmbeddings;

namespace Iciclecreek.Lucene.Net.Vector.Tests;

/// <summary>
/// Adapts SmartComponents.LocalEmbeddings.LocalEmbedder to the standard
/// IEmbeddingGenerator&lt;string, Embedding&lt;float&gt;&gt; interface.
/// Owns the LocalEmbedder and disposes it.
/// </summary>
public class LocalEmbedderGenerator : IEmbeddingGenerator<string, Embedding<float>>
{
    private readonly LocalEmbedder _embedder;
    private bool _disposed;

    public LocalEmbedderGenerator()
    {
        _embedder = new LocalEmbedder();
    }

    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<string> values,
        EmbeddingGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var results = new GeneratedEmbeddings<Embedding<float>>();
        foreach (var text in values)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var embedded = _embedder.Embed(text);
            results.Add(new Embedding<float>(embedded.Values));
        }
        return Task.FromResult(results);
    }

    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        if (serviceType == typeof(LocalEmbedder))
            return _embedder;
        return null;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _embedder.Dispose();
            _disposed = true;
        }
    }
}
