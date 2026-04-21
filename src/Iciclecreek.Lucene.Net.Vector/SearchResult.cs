namespace Iciclecreek.Lucene.Net.Vector;

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
