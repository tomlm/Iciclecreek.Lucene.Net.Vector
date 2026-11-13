namespace Iciclecreek.Lucene.Net.Vector;

/// <summary>
/// Configuration options for the HNSW vector index.
/// </summary>
public class VectorIndexOptions
{
    /// <summary>
    /// Maximum number of edges per node in the HNSW graph.
    /// Higher values increase recall but use more memory. Default: 16.
    /// </summary>
    public int M { get; set; } = 16;

    /// <summary>
    /// Size of the dynamic candidate list during construction.
    /// Higher values produce a better graph but slow down indexing. Default: 200.
    /// </summary>
    public int ConstructionPruning { get; set; } = 200;

    /// <summary>
    /// Size of the dynamic candidate list during search.
    /// Higher values increase recall at the cost of latency. Default: 50.
    /// </summary>
    public int EfSearch { get; set; } = 50;

    /// <summary>
    /// Distance function to use for similarity comparison.
    /// </summary>
    public VectorDistanceFunction Distance { get; set; } = VectorDistanceFunction.Cosine;

    /// <summary>
    /// Whether to enable thread-safe operations on the HNSW graph. Default: true.
    /// </summary>
    public bool ThreadSafe { get; set; } = true;

    /// <summary>
    /// The fewest vectors an index segment must hold to get an HNSW graph.
    /// A smaller segment is compared one by one, which is exact and, at this
    /// size, as fast as building and walking a graph. Default: 1000.
    /// </summary>
    public int MinimumGraphSize { get; set; } = 1000;
}

public enum VectorDistanceFunction
{
    Cosine,
    CosineForUnits,
}
