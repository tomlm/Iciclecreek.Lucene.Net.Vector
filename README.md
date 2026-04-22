![logo](https://github.com/user-attachments/assets/4220c13d-b6ba-446a-8a17-02109c017584)
# Iciclecreek.Lucene.Net.Vector

Adds vector similarity search to Lucene.Net 4.8.

Embeddings are stored as `BinaryDocValuesField` and queried through standard Lucene `Query` classes that compose with `BooleanQuery` for filtered and hybrid search.

The library multi-targets `netstandard2.0`, `net8.0`, and `net10.0`:
- **.NET 10+**: uses HNSW approximate nearest neighbor search (fast, via `KnnVectorQuery`)
- **All runtimes**: brute-force cosine similarity with SIMD acceleration (exact, via `CosineVectorQuery`)
- **`VectorQuery`**: smart wrapper that picks the best implementation automatically

## Installation

```bash
dotnet add package Iciclecreek.Lucene.Net.Vector
```

## Usage

### Indexing vectors

Store embeddings alongside your documents using `BinaryDocValuesField`:

```csharp
using Iciclecreek.Lucene.Net.Vector;
using Lucene.Net.Documents;
using Lucene.Net.Index;

var doc = new Document
{
    new StringField("id", "1", Field.Store.YES),
    new TextField("text", "The cat sat on the mat", Field.Store.YES),
    new StringField("category", "animals", Field.Store.YES),
    new BinaryDocValuesField("embedding", VectorSerializer.ToBytesRef(vector)),
};
writer.AddDocument(doc);
writer.Commit();
```

### Vector search

`VectorQuery` is the recommended entry point -- it selects HNSW or brute-force automatically:

```csharp
using var reader = DirectoryReader.Open(directory);
var searcher = new IndexSearcher(reader);
var query = new VectorQuery("embedding", queryVector, k: 10, reader);
var topDocs = searcher.Search(query, 10);

foreach (var scoreDoc in topDocs.ScoreDocs)
{
    var doc = searcher.Doc(scoreDoc.Doc);
    Console.WriteLine($"{doc.Get("id")}: {scoreDoc.Score}");
}
```

### Query classes

| Class | Runtime | Algorithm | Use case |
| --- | --- | --- | --- |
| `VectorQuery` | All | Auto-selects best available for runtime | Top-K nearest neighbor search |
| `CosineVectorQuery` | All | Brute-force cosine similarity (SIMD) | Top-K nearest neighbor search |
| `KnnVectorQuery` | .NET 10+ | HNSW approximate nearest neighbor | Top-K nearest neighbor search |
| `VectorScoreQuery` | All | Cosine re-scoring via `CustomScoreQuery` | Re-rank results from any Lucene query |

The first three find the K most similar vectors and return them as results. `VectorScoreQuery` does the opposite — it takes an existing query that controls which documents match, and replaces the score with cosine similarity. This is useful when you already have a filter or full-text query selecting documents and want to rank them by vector similarity without a separate top-K pass.

All four extend `Lucene.Net.Search.Query` and compose naturally with `BooleanQuery`.

### Filtered search

Compose any vector query with `BooleanQuery` to combine similarity with field filters:

```csharp
var boolQuery = new BooleanQuery
{
    { new TermQuery(new Term("category", "animals")), Occur.MUST },
    { new VectorQuery("embedding", queryVector, k: 10, reader), Occur.MUST },
};
var topDocs = searcher.Search(boolQuery, 10);
```

### Vector re-scoring

`VectorScoreQuery` re-ranks results from any Lucene query by cosine similarity. The sub-query controls which documents match; the vector score controls ranking:

```csharp
// Re-rank full-text search results by vector similarity
var query = new VectorScoreQuery(
    new TermQuery(new Term("category", "animals")),  // filter: which docs match
    "embedding",                                      // vector field name
    queryVector);                                     // query vector

var topDocs = searcher.Search(query, 10);
// Results are filtered to "animals" category, ranked by cosine similarity
```

This is useful when you want Lucene's standard query engine to handle filtering (full-text, term, range, boolean) and just need vector similarity for ranking — no top-K limit, no HNSW index.

### After updates

When you add, update, or delete documents, open a new reader -- the HNSW index rebuilds automatically:

```csharp
writer.DeleteDocuments(new Term("id", "old-doc"));
writer.Commit();

using var newReader = DirectoryReader.Open(directory);
var searcher = new IndexSearcher(newReader);
var query = new VectorQuery("embedding", queryVector, k: 10, newReader);
```

### HNSW options (.NET 10+)

Pass `VectorIndexOptions` to tune the HNSW graph when using `KnnVectorQuery` or `VectorQuery`:

```csharp
var options = new VectorIndexOptions
{
    M = 16,                                        // Max edges per node (default: 16)
    ConstructionPruning = 200,                     // Build quality (default: 200)
    EfSearch = 50,                                 // Search quality (default: 50)
    Distance = VectorDistanceFunction.Cosine,      // Distance function (default: Cosine)
};
var query = new VectorQuery("embedding", queryVector, k: 10, reader, options);
```

On runtimes without HNSW, options are accepted but ignored -- `CosineVectorQuery` is used instead.

### RAG (Retrieval-Augmented Generation)

A typical RAG pipeline: embed your documents at index time, then at query time embed the user's question, retrieve relevant context via vector search, and pass it to an LLM.

```csharp
// --- Setup: any IEmbeddingGenerator (OpenAI, Ollama, local ONNX, etc.) ---
IEmbeddingGenerator<string, Embedding<float>> embedder = ...;
IChatClient chatClient = ...;

// --- Index time: embed and store documents ---
using var directory = FSDirectory.Open("my-index");
using var analyzer = new StandardAnalyzer(LuceneVersion.LUCENE_48);
var config = new IndexWriterConfig(LuceneVersion.LUCENE_48, analyzer);
using var writer = new IndexWriter(directory, config);

foreach (var chunk in documentChunks)
{
    var embedding = await embedder.GenerateAsync([chunk.Text]);
    writer.AddDocument(new Document
    {
        new StringField("id", chunk.Id, Field.Store.YES),
        new StoredField("text", chunk.Text),
        new BinaryDocValuesField("embedding",
            VectorSerializer.ToBytesRef(embedding[0].Vector.ToArray())),
    });
}
writer.Commit();

// --- Query time: embed question, retrieve context, generate answer ---
var question = "How does photosynthesis work?";
var questionEmbedding = await embedder.GenerateAsync([question]);
var queryVector = questionEmbedding[0].Vector.ToArray();

using var reader = DirectoryReader.Open(directory);
var searcher = new IndexSearcher(reader);
var query = new VectorQuery("embedding", queryVector, k: 5, reader);
var topDocs = searcher.Search(query, 5);

// Gather context from top matches
var context = string.Join("\n\n", topDocs.ScoreDocs
    .Select(sd => searcher.Doc(sd.Doc).Get("text")));

// Pass to LLM
var response = await chatClient.GetResponseAsync($"""
    Answer the question based on the following context:

    {context}

    Question: {question}
    """);

Console.WriteLine(response);
```

This composes with Lucene's full query model — you can add metadata filters, full-text search, or boost certain fields alongside the vector search:

```csharp
// RAG with metadata filter: only search recent documents
var filteredRag = new BooleanQuery
{
    { NumericRangeQuery.NewInt64Range("timestamp", recentCutoff, null, true, false), Occur.MUST },
    { new VectorQuery("embedding", queryVector, k: 10, reader), Occur.MUST },
};
```

### Utilities

- **`VectorSerializer`** — converts between `float[]` and Lucene's `BytesRef`/`byte[]` for storage
- **`VectorMath`** — SIMD-accelerated cosine similarity and vector norm utilities

## Notes

- **Vectors are stored in Lucene** as `BinaryDocValuesField` (4 bytes per float, little-endian). The HNSW graph is an in-memory acceleration structure built automatically from DocValues.
- **HNSW caching** — the graph is cached per `(IndexReader, fieldName)`. Since an `IndexReader` is an immutable snapshot, the cached graph is valid for the reader's entire lifetime. Opening a new reader after commits triggers a fresh build.
- **Deleted documents** are automatically excluded — Lucene's `LiveDocs` filtering ensures deleted docs are not loaded into the HNSW graph or evaluated by brute-force search.
- **Distance functions**: `Cosine` (general purpose, handles unnormalized vectors) and `CosineForUnits` (faster, requires pre-normalized unit vectors). These apply to HNSW only; `CosineVectorQuery` always uses cosine similarity.

## License

MIT
