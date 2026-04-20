# Iciclecreek.Lucene.Net.Vector

Adds vector similarity search to Lucene.Net 4.8.

Embeddings are stored as `BinaryDocValuesField` and queried through `KnnVectorQuery`, a native Lucene `Query` that composes with `BooleanQuery` for filtered and hybrid search. An HNSW index is built and cached automatically per `IndexReader` — no manual index management required.

## Installation

```bash
dotnet add package Iciclecreek.Lucene.Net.Vector
```

### Dependencies

- [Lucene.Net](https://www.nuget.org/packages/Lucene.Net/) 4.8.0-beta00017
- [HNSW](https://www.nuget.org/packages/HNSW/) (Microsoft HNSW.Net — SIMD cosine distance)

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

`KnnVectorQuery` is a standard Lucene `Query` — use it with `IndexSearcher` like any other query:

```csharp
using var reader = DirectoryReader.Open(directory);
var searcher = new IndexSearcher(reader);
var queryVector = GetVector(userQuery) 
var query = new KnnVectorQuery("embedding", queryVector, k: 10, reader);
var topDocs = searcher.Search(query, 10);

foreach (var scoreDoc in topDocs.ScoreDocs)
{
    var doc = searcher.Doc(scoreDoc.Doc);
    Console.WriteLine($"{doc.Get("id")}: {scoreDoc.Score}");
}
```

The HNSW index is built automatically on the first query and cached for the lifetime of the `IndexReader`.

### Filtered search

Compose `KnnVectorQuery` with `BooleanQuery` to combine vector similarity with field filters:

```csharp
var boolQuery = new BooleanQuery
{
    { new TermQuery(new Term("category", "animals")), Occur.MUST },
    { new KnnVectorQuery("embedding", queryVector, k: 10, reader), Occur.MUST },
};
var topDocs = searcher.Search(boolQuery, 10);
```

### After updates

When you add, update, or delete documents, open a new reader — the HNSW index rebuilds automatically:

```csharp
writer.DeleteDocuments(new Term("id", "old-doc"));
writer.Commit();

using var newReader = DirectoryReader.Open(directory);
var searcher = new IndexSearcher(newReader);
// KnnVectorQuery will build a fresh HNSW index from the new reader's DocValues
var query = new KnnVectorQuery("embedding", queryVector, k: 10, newReader);
```

### HNSW options

Pass `VectorIndexOptions` to tune the HNSW graph:

```csharp
var options = new VectorIndexOptions
{
    M = 16,                                        // Max edges per node (default: 16)
    ConstructionPruning = 200,                     // Build quality (default: 200)
    EfSearch = 50,                                 // Search quality (default: 50)
    Distance = VectorDistanceFunction.Cosine,      // Distance function (default: Cosine)
};
var query = new KnnVectorQuery("embedding", queryVector, k: 10, reader, options);
```

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
var query = new KnnVectorQuery("embedding", queryVector, k: 5, reader);
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
    { new KnnVectorQuery("embedding", queryVector, k: 10, reader), Occur.MUST },
};
```

## Notes

- **Vectors are stored in Lucene** as `BinaryDocValuesField` (4 bytes per float, little-endian). The HNSW graph is an in-memory acceleration structure built automatically from DocValues.
- **HNSW caching** — the graph is cached per `(IndexReader, fieldName)`. Since an `IndexReader` is an immutable snapshot, the cached graph is valid for the reader's entire lifetime. Opening a new reader after commits triggers a fresh build.
- **Deleted documents** are automatically excluded — Lucene's `LiveDocs` filtering ensures deleted docs are not loaded into the HNSW graph.
- **Distance functions**: `Cosine` (general purpose, handles unnormalized vectors) and `CosineForUnits` (faster, requires pre-normalized unit vectors).

## License

MIT
