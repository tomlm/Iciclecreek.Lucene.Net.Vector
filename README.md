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

## Notes

- **Vectors are stored in Lucene** as `BinaryDocValuesField` (4 bytes per float, little-endian). The HNSW graph is an in-memory acceleration structure built automatically from DocValues.
- **HNSW caching** — the graph is cached per `(IndexReader, fieldName)`. Since an `IndexReader` is an immutable snapshot, the cached graph is valid for the reader's entire lifetime. Opening a new reader after commits triggers a fresh build.
- **Deleted documents** are automatically excluded — Lucene's `LiveDocs` filtering ensures deleted docs are not loaded into the HNSW graph.
- **Distance functions**: `Cosine` (general purpose, handles unnormalized vectors) and `CosineForUnits` (faster, requires pre-normalized unit vectors).

## License

MIT
