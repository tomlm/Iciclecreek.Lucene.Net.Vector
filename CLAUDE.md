# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Iciclecreek.Lucene.Net.Vector adds vector similarity search to Lucene.Net 4.8. Vectors are stored as `BinaryDocValuesField` and searched via `KnnVectorQuery` — a native Lucene `Query` subclass that composes with `BooleanQuery`. The HNSW index is built and cached automatically per index segment.

This is the **core library only**. A separate VectorData connector library (Microsoft.Extensions.VectorData) wraps this one.

## Build & Test Commands

```bash
dotnet build src/Iciclecreek.Lucene.Net.Vector.slnx
dotnet test src/Iciclecreek.Lucene.Net.Vector.slnx
dotnet test src/Iciclecreek.Lucene.Net.Vector.Tests --filter "FullyQualifiedName~TestName"
```

## Architecture

- **Solution**: `src/Iciclecreek.Lucene.Net.Vector.slnx`
- **Library**: `src/Iciclecreek.Lucene.Net.Vector/` — namespace `Iciclecreek.Lucene.Net.Vector`
- **Tests**: `src/Iciclecreek.Lucene.Net.Vector.Tests/` — NUnit 4.3

### Public API

- **KnnVectorQuery** — `Lucene.Net.Search.Query` subclass. Takes `(field, queryVector, k, reader, options?)`. HNSW index built/cached automatically per `(segment, field)` via internal `VectorIndexCache`; results of all segments are merged to the top K.
- **VectorSerializer** — `float[] ↔ byte[] ↔ BytesRef` conversion for `BinaryDocValuesField` storage.
- **VectorIndexOptions** — HNSW tuning (M, ConstructionPruning, EfSearch, Distance, MinimumGraphSize).
- **SearchResult** — `(DocId, Distance, Score)` record struct.

### Internal

- **LuceneVectorIndex** — the vectors of one segment, loaded from DocValues, with segment-local doc ids. Holds deleted documents too; each search passes the reader's `LiveDocs`. Segments with at least `MinimumGraphSize` vectors get an HNSW graph, built on a background queue (one at a time); until it is ready, and for smaller segments, search compares one by one. Internal class, not part of the public API.
- **VectorIndexCache** — `ConditionalWeakTable`-based cache keyed on the segment's `CoreCacheKey`, so readers reopened with `OpenIfChanged` share entries. Entries are removed when the segment core is disposed, or GC'd. Disposing is what frees an HNSW graph: `LuceneVectorIndex.ReleaseGraph` empties the package's private searcher pool by reflection, because the pool otherwise keeps the graph reachable for the life of the process (HNSW has no public API for it; `KnnSegmentTests.ADisposedSegment_LetsItsGraphBeCollected` fails if an upgrade breaks this).

### Namespace Conflict

The project namespace `Iciclecreek.Lucene.Net.Vector` contains `Lucene.Net` as a prefix, which conflicts with the `Lucene.Net` NuGet namespace. Always use `global::Lucene.Net.*` in `using` directives (e.g., `using global::Lucene.Net.Index;`).

### Dependencies

- `Lucene.Net` (4.8.0-beta00017) — no `Occur.FILTER` in 4.8, use `Occur.MUST` instead
- `HNSW` (26.9.5494) — `SmallWorld<float[], float>`, `CosineDistance.SIMD`, `DefaultRandomGenerator.Instance` (no public constructor)
- HNSW `KNNSearch` filter is `Func<float[], bool>` (filters by vector item, not by index). Use `ReferenceEqualityComparer` to map vectors back to doc IDs
