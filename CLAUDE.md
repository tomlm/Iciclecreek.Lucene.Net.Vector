# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Iciclecreek.Lucene.Net.Vector adds vector similarity search to Lucene.Net 4.8. It stores embeddings as `BinaryDocValuesField`, builds HNSW indexes for approximate nearest neighbor search, and provides `KnnVectorQuery` — a native Lucene `Query` subclass that composes with `BooleanQuery`.

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

### Key Components

- **VectorSerializer** — `float[] ↔ byte[] ↔ BytesRef` conversion for `BinaryDocValuesField` storage
- **LuceneVectorIndex** — HNSW graph management over Lucene DocValues. Supports build, incremental add/remove, search with optional doc ID filtering, and graph serialization
- **KnnVectorQuery** — `Lucene.Net.Search.Query` subclass. Composes with `BooleanQuery` for filtered/hybrid search. Uses `LuceneVectorIndex` internally

### Namespace Conflict

The project namespace `Iciclecreek.Lucene.Net.Vector` contains `Lucene.Net` as a prefix, which conflicts with the `Lucene.Net` NuGet namespace. Always use `global::Lucene.Net.*` in `using` directives (e.g., `using global::Lucene.Net.Index;`).

### Dependencies

- `Lucene.Net` / `Lucene.Net.QueryParser` (4.8.0-beta00017) — no `Occur.FILTER` in 4.8, use `Occur.MUST` instead
- `HNSW` (26.4.177) — `SmallWorld<float[], float>`, `CosineDistance.SIMD`, `DefaultRandomGenerator.Instance` (no public constructor)
- HNSW `KNNSearch` filter is `Func<float[], bool>` (filters by vector item, not by index). Use `ReferenceEqualityComparer` to map vectors back to doc IDs
