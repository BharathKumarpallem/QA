# System Architecture: AI Document Q&A Service

This document provides a detailed breakdown of the Clean Architecture pattern, data flows, and infrastructure design choices implemented in the project.

---

## 1. Clean Architecture Boundaries

To ensure testability, maintainability, and scalability, the application is divided into decoupled layers. Code dependency flows strictly **inward**: routers depend on services, which depend on core utilities/databases and input/output schemas.

```
                  ┌───────────────────────┐
                  │   HTTP Requests       │
                  └──────────┬────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │ Routers (HTTP Layer)                                   │
 │ - Validates endpoints, parses inputs, triggers services│
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │ Services (Business Logic Layer)                        │
 │ - PDF parsing, chunking, LLM calls, DB CRUD queries    │
 └──────────┬───────────────────────────┬─────────────────┘
            │                           │
            ▼                           ▼
 ┌─────────────────────┐    ┌─────────────────────────────┐
 │ Core / DB (Data)    │    │ Models (Entity Layer)       │
 │ - Chroma, OpenAI    │    │ - Pydantic Request/Response │
 └─────────────────────┘    └─────────────────────────────┘
```

### 1.1. HTTP Router Layer (`app/routers/`)
- Handles incoming API requests, HTTP status codes, and multi-part upload parameters.
- Exposes routes for health, versioning, document management, search, and question answering.
- Employs FastAPI dependency injection (`Depends`) to resolve business services.

### 1.2. Business Logic Services Layer (`app/services/`)
- Contains the core logic of the application.
- **PDFProcessor**: Handles document tokenization boundaries, text extraction, page splitting, and character statistics.
- **EmbeddingService**: Handles text vectorization requests to OpenAI.
- **VectorStoreService**: Interfaces with ChromaDB client, performing collections queries and CRUD logic.
- **LLMService**: Interfaces with GPT-4o-mini structured schema parser, enforces system instructions, and validates citation matches.
- **ChatHistoryService**: Thread-safe in-memory session history manager.

### 1.3. Models Layer (`app/models/`)
- Contains Pydantic v2 schemas.
- Declares the contracts for API requests and JSON responses.
- Ensures all data inputs and outputs are validated at the border.

### 1.4. Core Layer (`app/core/`)
- Configures environment loading (`pydantic-settings`), logging wrappers (`loguru`), exception mappings (`exceptions.py`), and database lifespans.

---

## 2. Cross-Cutting Concerns

### 2.1. Request ID Tracing Middleware
- Implemented as a Starlette HTTP Middleware.
- For every request:
  1. Captures or generates a unique UUID `request_id`.
  2. Binds `request_id` to a thread-safe/async-safe `ContextVar` inside `app/core/logging.py`.
  3. Automatically intercepts and logs API entry and latency metrics, displaying the `request_id` on every line.
  4. Returns the `request_id` in the `X-Request-ID` HTTP header for client-side correlation.

### 2.2. Centralized Exception Handler
- Custom exceptions inherit from `ServiceException` with an associated HTTP status code.
- Custom validation handlers parse and format Pydantic errors into human-friendly JSON.
- A global exception fallback catches all unhandled code errors, logs the full stack trace, and returns a clean `500 Internal Server Error` containing the associated `request_id`.

---

## 3. Data Flow Diagrams

### 3.1. Document Ingestion Flow
1. **User** uploads a PDF to `/documents/upload`.
2. **Router** checks Content-Type.
3. **PDFProcessor** computes SHA-256 and checks ChromaDB for duplicates.
4. **PDFProcessor** extracts page texts, measures characters for scanned PDF warning, and chunks text.
5. **EmbeddingService** requests vectors from OpenAI in batch.
6. **VectorStoreService** writes chunks, embeddings, and metadata to ChromaDB.
7. **Router** returns JSON metadata and a `201 Created` status code.

### 3.2. Question Answering (RAG) Flow
1. **User** posts a query and session ID to `/qa`.
2. **EmbeddingService** embeds the query string.
3. **VectorStoreService** queries ChromaDB for top similarity matches (filtered by document IDs if supplied).
4. **LLMService** checks if the best match score is below `0.55`. If so, returns a "not found" response pre-emptively.
5. **ChatHistoryService** fetches past conversation threads.
6. **LLMService** queries `gpt-4o-mini` with system prompt, history context, and retrieved chunks.
7. **LLMService** validates returned citations.
8. **ChatHistoryService** appends query and answer to context.
9. **Router** returns the validated answer, score, citations, and conversation ID.
