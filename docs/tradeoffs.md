# Architecture Trade-offs & Alternatives

This document records the architectural and technology stack trade-offs considered during the design and development of the AI Document Q&A Service.

---

## 1. Vector Database: ChromaDB vs. alternatives

### ChromaDB (Chosen)
* **Pros**: Simple to set up, runs embedded locally (no external container or cluster needed), and integrates out of the box with Python applications. Supports rich metadata filtering, HNWS indexing, and customizable distance metrics (cosine, L2, IP).
* **Cons**: In-memory or local file-based storage does not scale horizontally across multiple application nodes.
* **Verdict**: Perfect for single-instance, lightweight deployments and developers. For horizontal scale, a transition to **Pinecone** (managed SaaS) or **PGVector** (PostgreSQL-based relational vector DB) is recommended.

---

## 2. PDF Parsing: pdfplumber vs. PyPDF / PyMuPDF

### pdfplumber (Chosen)
* **Pros**: Highly accurate text coordinate extraction. Easily handles multi-column layouts and tables, and returns clean page-by-page structures.
* **Cons**: Marginally slower than pure C-based libraries like PyMuPDF.
* **Verdict**: The accuracy of text layout parsing directly impacts chunk quality. The speed trade-off is negligible for documents under 50MB, making pdfplumber the optimal choice.

---

## 3. Chunking: Page-Aware Token Chunking vs. Global Character Chunking

### Page-Aware Token Chunking (Chosen)
* **Pros**: Restricting chunking boundaries to individual pages guarantees that citation metadata (`page_number`) is 100% correct. Using `tiktoken` guarantees that we fit chunks exactly within LLM context constraints.
* **Cons**: If a sentence spans across a page boundary, it might be split into two chunks, potentially losing local context.
* **Verdict**: Absolute citation accuracy is a key business requirement. Global character-based chunking would smear citations across multiple pages, which is unacceptable for production-grade audit trails.

---

## 4. LLM Schema Format: OpenAI Structured Outputs vs. Pydantic Parsing

### OpenAI Structured Outputs (Chosen)
* **Pros**: Built directly into the model's decoding stage (constrains token generation to match the JSON schema perfectly). Zero parser validation failures, and eliminates the need for regex or external libraries like Instructor or LangChain.
* **Cons**: Generates a small amount of latency overhead on the first API call to register the schema.
* **Verdict**: Guarantees that the LLM's output conforms exactly to our Pydantic response models, preventing JSON parse failures.

---

## 5. Chat History: Thread-Safe Memory vs. Redis / SQL Database

### Thread-Safe In-Memory Singleton (Chosen)
* **Pros**: Extremely fast, simple, and requires zero external infrastructure dependencies.
* **Cons**: Session logs are volatile. Restarting the server clears all conversation history.
* **Verdict**: Ideal for technical assessments and single-node setups. For production horizontal load balancing, session history should be backed by a persistent store like **Redis** (with TTL cache expiration) or a relational database (PostgreSQL/MySQL).
