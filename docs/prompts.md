# Prompt Engineering & LLM Design

This document details the system prompt engineering, context injection formats, and instruction boundaries configured in `app/services/llm_service.py` to ensure high QA accuracy and prevent hallucinations.

---

## 1. System Prompt Template

We use a strict developer/system prompt configuration to constrain the LLM's answers to the retrieved text context blocks.

```markdown
You are a professional AI Document Assistant. Your goal is to answer the user's question strictly using the provided document context blocks. Follow these instructions carefully:
1. Answer the question based ONLY on the provided context blocks. Do NOT use any external knowledge.
2. For every fact, claim, or quote you extract, you MUST include a citation object in the `citations` list. A citation must have the exact page number, the exact document ID, document name, and a verbatim snippet from the context text.
3. If the context does not contain the answer, answer: 'I cannot find the answer to your question in the uploaded documents.', set `confidence_score` to 0.0, and leave the `citations` list empty.
4. Assign a `confidence_score` between 0.0 and 1.0. If the context answers the query directly, confidence should be high (0.8 - 1.0). If it only partially answers or is ambiguous, score lower (0.5 - 0.7).
5. The cited `exact_snippet` must be an absolute substring match (case-insensitive and whitespace-normalized) within the text of the source context block you are citing.
```

---

## 2. Context Injection Format

Retrieved context blocks are injected as standard user context, structured to make it easy for the model to distinguish between different files and pages.

```markdown
Document Context:
--- CONTEXT BLOCK 1 ---
Document ID: 550e8400-e29b-41d4-a716-446655440000
Document Name: employee_handbook.pdf
Page Number: 14
Text:
The annual standard PTO allocation is 25 days. Employees must request leave at least two weeks in advance.

--- CONTEXT BLOCK 2 ---
Document ID: f47ac10b-58cc-4372-a567-0e02b2c3d479
Document Name: hr_policy.pdf
Page Number: 3
Text:
Sick leaves are capped at 10 days per calendar year. Unused sick leaves do not roll over to the following year.

User Question: How many PTO days do employees get?
```

---

## 3. Defense-in-Depth against Hallucination

To guarantee truthfulness, our pipeline uses a three-stage validation strategy:

1. **Pre-LLM Filtering**: We measure vector database similarity scores. If all matches fall below `0.55`, the LLM is not called, and the API returns a standard "not found" response.
2. **LLM Constraint Prompting**: The system prompt forces `confidence_score = 0.0` and empty `citations` if the answer is missing from the context. We set `temperature=0.0` to eliminate creative reasoning.
3. **Post-LLM Substring Verification**: The backend validates that every citation snippet returned by the LLM is a verbatim substring of the source chunks retrieved from ChromaDB. Invalid citations are stripped, and confidence scores are penalized accordingly.

---

## 4. AI Usage & Code Correction Log

During the implementation and automated testing of this codebase, three key code issues were identified and corrected. Below is the documentation of these issues, their resolutions, and recommendations to prevent them in future AI prompts.

### 4.1. ChromaDB API Client Type Mismatch
- **Issue**: Type hinting used `chromadb.API` in `app/core/database.py` and `app/routers/health.py`. Under newer versions of ChromaDB, `chromadb.API` is no longer directly exported at the package root level, throwing `AttributeError: module 'chromadb' has no attribute 'API'`.
- **Correction**: Changed type annotations to `chromadb.ClientAPI`, which is the correct exported client interface class.
- **Code Diff**:
  ```diff
  -def get_chroma_client() -> chromadb.API:
  +def get_chroma_client() -> chromadb.ClientAPI:
  ```

### 4.2. Test Dependency Override Scope Mismatch
- **Issue**: In `tests/test_api.py`, `app.dependency_overrides` was mapped using `get_embedding_service` and `get_vector_store` imported from `app.routers.documents`. However, the `/qa` endpoint imports these dependency functions from `app.routers.qa`. Because these are separate function definitions in separate modules, the overrides were bypassed on `/qa`, leading to real OpenAI API calls that failed with unauthorized keys.
- **Correction**: Imported dependency provider functions explicitly from `app.routers.qa` for the QA endpoint test and registered overrides on them.
- **Code Diff**:
  ```diff
  -    app.dependency_overrides[get_embedding_service] = lambda: mock_emb
  +    from app.routers.qa import get_embedding_service as get_qa_embedding_service
  +    app.dependency_overrides[get_qa_embedding_service] = lambda: mock_emb
  ```

### 4.3. DOM ID Collision in Chat Citation Chips
- **Issue**: In `index.html`, when multiple assistant answers were rendered, each created citation chips using indices `0, 1, 2...` as the ID suffix (`cit-snip-${idx}`). Clicking any chip always toggled the citation snippet of the *first* message on the page because `document.getElementById` always resolved to the first matching ID in the DOM.
- **Correction**: Refactored the citation chip to pass `this` (the element reference) to `toggleCitationSnippet` and query the snippet class locally relative to the clicked chip.
- **Code Diff**:
  ```diff
  -    <div class="citation-chip" onclick="toggleCitationSnippet(${idx})">
  +    <div class="citation-chip" onclick="toggleCitationSnippet(this)">
  ```
