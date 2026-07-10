from app.services.vector_store import VectorStoreService


def test_add_search_delete_flow(test_chroma_client) -> None:
    service = VectorStoreService()

    # Cleanup any leftovers
    try:
        service.collection.delete(where={"document_id": "doc-uuid-123"})
    except Exception:
        pass

    chunks = [
        {
            "text": "Python 3.12 is the primary language used in this backend service.",
            "page_number": 1,
        },
        {
            "text": "ChromaDB is utilized as the persistent vector store for similarity searches.",
            "page_number": 2,
        },
    ]
    embeddings = [[0.1] * 1536, [0.9] * 1536]

    # 1. Store chunks
    service.add_document_chunks(
        document_id="doc-uuid-123",
        filename="manual.pdf",
        sha256="abc123sha",
        chunks=chunks,
        embeddings=embeddings,
        total_pages=2,
    )

    # 2. Check Duplicate detection
    dup = service.check_duplicate_by_hash("abc123sha")
    assert dup is not None
    assert dup["document_id"] == "doc-uuid-123"
    assert dup["filename"] == "manual.pdf"
    assert dup["chunk_count"] == 2
    assert dup["page_count"] == 2

    # Verify duplicate returns None for missing hash
    no_dup = service.check_duplicate_by_hash("nonexistentsha")
    assert no_dup is None

    # 3. Perform semantic search matching (cosine space)
    # Query vector close to second chunk
    query_vector = [0.85] * 1536
    results = service.search_similar_chunks(query_vector, top_k=1)

    assert len(results) == 1
    assert results[0]["document_id"] == "doc-uuid-123"
    assert results[0]["page_number"] == 2
    assert "ChromaDB" in results[0]["text"]
    assert results[0]["score"] > 0.90  # High cosine similarity

    # 4. Multi-document filtering search
    # Filter with matching ID
    filtered_results = service.search_similar_chunks(
        query_vector, document_ids=["doc-uuid-123"], top_k=1
    )
    assert len(filtered_results) == 1

    # Filter with non-matching ID
    empty_results = service.search_similar_chunks(
        query_vector, document_ids=["another-uuid"], top_k=1
    )
    assert len(empty_results) == 0

    # 5. Delete document
    service.delete_document("doc-uuid-123")

    # Verify document chunks are gone
    dup_after = service.check_duplicate_by_hash("abc123sha")
    assert dup_after is None
