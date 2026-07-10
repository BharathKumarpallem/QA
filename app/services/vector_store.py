from typing import List, Dict, Any, Optional
from app.core.database import get_chroma_client
from app.core.exceptions import VectorStoreException
from loguru import logger


class VectorStoreService:
    """
    Service wrapping ChromaDB operations. Manages the document chunks collection,
    duplicate check querying, semantic searches, and document deletions.
    """

    def __init__(self) -> None:
        self.client = get_chroma_client()
        self.collection_name = "document_chunks"
        try:
            # Create or get collection using cosine similarity metric
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize Chroma collection '{self.collection_name}': {str(e)}"
            )
            raise VectorStoreException(f"ChromaDB initialization failure: {str(e)}")

    def check_duplicate_by_hash(self, sha256: str) -> Optional[Dict[str, Any]]:
        """
        Queries Chroma to check if a document with the same SHA-256 hash exists.

        Returns:
            Optional[Dict[str, Any]]: Information about the existing document if found.
        """
        try:
            results = self.collection.get(where={"sha256": sha256}, limit=1)
            if results and results["ids"] and len(results["ids"]) > 0:
                metadata = results["metadatas"][0]
                return {
                    "document_id": metadata["document_id"],
                    "filename": metadata["document_name"],
                    "sha256": sha256,
                    "page_count": metadata.get("total_pages", 0),
                    "chunk_count": metadata.get("total_chunks", 0),
                    "status": "duplicate",
                }
            return None
        except Exception as e:
            logger.error(f"Error checking duplicate hash in Chroma: {str(e)}")
            raise VectorStoreException(
                f"Failed to check duplicate document hash: {str(e)}"
            )

    def add_document_chunks(
        self,
        document_id: str,
        filename: str,
        sha256: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        total_pages: int,
    ) -> None:
        """
        Adds text chunks and their OpenAI embeddings to the Chroma collection.
        """
        if not chunks:
            return

        ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "document_id": document_id,
                "document_name": filename,
                "sha256": sha256,
                "page_number": chunk["page_number"],
                "total_pages": total_pages,
                "total_chunks": len(chunks),
            }
            for chunk in chunks
        ]

        logger.info(
            f"Adding {len(chunks)} chunks for document {filename} (ID: {document_id}) to ChromaDB."
        )
        try:
            self.collection.add(
                ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
            )
        except Exception as e:
            logger.error(f"Failed to add document chunks to ChromaDB: {str(e)}")
            raise VectorStoreException(
                f"Failed to write chunks to vector database: {str(e)}"
            )

    def search_similar_chunks(
        self,
        query_embedding: List[float],
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic search querying the collection with distance metrics.

        Args:
            query_embedding: The vector representation of the query.
            document_ids: Optional list of document UUIDs to filter search context.
            top_k: Max similar chunks to return.

        Returns:
            List[Dict]: List of match records containing score, text, and metadata.
        """
        # Formulate query filters
        where_filter = None
        if document_ids:
            if len(document_ids) == 1:
                where_filter = {"document_id": document_ids[0]}
            else:
                where_filter = {"document_id": {"$in": document_ids}}

        try:
            logger.info(
                f"Querying ChromaDB for top {top_k} results. Filter: {where_filter}"
            )
            results = self.collection.query(
                query_embeddings=[query_embedding], n_results=top_k, where=where_filter
            )

            chunk_results = []
            if not results or not results["ids"] or len(results["ids"][0]) == 0:
                return []

            ids = results["ids"][0]
            distances = results["distances"][0]
            metadatas = results["metadatas"][0]
            documents = results["documents"][0]

            for i in range(len(ids)):
                # Since we used cosine metric, distance = 1 - cosine_similarity
                # Cosine similarity = 1 - distance
                similarity = 1.0 - distances[i]

                chunk_results.append(
                    {
                        "text": documents[i],
                        "document_id": metadatas[i]["document_id"],
                        "document_name": metadatas[i]["document_name"],
                        "page_number": metadatas[i]["page_number"],
                        "score": similarity,
                    }
                )

            return chunk_results
        except Exception as e:
            logger.error(f"ChromaDB search query failed: {str(e)}")
            raise VectorStoreException(f"Failed to query semantic vectors: {str(e)}")

    def delete_document(self, document_id: str) -> None:
        """
        Deletes all chunks associated with a document ID.
        """
        try:
            logger.info(f"Deleting document chunks for ID: {document_id}")
            self.collection.delete(where={"document_id": document_id})
        except Exception as e:
            logger.error(
                f"Failed to delete document {document_id} from Chroma: {str(e)}"
            )
            raise VectorStoreException(f"Failed to delete document: {str(e)}")
