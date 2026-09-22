from typing import Any


class MetadataBuilder:
    """
    Build metadata for document chunks.

    The metadata builder enriches each chunk with information
    useful for retrieval, filtering, provenance and traceability.
    """

    def build(
        self,
        document: dict[str, Any],
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Add metadata to each chunk.

        Args:
            document:
                The processed PDF document.

            chunks:
                Chunks produced by PDFChunker.

        Returns:
            A list of enriched chunks.
        """

        enriched_chunks = []

        for index, chunk in enumerate(chunks, start=1):

            enriched_chunk = {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "file_name": document["file_name"],
                "file_type": document["file_type"],
                "page": chunk["page"],
                "chunk_index": index,
                "text": chunk["text"],
            }

            enriched_chunks.append(enriched_chunk)

        return enriched_chunks