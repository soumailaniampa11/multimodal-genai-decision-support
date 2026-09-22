from src.ingestion.document_loader import DocumentLoader
from src.cleaning.pdf_cleaner import PDFTextCleaner
from src.cleaning.encoding_corrector import PDFEncodingCorrector
from src.chunking.pdf_chunker import PDFChunker
from src.metadata.metadata_builder import MetadataBuilder
from src.embeddings.embedding_model import EmbeddingModel
from src.retrieval.qdrant_store import QdrantStore


def main():
    file_path = "data/raw/GovInst-AI-Whitepaper.pdf"

    # 1. Load PDF
    loader = DocumentLoader(file_path)
    document = loader.load()

    # 2. Clean PDF text
    cleaner = PDFTextCleaner()
    document = cleaner.clean(document)

    # 3. Correct encoding issues
    corrector = PDFEncodingCorrector()
    document = corrector.correct(document)

    # 4. Create chunks
    chunker = PDFChunker(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = chunker.chunk(document)

    print("Number of chunks:", len(chunks))

    # 5. Add metadata
    metadata_builder = MetadataBuilder()
    chunks = metadata_builder.build(
        document,
        chunks,
    )

    # 6. Generate embeddings
    embedding_model = EmbeddingModel()
    embedded_chunks = embedding_model.encode(chunks)

    print(
        "Embedding dimension:",
        embedding_model.dimension(),
    )

    # 7. Connect to Qdrant
    store = QdrantStore()

    # 8. Create collection if needed
    store.create_collection(
        vector_size=embedding_model.dimension(),
    )

    # 9. Insert vectors + metadata
    store.insert_chunks(
        embedded_chunks,
    )


if __name__ == "__main__":
    main()