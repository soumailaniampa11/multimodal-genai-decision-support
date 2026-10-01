from src.ingestion.document_loader import DocumentLoader
from src.cleaning.pdf_cleaner import PDFTextCleaner
from src.cleaning.encoding_corrector import PDFEncodingCorrector
from src.chunking.pdf_chunker import PDFChunker
from src.metadata.metadata_builder import MetadataBuilder
from src.embeddings.embedding_model import EmbeddingModel
from src.retrieval.qdrant_store import QdrantStore
from src.config import load_config


def main():
    config = load_config()

    file_path = config["data"]["pdf_path"]

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
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"],
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
    embedding_model = EmbeddingModel(
        model_name=config["embeddings"]["model_name"],
    )
    embedded_chunks = embedding_model.encode(chunks)

    print(
        "Embedding dimension:",
        embedding_model.dimension(),
    )

    # 7. Connect to Qdrant
    store = QdrantStore(**config["qdrant"])

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