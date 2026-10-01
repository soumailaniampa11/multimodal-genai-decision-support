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

    # 1. Find every supported document in the raw data folder
    file_paths = DocumentLoader.find_documents(
        config["data"]["raw_dir"],
    )

    print("Documents found:", len(file_paths))

    cleaner = PDFTextCleaner()
    corrector = PDFEncodingCorrector()
    chunker = PDFChunker(
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"],
    )
    metadata_builder = MetadataBuilder()

    all_chunks = []

    for file_path in file_paths:

        # 2. Load the document
        loader = DocumentLoader(
            str(file_path),
            ocr_languages=config["ingestion"]["ocr_languages"],
        )

        try:
            document = loader.load()
        except Exception as exc:
            print(f"Skipped {file_path.name}: {exc}")
            continue

        # 3. Clean text
        document = cleaner.clean(document)

        # 4. Correct encoding issues
        document = corrector.correct(document)

        # 5. Create chunks
        chunks = chunker.chunk(document)

        # 6. Add metadata
        chunks = metadata_builder.build(
            document,
            chunks,
        )

        print(
            f"{file_path.name}: "
            f"{len(document['pages'])} {document['unit']}(s), "
            f"{len(chunks)} chunks"
        )

        all_chunks.extend(chunks)

    print("Total chunks:", len(all_chunks))

    # 7. Generate embeddings
    embedding_model = EmbeddingModel(**config["embeddings"])
    embedded_chunks = embedding_model.encode(all_chunks)

    print(
        "Embedding dimension:",
        embedding_model.dimension(),
    )

    # 8. Connect to Qdrant
    store = QdrantStore(**config["qdrant"])

    # 9. Rebuild the collection from scratch
    store.create_collection(
        vector_size=embedding_model.dimension(),
        recreate=True,
    )

    # 10. Insert vectors + metadata
    store.insert_chunks(
        embedded_chunks,
    )


if __name__ == "__main__":
    main()
