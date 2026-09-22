from src.ingestion.document_loader import DocumentLoader
from src.cleaning.pdf_cleaner import PDFTextCleaner
from src.cleaning.encoding_corrector import PDFEncodingCorrector
from src.chunking.pdf_chunker import PDFChunker
from src.metadata.metadata_builder import MetadataBuilder
from src.embeddings.embedding_model import EmbeddingModel


def main():
    file_path = "data/raw/GovInst-AI-Whitepaper.pdf"

    # 1. Load
    loader = DocumentLoader(file_path)
    document = loader.load()

    # 2. Clean
    cleaner = PDFTextCleaner()
    document = cleaner.clean(document)

    # 3. Correct encoding
    corrector = PDFEncodingCorrector()
    document = corrector.correct(document)

    # 4. Chunk
    chunker = PDFChunker(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = chunker.chunk(document)

    # 5. Metadata
    metadata_builder = MetadataBuilder()
    chunks = metadata_builder.build(document, chunks)

    # 6. Embeddings
    embedding_model = EmbeddingModel()
    embedded_chunks = embedding_model.encode(chunks)

    # Results
    print()
    print("Number of chunks:", len(embedded_chunks))
    print("Embedding dimension:", embedding_model.dimension())

    first_chunk = embedded_chunks[0]

    print()
    print("--- FIRST EMBEDDED CHUNK ---")
    print("Chunk ID:", first_chunk["chunk_id"])
    print("Page:", first_chunk["page"])
    print("Text:", first_chunk["text"][:200])
    print("Embedding length:", len(first_chunk["embedding"]))
    print("First 10 values:", first_chunk["embedding"][:10])


if __name__ == "__main__":
    main()