from src.ingestion.document_loader import DocumentLoader
from src.cleaning.pdf_cleaner import PDFTextCleaner
from src.cleaning.encoding_corrector import PDFEncodingCorrector
from src.chunking.pdf_chunker import PDFChunker
from src.metadata.metadata_builder import MetadataBuilder


pdf_path = "data/raw/GovInst-AI-Whitepaper.pdf"


# 1. Load PDF
loader = DocumentLoader(pdf_path)
document = loader.load()


# 2. Clean PDF
cleaner = PDFTextCleaner()
cleaned_document = cleaner.clean(document)


# 3. Correct encoding
corrector = PDFEncodingCorrector()
corrected_document = corrector.correct(
    cleaned_document
)


# 4. Chunk document
chunker = PDFChunker(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = chunker.chunk(
    corrected_document
)


# 5. Build metadata
metadata_builder = MetadataBuilder()

enriched_chunks = metadata_builder.build(
    corrected_document,
    chunks,
)


print("Number of chunks:", len(enriched_chunks))

print("\n--- FIRST CHUNK ---")

first_chunk = enriched_chunks[0]

for key, value in first_chunk.items():

    if key == "text":
        print(f"{key}:")
        print(value)
    else:
        print(f"{key}: {value}")


print("\n--- SECOND CHUNK ---")

second_chunk = enriched_chunks[1]

for key, value in second_chunk.items():

    if key == "text":
        print(f"{key}:")
        print(value)
    else:
        print(f"{key}: {value}")