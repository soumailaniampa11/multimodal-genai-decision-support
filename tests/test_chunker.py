from src.ingestion.document_loader import DocumentLoader
from src.cleaning.pdf_cleaner import PDFTextCleaner
from src.cleaning.encoding_corrector import PDFEncodingCorrector
from src.chunking.pdf_chunker import PDFChunker


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


print("File:", corrected_document["file_name"])
print(
    "Number of pages:",
    corrected_document["number_of_pages"],
)

print(
    "Number of chunks:",
    len(chunks)
)


print("\n--- FIRST 5 CHUNKS ---")


for chunk in chunks[:5]:

    print("\n==============================")

    print("Chunk ID:", chunk["chunk_id"])
    print("Document:", chunk["document_id"])
    print("Page:", chunk["page"])
    print("Characters:", len(chunk["text"]))

    print("\nText:")
    print(chunk["text"])