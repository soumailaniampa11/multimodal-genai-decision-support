from src.ingestion.document_loader import DocumentLoader
from src.cleaning.pdf_cleaner import PDFTextCleaner


pdf_path = "data/raw/GovInst-AI-Whitepaper.pdf"

loader = DocumentLoader(pdf_path)
document = loader.load()

cleaner = PDFTextCleaner()
cleaned_document = cleaner.clean(document)

for page in cleaned_document["pages"][:10]:
    lines = page["text"].split("\n")

    print(f"\n===== PAGE {page['page']} =====")

    print("--- FIRST LINES ---")
    for line in lines[:3]:
        print(line)

    print("--- LAST LINES ---")
    for line in lines[-3:]:
        print(line)