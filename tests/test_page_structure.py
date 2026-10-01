from src.ingestion.document_loader import DocumentLoader
from src.cleaning.pdf_cleaner import PDFTextCleaner
from src.config import load_config


config = load_config()

pdf_path = config["data"]["pdf_path"]

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