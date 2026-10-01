from src.ingestion.document_loader import DocumentLoader
from src.cleaning.pdf_cleaner import PDFTextCleaner
from src.config import load_config


config = load_config()

pdf_path = config["data"]["pdf_path"]

# Load PDF
loader = DocumentLoader(pdf_path)
document = loader.load()

# Clean PDF
cleaner = PDFTextCleaner()

cleaned_document = cleaner.clean(document)

print("File name:", cleaned_document["file_name"])
print("Number of pages:", len(cleaned_document["pages"]))

print("\n--- ORIGINAL PAGE 3 ---")
print(document["pages"][2]["text"])

print("\n--- CLEANED PAGE 3 ---")
print(cleaned_document["pages"][2]["text"])

print("\n--- ORIGINAL PAGE 4 ---")
print(document["pages"][3]["text"])

print("\n--- CLEANED PAGE 4 ---")
print(cleaned_document["pages"][3]["text"])