from src.ingestion.document_loader import DocumentLoader
from src.config import load_config


config = load_config()

pdf_path = config["data"]["pdf_path"]

loader = DocumentLoader(pdf_path)

document = loader.load()

print("File name:", document["file_name"])
print("File type:", document["file_type"])
print("Number of pages:", len(document["pages"]))

print("\n--- FIRST PAGE ---")
print(document["pages"][0]["text"][:2000])