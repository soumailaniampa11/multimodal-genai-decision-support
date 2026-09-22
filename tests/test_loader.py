from src.ingestion.document_loader import DocumentLoader


pdf_path = "data/raw/GovInst-AI-Whitepaper.pdf"

loader = DocumentLoader(pdf_path)

document = loader.load()

print("File name:", document["file_name"])
print("File type:", document["file_type"])
print("Title:", document["title"])
print("Author:", document["author"])
print("Subject:", document["subject"])
print("Number of pages:", document["number_of_pages"])

print("\n--- FIRST PAGE ---")
print(document["pages"][0]["text"][:2000])