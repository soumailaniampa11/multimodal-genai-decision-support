import fitz
from src.config import load_config


config = load_config()

pdf_path = config["data"]["pdf_path"]

document = fitz.open(pdf_path)

page = document[2]

text = page.get_text()

print("--- PYMuPDF PAGE 3 ---")
print(text)