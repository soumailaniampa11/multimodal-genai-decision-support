import fitz


pdf_path = "data/raw/GovInst-AI-Whitepaper.pdf"

document = fitz.open(pdf_path)

page = document[2]

text = page.get_text()

print("--- PYMuPDF PAGE 3 ---")
print(text)