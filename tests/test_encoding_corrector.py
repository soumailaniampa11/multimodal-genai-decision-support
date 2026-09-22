from collections import Counter

from src.ingestion.document_loader import DocumentLoader
from src.cleaning.pdf_cleaner import PDFTextCleaner
from src.cleaning.encoding_corrector import PDFEncodingCorrector


pdf_path = "data/raw/GovInst-AI-Whitepaper.pdf"

# 1. Load
loader = DocumentLoader(pdf_path)
document = loader.load()

# 2. Clean
cleaner = PDFTextCleaner()
cleaned_document = cleaner.clean(document)

# 3. Correct encoding
corrector = PDFEncodingCorrector()
corrected_document = corrector.correct(cleaned_document)

# Combine all pages
full_text = "\n".join(
    page["text"]
    for page in corrected_document["pages"]
)

# Count non-ASCII characters
non_ascii = Counter(
    character
    for character in full_text
    if ord(character) > 127
)

print("Number of non-ASCII characters:", len(non_ascii))

print("\n--- NON-ASCII CHARACTERS ---")

for character, count in non_ascii.most_common():
    print(
        f"Character: {repr(character)} | "
        f"Unicode: U+{ord(character):04X} | "
        f"Count: {count}"
    )

print("\n--- REMAINING SUSPICIOUS WORDS ---")

suspicious_words = [
    word
    for word in full_text.split()
    if "À" in word
    or "੔" in word
    or "੕" in word
]

if suspicious_words:
    for word in suspicious_words[:50]:
        print(word)
else:
    print("No known corrupted glyphs found.")