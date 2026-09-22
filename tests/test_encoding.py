from collections import Counter

from src.ingestion.document_loader import DocumentLoader
from src.cleaning.pdf_cleaner import PDFTextCleaner


pdf_path = "data/raw/GovInst-AI-Whitepaper.pdf"

# Load PDF
loader = DocumentLoader(pdf_path)
document = loader.load()

# Clean PDF
cleaner = PDFTextCleaner()
cleaned_document = cleaner.clean(document)

# Combine all extracted text
full_text = "\n".join(
    page["text"]
    for page in cleaned_document["pages"]
)

# Count suspicious characters
suspicious_characters = Counter(
    character
    for character in full_text
    if ord(character) > 127
)

print("Number of suspicious characters:", len(suspicious_characters))

print("\n--- SUSPICIOUS CHARACTERS ---")

for character, count in suspicious_characters.most_common():
    print(
        f"Character: {repr(character)} | "
        f"Unicode: U+{ord(character):04X} | "
        f"Count: {count}"
    )

print("\n--- EXAMPLES ---")

suspicious_words = [
    word
    for word in full_text.split()
    if "À" in word or "੔" in word or "੕" in word
]

for word in suspicious_words[:30]:
    print(word)