from pathlib import Path
from typing import Any


class DocumentLoader:
    """
    Load and extract content from supported document formats.
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".txt",
        ".png",
        ".jpg",
        ".jpeg",
    }

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def validate(self) -> None:
        """Validate that the file exists and has a supported extension."""

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"File not found: {self.file_path}"
            )

        if self.file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {self.file_path.suffix}"
            )

    def load(self) -> dict[str, Any]:
        """
        Load the document and return a standardized representation.
        """

        self.validate()

        extension = self.file_path.suffix.lower()

        if extension == ".pdf":
            return self._load_pdf()

        if extension == ".docx":
            return self._load_docx()

        if extension == ".txt":
            return self._load_txt()

        if extension in {".png", ".jpg", ".jpeg"}:
            return self._load_image()

        raise ValueError(f"Unsupported file type: {extension}")

    def _load_pdf(self) -> dict[str, Any]:
        """Extract text and metadata from a PDF page by page."""

        from pypdf import PdfReader

        reader = PdfReader(self.file_path)

        metadata = reader.metadata or {}

        pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

        return {
            "file_name": self.file_path.name,
            "file_type": "pdf",
            "title": metadata.get("/Title"),
            "author": metadata.get("/Author"),
            "subject": metadata.get("/Subject"),
            "number_of_pages": len(reader.pages),
            "pages": pages,
        }

    def _load_docx(self) -> dict[str, Any]:
        """Extract paragraphs from a Word document."""

        from docx import Document

        document = Document(self.file_path)

        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        return {
            "file_name": self.file_path.name,
            "file_type": "docx",
            "paragraphs": paragraphs,
        }

    def _load_txt(self) -> dict[str, Any]:
        """Read a text file."""

        text = self.file_path.read_text(encoding="utf-8")

        return {
            "file_name": self.file_path.name,
            "file_type": "txt",
            "text": text,
        }

    def _load_image(self) -> dict[str, Any]:
        """Load image information."""

        from PIL import Image

        image = Image.open(self.file_path)

        return {
            "file_name": self.file_path.name,
            "file_type": "image",
            "image_path": str(self.file_path),
            "width": image.width,
            "height": image.height,
            "format": image.format,
        }