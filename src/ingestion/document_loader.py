import re
from pathlib import Path
from typing import Any


class DocumentLoader:
    """
    Load and extract content from supported document formats.

    Every format is returned in the same standardized representation:

        {
            "file_name": ...,
            "file_type": ...,
            "unit": "page" | "slide" | "sheet" | "section",
            "pages": [{"page": 1, "text": ...}, ...],
        }

    "pages" holds the citable units of the document (PDF pages,
    slides, spreadsheet sheets or document sections), so the rest of
    the pipeline can keep page-level provenance for every format.
    """

    IMAGE_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
    }

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".csv",
        ".html",
        ".htm",
        ".txt",
        ".md",
    } | IMAGE_EXTENSIONS

    # Below this number of characters, a PDF page is treated as scanned
    # and its text is recovered with OCR.
    MIN_PDF_PAGE_CHARACTERS = 20

    def __init__(
        self,
        file_path: str,
        ocr_languages: str = "eng+fra",
    ):
        self.file_path = Path(file_path)
        self.ocr_languages = ocr_languages

    @classmethod
    def find_documents(cls, directory: str) -> list[Path]:
        """Return all supported files in a directory, recursively."""

        return sorted(
            path
            for path in Path(directory).rglob("*")
            if path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() in cls.SUPPORTED_EXTENSIONS
        )

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

        if extension == ".pptx":
            return self._load_pptx()

        if extension == ".xlsx":
            return self._load_xlsx()

        if extension == ".csv":
            return self._load_csv()

        if extension in {".html", ".htm"}:
            return self._load_html()

        if extension == ".md":
            return self._load_markdown()

        if extension == ".txt":
            return self._load_txt()

        if extension in self.IMAGE_EXTENSIONS:
            return self._load_image()

        raise ValueError(f"Unsupported file type: {extension}")

    def _document(
        self,
        file_type: str,
        unit: str,
        texts: list[str],
    ) -> dict[str, Any]:
        """Build the standardized representation from unit texts."""

        return {
            "file_name": self.file_path.name,
            "file_type": file_type,
            "unit": unit,
            "pages": [
                {
                    "page": number,
                    "text": text,
                }
                for number, text in enumerate(texts, start=1)
            ],
        }

    def _load_pdf(self) -> dict[str, Any]:
        """
        Extract text from a PDF page by page.

        Pages without a text layer (scanned pages) are processed with OCR.
        """

        from pypdf import PdfReader

        reader = PdfReader(self.file_path)

        texts = []

        for page_index, page in enumerate(reader.pages):
            text = page.extract_text() or ""

            if len(text.strip()) < self.MIN_PDF_PAGE_CHARACTERS:
                try:
                    text = self._ocr_pdf_page(page_index) or text
                except (ImportError, RuntimeError) as exc:
                    print(
                        f"OCR skipped for {self.file_path.name}, "
                        f"page {page_index + 1}: {exc}"
                    )

            texts.append(text)

        return self._document("pdf", "page", texts)

    def _ocr_pdf_page(self, page_index: int) -> str:
        """Render one PDF page as an image and extract its text with OCR."""

        import pymupdf
        from PIL import Image

        with pymupdf.open(self.file_path) as pdf:
            pixmap = pdf[page_index].get_pixmap(dpi=300)

        image = Image.frombytes(
            "RGB",
            (pixmap.width, pixmap.height),
            pixmap.samples,
        )

        return self._ocr(image)

    def _load_docx(self) -> dict[str, Any]:
        """
        Extract paragraphs and tables from a Word document.

        Word files have no stable pages, so the document is split into
        sections, a new section starting at each heading.
        """

        from docx import Document
        from docx.table import Table

        document = Document(self.file_path)

        sections = []
        current = []

        for block in document.iter_inner_content():

            if isinstance(block, Table):
                current.append(self._format_rows(
                    [[cell.text for cell in row.cells] for row in block.rows]
                ))
                continue

            text = block.text.strip()

            if not text:
                continue

            style = (block.style.name or "") if block.style else ""

            if style.startswith(("Heading", "Titre", "Title")) and current:
                sections.append("\n\n".join(current))
                current = []

            current.append(text)

        if current:
            sections.append("\n\n".join(current))

        return self._document("docx", "section", sections)

    def _load_pptx(self) -> dict[str, Any]:
        """Extract text, tables and speaker notes slide by slide."""

        from pptx import Presentation

        presentation = Presentation(self.file_path)

        slides = []

        for slide in presentation.slides:
            parts = []

            for shape in slide.shapes:

                if shape.has_text_frame and shape.text_frame.text.strip():
                    parts.append(shape.text_frame.text.strip())

                if getattr(shape, "has_table", False) and shape.has_table:
                    parts.append(self._format_rows(
                        [
                            [cell.text for cell in row.cells]
                            for row in shape.table.rows
                        ]
                    ))

            if slide.has_notes_slide:
                notes = slide.notes_slide.notes_text_frame.text.strip()

                if notes:
                    parts.append(f"Notes: {notes}")

            slides.append("\n\n".join(parts))

        return self._document("pptx", "slide", slides)

    def _load_xlsx(self) -> dict[str, Any]:
        """Extract every sheet of an Excel workbook."""

        from openpyxl import load_workbook

        workbook = load_workbook(
            self.file_path,
            read_only=True,
            data_only=True,
        )

        sheets = []

        for sheet in workbook.worksheets:
            rows = [
                ["" if value is None else str(value) for value in row]
                for row in sheet.iter_rows(values_only=True)
            ]

            sheets.append(
                f"Sheet: {sheet.title}\n\n{self._format_rows(rows)}"
            )

        workbook.close()

        return self._document("xlsx", "sheet", sheets)

    def _load_csv(self) -> dict[str, Any]:
        """Extract the rows of a CSV file."""

        import pandas as pd

        table = pd.read_csv(
            self.file_path,
            dtype=str,
            keep_default_na=False,
            sep=None,
            engine="python",
        )

        rows = [list(table.columns)] + table.values.tolist()

        return self._document("csv", "sheet", [self._format_rows(rows)])

    def _format_rows(self, rows: list[list[str]]) -> str:
        """
        Convert table rows into text.

        Each row becomes "header: value | header: value", so its meaning
        survives embedding, and rows are separated by blank lines so the
        chunker can split between them.
        """

        rows = [
            [cell.strip() for cell in row]
            for row in rows
            if any(cell.strip() for cell in row)
        ]

        if not rows:
            return ""

        header, *body = rows

        if not body:
            return " | ".join(header)

        lines = []

        for row in body:
            cells = [
                f"{name}: {value}" if name else value
                for name, value in zip(header, row)
                if value
            ]
            lines.append(" | ".join(cells))

        return "\n\n".join(lines)

    def _load_html(self) -> dict[str, Any]:
        """Extract the visible text of an HTML page."""

        import lxml.html

        tree = lxml.html.parse(str(self.file_path)).getroot()

        for element in tree.xpath(
            "//script | //style | //noscript | //nav | //footer"
        ):
            element.drop_tree()

        blocks = [
            " ".join(element.text_content().split())
            for element in tree.xpath(
                "//h1 | //h2 | //h3 | //h4 | //p | //li | //td | //th"
                " | //pre | //blockquote"
            )
        ]

        text = "\n\n".join(block for block in blocks if block)

        if not text:
            text = " ".join(tree.text_content().split())

        return self._document("html", "page", [text])

    def _load_markdown(self) -> dict[str, Any]:
        """Read a Markdown file, one section per heading."""

        text = self.file_path.read_text(encoding="utf-8")

        sections = [
            section.strip()
            for section in re.split(r"\n(?=#{1,6} )", text)
            if section.strip()
        ]

        return self._document("md", "section", sections)

    def _load_txt(self) -> dict[str, Any]:
        """Read a text file."""

        text = self.file_path.read_text(encoding="utf-8")

        return self._document("txt", "page", [text])

    def _load_image(self) -> dict[str, Any]:
        """Extract the text of an image with OCR."""

        from PIL import Image

        with Image.open(self.file_path) as image:
            text = self._ocr(image.convert("RGB"))

        return self._document("image", "page", [text])

    def _ocr(self, image) -> str:
        """Extract text from an image with Tesseract."""

        import pytesseract

        try:
            return pytesseract.image_to_string(
                image,
                lang=self.ocr_languages,
            )
        except pytesseract.TesseractNotFoundError as exc:
            raise RuntimeError(
                "OCR requires Tesseract. Install it with: "
                "brew install tesseract"
            ) from exc
