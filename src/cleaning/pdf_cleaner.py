import re
from collections import Counter
from typing import Any


class PDFTextCleaner:
    """
    Clean text extracted from PDF documents.

    The cleaner:
    - normalizes whitespace
    - detects repeated headers and footers
    - removes repeated headers and footers
    - preserves page numbers as metadata
    """

    def __init__(self, repetition_threshold: float = 0.3):
        """
        Args:
            repetition_threshold:
                Minimum proportion of pages on which a line must appear
                to be considered a repeated header or footer.
        """
        self.repetition_threshold = repetition_threshold

    def clean(self, document: dict[str, Any]) -> dict[str, Any]:
        """
        Clean all pages of a PDF document.
        """

        repeated_lines = self._detect_repeated_lines(document)

        cleaned_pages = []

        for page in document["pages"]:
            cleaned_text = self._clean_page(
                page["text"],
                repeated_lines,
            )

            cleaned_pages.append(
                {
                    "page": page["page"],
                    "text": cleaned_text,
                }
            )

        cleaned_document = document.copy()
        cleaned_document["pages"] = cleaned_pages

        return cleaned_document

    def _detect_repeated_lines(
        self,
        document: dict[str, Any],
    ) -> set[str]:
        """
        Detect lines repeated across many pages.

        Only the beginning and end of each page are inspected,
        because headers and footers normally appear there.
        """

        counter = Counter()

        total_pages = len(document["pages"])

        for page in document["pages"]:

            lines = [
                self._normalize_header_footer(line)
                for line in page["text"].split("\n")
            ]

            lines = [
                line
                for line in lines
                if line
            ]

            # Inspect the first 3 and last 3 lines
            boundary_lines = lines[:3] + lines[-3:]

            # Count a line only once per page
            for line in set(boundary_lines):
                counter[line] += 1

        threshold = total_pages * self.repetition_threshold

        repeated_lines = {
            line
            for line, count in counter.items()
            if count >= threshold
        }

        return repeated_lines

    def _clean_page(
        self,
        text: str,
        repeated_lines: set[str],
    ) -> str:
        """
        Clean a single page.
        """

        # Normalize line endings
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        lines = []

        for line in text.split("\n"):

            normalized_line = self._normalize_line(line)

            if not normalized_line:
                continue

            normalized_boundary_line = (
                self._normalize_header_footer(line)
            )

            if normalized_boundary_line in repeated_lines:
                continue

            lines.append(normalized_line)

        return "\n".join(lines)

    def _normalize_line(self, line: str) -> str:
        """
        Normalize whitespace inside a line.
        """

        line = line.strip()

        line = re.sub(r"[ \t]+", " ", line)

        return line

    def _normalize_header_footer(self, line: str) -> str:
        """
        Normalize a potential header or footer.

        This also removes page numbers that may be attached
        to the beginning or end of a PDF header/footer.
        """

        line = self._normalize_line(line)

        # Remove page number at the beginning.
        line = re.sub(r"^\d+\s*", "", line)

        # Remove page number at the end.
        line = re.sub(r"\s*\d+$", "", line)

        return line