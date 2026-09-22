import re
from typing import Any


class PDFChunker:
    """
    Split cleaned PDF text into semantically coherent chunks.

    The chunker:
    - works page by page
    - preserves paragraphs
    - splits large paragraphs into sentences
    - groups sentences until the target chunk size is reached
    - preserves page metadata
    - creates sentence-aware overlap
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Args:
            chunk_size:
                Target maximum number of characters per chunk.

            chunk_overlap:
                Approximate number of characters shared between
                consecutive chunks.
        """

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than 0"
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative"
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        document: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Convert a cleaned PDF document into chunks.
        """

        chunks = []

        chunk_number = 1

        for page in document["pages"]:

            page_number = page["page"]
            text = page["text"].strip()

            if not text:
                continue

            paragraphs = self._split_into_paragraphs(text)

            page_chunks = self._build_chunks(paragraphs)

            for chunk_text in page_chunks:

                chunks.append(
                    {
                        "chunk_id": f"chunk_{chunk_number:04d}",
                        "document_id": document["file_name"],
                        "page": page_number,
                        "text": chunk_text,
                    }
                )

                chunk_number += 1

        return chunks

    def _split_into_paragraphs(
        self,
        text: str,
    ) -> list[str]:
        """
        Split page text into paragraphs.
        """

        paragraphs = text.split("\n\n")

        cleaned_paragraphs = []

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if paragraph:
                cleaned_paragraphs.append(paragraph)

        return cleaned_paragraphs

    def _build_chunks(
        self,
        paragraphs: list[str],
    ) -> list[str]:
        """
        Build chunks while preserving sentence boundaries.
        """

        chunks = []

        current_chunk = ""

        for paragraph in paragraphs:

            # Normal paragraph: try to add it directly.
            if len(paragraph) <= self.chunk_size:

                if not current_chunk:

                    current_chunk = paragraph

                    continue

                candidate = (
                    current_chunk
                    + "\n\n"
                    + paragraph
                )

                if len(candidate) <= self.chunk_size:

                    current_chunk = candidate

                else:

                    chunks.append(current_chunk)

                    overlap = self._create_overlap(
                        current_chunk
                    )

                    current_chunk = (
                        overlap
                        + "\n\n"
                        + paragraph
                        if overlap
                        else paragraph
                    )

                continue

            # Large paragraph:
            # split it into sentences first.
            sentences = self._split_into_sentences(
                paragraph
            )

            for sentence in sentences:

                sentence = sentence.strip()

                if not sentence:
                    continue

                # A single sentence can still be larger
                # than the target chunk size.
                if len(sentence) > self.chunk_size:

                    if current_chunk:

                        chunks.append(current_chunk)
                        current_chunk = ""

                    large_chunks = (
                        self._split_large_sentence(
                            sentence
                        )
                    )

                    chunks.extend(large_chunks)

                    continue

                if not current_chunk:

                    current_chunk = sentence

                    continue

                candidate = (
                    current_chunk
                    + " "
                    + sentence
                )

                if len(candidate) <= self.chunk_size:

                    current_chunk = candidate

                else:

                    chunks.append(current_chunk)

                    overlap = self._create_overlap(
                        current_chunk
                    )

                    current_chunk = (
                        overlap
                        + " "
                        + sentence
                        if overlap
                        else sentence
                    )

        if current_chunk:

            chunks.append(current_chunk)

        return chunks

    def _split_into_sentences(
        self,
        text: str,
    ) -> list[str]:
        """
        Split text into sentences.

        The rule looks for sentence-ending punctuation:
        '.', '!' or '?' followed by whitespace.
        """

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text,
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    def _split_large_sentence(
        self,
        sentence: str,
    ) -> list[str]:
        """
        Split an exceptionally large sentence.

        We first try to split at commas, semicolons,
        colons or whitespace before falling back
        to a hard character boundary.
        """

        chunks = []

        remaining = sentence.strip()

        while len(remaining) > self.chunk_size:

            candidate = remaining[:self.chunk_size]

            # Look for a natural boundary near the end
            # of the allowed chunk.
            boundary_match = list(
                re.finditer(
                    r"[,;:]\s|\s",
                    candidate,
                )
            )

            if boundary_match:

                boundary = boundary_match[-1].end()

            else:

                boundary = self.chunk_size

            chunk = remaining[:boundary].strip()

            if chunk:
                chunks.append(chunk)

            remaining = remaining[boundary:].strip()

        if remaining:
            chunks.append(remaining)

        return chunks

    def _create_overlap(
        self,
        text: str,
    ) -> str:
        """
        Create an overlap from the end of the previous chunk.

        The overlap is sentence-aware whenever possible.
        """

        if self.chunk_overlap <= 0:
            return ""

        if len(text) <= self.chunk_overlap:
            return text

        candidate = text[-self.chunk_overlap:]

        # Try to start the overlap at a sentence boundary.
        sentence_boundary = re.search(
            r"[.!?]\s+",
            candidate,
        )

        if sentence_boundary:

            return candidate[
                sentence_boundary.end():
            ].strip()

        # Otherwise start at the beginning of a word.
        word_boundary = candidate.find(" ")

        if word_boundary != -1:

            return candidate[
                word_boundary + 1:
            ].strip()

        return candidate.strip()