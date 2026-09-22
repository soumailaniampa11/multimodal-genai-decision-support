import re
from typing import Any


class PDFEncodingCorrector:
    """
    Correct common glyph extraction issues found in PDF documents.

    Some PDFs encode certain characters incorrectly during text
    extraction. This class applies controlled word-level corrections.
    """

    # Cases where "À" represents "f" or "fi".
    FI_CORRECTIONS = {
        "ArtiÀcial": "Artificial",
        "artiÀcial": "artificial",
        "artiÀciality": "artificiality",

        "not-for-proÀt": "not-for-profit",
        "not-for-proÀts": "not-for-profits",

        "certiÀcates": "certificates",

        "redeÀning": "redefining",
        "redeÀne": "redefine",
        "reÀning": "refining",

        "beneÀt": "benefit",
        "beneÀts": "benefits",

        "signiÀcant": "significant",
        "signiÀcantly": "significantly",

        "speciÀc": "specific",
        "speciÀcations": "specifications",

        "conÀdent": "confident",
        "conÀdence": "confidence",
        "conÀdentiality": "confidentiality",

        "identiÀed": "identified",
        "identiÀcation": "identification",

        "justiÀcation": "justification",

        "Ànancial": "financial",
        "Àelds": "fields",

        "proÀle": "profile",
        "proÀcient": "proficient",

        "deÀned": "defined",
        "deÀnes": "defines",

        "Àndings": "findings",
        "Àrst": "first",

        "Àrm": "firm",
        "Àrms": "firms",

        "rectiÀcation": "rectification",

        "Àne-tune": "fine-tune",
        "Àne-tuning": "fine-tuning",

        "diversiÀed": "diversified",
        "simpliÀes": "simplifies",

        "size-Àts-all": "size-fits-all",
    }

    # Cases where the corrupted glyph represents "ff".
    FF_CORRECTIONS = {
        "o੔er": "offer",
        "o੔ers": "offers",
        "o੔ering": "offering",
        "o੔erings": "offerings",

        "e੔ects": "effects",
        "e੔ect": "effect",

        "E੔ective": "Effective",
        "e੔ective": "effective",
        "e੔ectively": "effectively",

        "a੔ect": "affect",
        "a੔ected": "affected",
        "a੔ord": "afford",
        "a੔ecting": "affecting",

        "di੔er": "differ",
        "di੔erent": "different",
        "di੔erences": "differences",
        "Di੔erences": "Differences",
        "di੔erentiate": "differentiate",

        "cost-e੔ectiveness": "cost-effectiveness",

        "o੕ces": "offices",
        "O੕ce": "Office",
        "o੕cers": "officers",
        "O੕cer": "Officer",

        "e੕ciency": "efficiency",
        "e੕ciencies": "efficiencies",
        "e੕ciently": "efficiently",
        "di੕cult": "difficult",

        "ine੕ciency": "inefficiency",

        "Mo੔att": "Moffatt",
    }

    def correct(self, document: dict[str, Any]) -> dict[str, Any]:
        """
        Correct encoding issues in all pages.
        """

        corrected_pages = []

        for page in document["pages"]:

            corrected_text = self._correct_text(
                page["text"]
            )

            corrected_pages.append(
                {
                    "page": page["page"],
                    "text": corrected_text,
                }
            )

        corrected_document = document.copy()
        corrected_document["pages"] = corrected_pages

        return corrected_document

    def _correct_text(self, text: str) -> str:
        """
        Apply controlled word-level corrections.
        """

        # Apply corrections where "À" is corrupted.
        for corrupted, corrected in self.FI_CORRECTIONS.items():

            text = re.sub(
                rf"(?<!\w){re.escape(corrupted)}(?!\w)",
                corrected,
                text,
            )

        # Apply corrections where the glyph represents "ff".
        for corrupted, corrected in self.FF_CORRECTIONS.items():

            text = re.sub(
                rf"(?<!\w){re.escape(corrupted)}(?!\w)",
                corrected,
                text,
            )

        return text