from __future__ import annotations

import io
from typing import Iterable, List, Sequence, Tuple

from pypdf import PdfReader


PageText = Tuple[int, str]


def extract_pdf_pages(pdf_bytes: bytes) -> List[PageText]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages: List[PageText] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((page_index, text.strip()))
    return pages


def combine_selected_pages(pages: Sequence[PageText], selected_page_numbers: Iterable[int]) -> str:
    selected = set(selected_page_numbers)
    chunks: List[str] = []
    for page_number, text in pages:
        if page_number in selected and text:
            chunks.append(f"[Page {page_number}]\n{text}")
    return "\n\n".join(chunks).strip()
