from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


ChunkStrategy = Literal["auto", "slide", "section", "recursive"]

PERIOD_RE = re.compile(r"\b(?:[1-4]Q\d{2}|FY\d{4}|\d{4}\.\d{1,2}\.\d{1,2})\b")
AMOUNT_RE = re.compile(r"(?<!\d)(?:\d{1,3}(?:,\d{3})+|\d{1,3}\.\d+|\d{4,})(?:%|%p|조 원|십억 원)?(?!\d)")
SECTION_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+|[A-Z][A-Za-z ]{2,}:?\s*$|[가-힣A-Za-z][^.!?]{2,60}$)"
)

SLIDE_KEYWORDS = [
    "실적발표",
    "분기 별",
    "YoY",
    "QoQ",
    "Y/Y",
    "Q/Q",
    "단위:",
    "매출",
    "영업이익",
]
CHART_KEYWORDS = ["그래프", "차트", "분기 별", "YoY", "QoQ", "Y/Y", "Q/Q"]
TABLE_KEYWORDS = ["단위:", "구분", "1Q", "2Q", "3Q", "4Q", "FY", "Y/Y", "Q/Q"]


@dataclass(frozen=True)
class DocumentProfile:
    page_count: int
    avg_chars_per_page: int
    detected_strategy: str
    reason: str


def detect_document_profile(documents: list[Document]) -> DocumentProfile:
    page_count = len(documents)
    avg_chars = int(sum(len(doc.page_content) for doc in documents) / max(page_count, 1))
    sample_text = "\n".join(doc.page_content[:1200] for doc in documents[: min(5, page_count)])
    keyword_hits = sum(1 for keyword in SLIDE_KEYWORDS if keyword in sample_text)

    if page_count <= 40 and avg_chars <= 900 and keyword_hits >= 2:
        return DocumentProfile(
            page_count=page_count,
            avg_chars_per_page=avg_chars,
            detected_strategy="slide",
            reason="short slide-like pages with earnings/chart keywords",
        )

    if avg_chars >= 2200 or "Abstract" in sample_text or "Introduction" in sample_text:
        return DocumentProfile(
            page_count=page_count,
            avg_chars_per_page=avg_chars,
            detected_strategy="section",
            reason="dense report/paper-like pages",
        )

    return DocumentProfile(
        page_count=page_count,
        avg_chars_per_page=avg_chars,
        detected_strategy="recursive",
        reason="general text document",
    )


def split_documents(documents: list[Document], strategy: ChunkStrategy = "auto") -> list[Document]:
    if strategy == "auto":
        strategy = detect_document_profile(documents).detected_strategy  # type: ignore[assignment]

    if strategy == "slide":
        return slide_aware_split_documents(documents)
    if strategy == "section":
        return section_aware_split_documents(documents)
    if strategy == "recursive":
        return recursive_split_documents(documents)

    raise ValueError(f"Unknown chunk strategy: {strategy}")


def recursive_split_documents(
    documents: list[Document],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    strategy_name: str = "recursive",
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "chunk_strategy": strategy_name,
                "chunk_index": index,
                "section_type": chunk.metadata.get("section_type", "text"),
            }
        )
    return chunks


def slide_aware_split_documents(documents: list[Document]) -> list[Document]:
    chunks: list[Document] = []

    for doc in documents:
        page = doc.metadata.get("page")
        source = doc.metadata.get("source")
        text = normalize_text(doc.page_content)
        lines = nonempty_lines(text)
        slide_title = infer_slide_title(lines)
        unit = infer_unit(lines)
        periods = infer_periods(text)

        base_metadata = {
            **doc.metadata,
            "source": source,
            "page": page,
            "chunk_strategy": "slide",
            "slide_title": slide_title,
            "unit": unit,
            "periods": ", ".join(periods),
        }

        if text:
            chunks.append(
                Document(
                    page_content=format_slide_chunk(slide_title, "slide_summary", text),
                    metadata={**base_metadata, "section_type": "slide_summary"},
                )
            )

        chart_lines = select_chart_or_table_lines(lines)
        if chart_lines:
            chunks.append(
                Document(
                    page_content=format_slide_chunk(slide_title, "chart_or_table", "\n".join(chart_lines)),
                    metadata={**base_metadata, "section_type": "chart_or_table"},
                )
            )

        bullet_lines = [line for line in lines if is_bullet_line(line)]
        if bullet_lines:
            chunks.append(
                Document(
                    page_content=format_slide_chunk(slide_title, "bullet", "\n".join(bullet_lines)),
                    metadata={**base_metadata, "section_type": "bullet"},
                )
            )

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index

    return chunks


def section_aware_split_documents(documents: list[Document]) -> list[Document]:
    section_docs: list[Document] = []

    for doc in documents:
        text = normalize_text(doc.page_content)
        lines = nonempty_lines(text)
        if not lines:
            continue

        section_docs.append(make_section_doc(doc, infer_section_title(lines), lines))

    return recursive_split_documents(
        section_docs,
        chunk_size=1400,
        chunk_overlap=220,
        strategy_name="section",
    )


def make_section_doc(source_doc: Document, section_title: str, lines: list[str]) -> Document:
    return Document(
        page_content="\n".join(lines),
        metadata={
            **source_doc.metadata,
            "chunk_strategy": "section",
            "section_type": "section",
            "section_title": section_title,
        },
    )


def normalize_text(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def infer_slide_title(lines: list[str]) -> str:
    for line in lines[:8]:
        if len(line) <= 80 and not AMOUNT_RE.search(line):
            return line
    return lines[0][:80] if lines else "Untitled slide"


def infer_section_title(lines: list[str]) -> str:
    known_headings = {
        "Abstract",
        "Introduction",
        "Related Work",
        "Dataset",
        "Experiments",
        "Conclusion",
        "References",
    }
    for line in lines[:12]:
        if line in known_headings or re.match(r"^\d+(?:\.\d+)*\.?\s+\S+", line):
            return line[:100]
    return lines[0][:100]


def infer_unit(lines: list[str]) -> str:
    for line in lines:
        if "단위" in line:
            return line[:120]
    for line in lines:
        if "십억 원" in line or "조 원" in line or "백만" in line:
            return line[:120]
    return ""


def infer_periods(text: str) -> list[str]:
    counts = Counter(PERIOD_RE.findall(text))
    return [period for period, _ in counts.most_common(12)]


def is_bullet_line(line: str) -> bool:
    return line.startswith(("•", "-", "*", "·")) or line[:2] in {"1)", "2)", "3)", "4)"}


def is_section_heading(line: str) -> bool:
    if len(line) > 100:
        return False
    if line.endswith(".") and len(line.split()) > 8:
        return False
    return bool(SECTION_HEADING_RE.match(line))


def select_chart_or_table_lines(lines: list[str]) -> list[str]:
    selected: list[str] = []
    for line in lines:
        if is_date_only_line(line):
            continue
        has_chart_keyword = any(keyword in line for keyword in CHART_KEYWORDS)
        has_table_keyword = any(keyword in line for keyword in TABLE_KEYWORDS)
        amount_count = len(AMOUNT_RE.findall(line))
        period_count = len(PERIOD_RE.findall(line))
        if has_chart_keyword or has_table_keyword or amount_count >= 2 or period_count >= 2:
            selected.append(line)
    return selected


def format_slide_chunk(slide_title: str, section_type: str, content: str) -> str:
    return f"Slide title: {slide_title}\nSection type: {section_type}\n\n{content}"


def is_date_only_line(line: str) -> bool:
    normalized = line.strip()
    return bool(re.fullmatch(r"\d{4}\.\d{1,2}\.\d{1,2}", normalized))
