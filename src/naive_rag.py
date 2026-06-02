from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    from chunking import ChunkStrategy, detect_document_profile, split_documents
except ImportError:
    from src.chunking import ChunkStrategy, detect_document_profile, split_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF_PATH = PROJECT_ROOT / "docs" / "1Q26_Naver_Earnings_KOR_vF.pdf"
DEFAULT_DB_DIR = PROJECT_ROOT / "data" / "chroma_naver_1q26"
COLLECTION_NAME = "naver_1q26_earnings"


def load_pdf(pdf_path: Path):
    loader = PyPDFLoader(str(pdf_path))
    return loader.load()


def build_vector_db(chunks, persist_dir: Path):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_dir),
    )
    if hasattr(vector_db, "persist"):
        vector_db.persist()
    return vector_db


def load_vector_db(persist_dir: Path):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def format_context(documents):
    formatted = []
    for doc in documents:
        page = doc.metadata.get("page")
        page_label = f"p.{page + 1}" if isinstance(page, int) else "page unknown"
        metadata_lines = [
            f"page={page_label}",
            f"strategy={doc.metadata.get('chunk_strategy', '')}",
            f"section_type={doc.metadata.get('section_type', '')}",
        ]
        if doc.metadata.get("slide_title"):
            metadata_lines.append(f"slide_title={doc.metadata['slide_title']}")
        if doc.metadata.get("section_title"):
            metadata_lines.append(f"section_title={doc.metadata['section_title']}")
        if doc.metadata.get("unit"):
            metadata_lines.append(f"unit={doc.metadata['unit']}")
        if doc.metadata.get("periods"):
            metadata_lines.append(f"periods={doc.metadata['periods']}")

        metadata_header = " | ".join(line for line in metadata_lines if not line.endswith("="))
        formatted.append(f"[{metadata_header}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def create_or_load_retriever(
    pdf_path: Path,
    persist_dir: Path,
    rebuild: bool,
    k: int,
    chunk_strategy: ChunkStrategy,
):
    has_existing_db = persist_dir.exists() and any(persist_dir.iterdir())
    if rebuild or not has_existing_db:
        documents = load_pdf(pdf_path)
        chunks = split_documents(documents, strategy=chunk_strategy)
        vector_db = build_vector_db(chunks, persist_dir)
    else:
        vector_db = load_vector_db(persist_dir)
    return vector_db.as_retriever(search_kwargs={"k": k})


def answer_question(
    question: str,
    pdf_path: Path,
    persist_dir: Path,
    rebuild: bool,
    k: int,
    chunk_strategy: ChunkStrategy,
):
    retriever = create_or_load_retriever(pdf_path, persist_dir, rebuild, k, chunk_strategy)
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        """너는 네이버 2026년 1분기 실적발표 PDF를 근거로 답하는 분석 assistant야.
주어진 context에 있는 내용만 사용해서 한국어로 답해.
숫자, 기간, 사업부 이름, 단위는 원문과 맞게 정확히 적고, 근거 페이지를 함께 표시해.
context의 metadata-like header에 slide title, section type, unit이 있으면 이를 답변 검증에 활용해.
context에 없는 내용이면 모른다고 말해.

Question:
{question}

Context:
{context}
"""
    )

    chain = (
        {
            "context": retriever | format_context,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(question)


def inspect_pdf(pdf_path: Path, chunk_strategy: ChunkStrategy):
    documents = load_pdf(pdf_path)
    profile = detect_document_profile(documents)
    chunks = split_documents(documents, strategy=chunk_strategy)
    effective_strategy = profile.detected_strategy if chunk_strategy == "auto" else chunk_strategy
    print(f"PDF pages: {len(documents)}")
    print(f"Avg chars/page: {profile.avg_chars_per_page}")
    print(f"Detected strategy: {profile.detected_strategy} ({profile.reason})")
    print(f"Effective strategy: {effective_strategy}")
    print(f"Chunks: {len(chunks)}")
    for index, chunk in enumerate(chunks[:5], start=1):
        print(f"\n--- chunk {index} ---")
        print(
            {
                "page": chunk.metadata.get("page"),
                "strategy": chunk.metadata.get("chunk_strategy"),
                "section_type": chunk.metadata.get("section_type"),
                "slide_title": chunk.metadata.get("slide_title"),
                "section_title": chunk.metadata.get("section_title"),
                "unit": chunk.metadata.get("unit"),
            }
        )
        print(chunk.page_content[:700])


def parse_args():
    parser = argparse.ArgumentParser(description="Naive RAG for NAVER 1Q26 earnings PDF")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF_PATH,
        help="Path to the earnings PDF",
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=DEFAULT_DB_DIR,
        help="Directory where Chroma DB is stored",
    )
    parser.add_argument(
        "--question",
        default="네이버 2026년 1분기 매출과 영업이익은 얼마인가요?",
        help="Question to ask the RAG pipeline",
    )
    parser.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve")
    parser.add_argument(
        "--chunk-strategy",
        choices=["auto", "slide", "section", "recursive"],
        default="auto",
        help="Chunking strategy. auto detects slide/report/general documents.",
    )
    parser.add_argument("--rebuild", action="store_true", help="Rebuild Chroma DB from the PDF")
    parser.add_argument("--inspect", action="store_true", help="Only load and split the PDF")
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    if args.inspect:
        inspect_pdf(args.pdf, args.chunk_strategy)
        return

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Copy .env.example to .env and set your API key first."
        )

    answer = answer_question(
        question=args.question,
        pdf_path=args.pdf,
        persist_dir=args.db_dir,
        rebuild=args.rebuild,
        k=args.k,
        chunk_strategy=args.chunk_strategy,
    )
    print(answer)


if __name__ == "__main__":
    main()
