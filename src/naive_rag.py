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
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF_PATH = PROJECT_ROOT / "docs" / "1Q26_Naver_Earnings_KOR_vF.pdf"
DEFAULT_DB_DIR = PROJECT_ROOT / "data" / "chroma_naver_1q26"
COLLECTION_NAME = "naver_1q26_earnings"


def load_pdf(pdf_path: Path):
    loader = PyPDFLoader(str(pdf_path))
    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


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
        formatted.append(f"[{page_label}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def create_or_load_retriever(pdf_path: Path, persist_dir: Path, rebuild: bool, k: int):
    has_existing_db = persist_dir.exists() and any(persist_dir.iterdir())
    if rebuild or not has_existing_db:
        documents = load_pdf(pdf_path)
        chunks = split_documents(documents)
        vector_db = build_vector_db(chunks, persist_dir)
    else:
        vector_db = load_vector_db(persist_dir)
    return vector_db.as_retriever(search_kwargs={"k": k})


def answer_question(question: str, pdf_path: Path, persist_dir: Path, rebuild: bool, k: int):
    retriever = create_or_load_retriever(pdf_path, persist_dir, rebuild, k)
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template(
        """너는 네이버 2026년 1분기 실적발표 PDF를 근거로 답하는 분석 assistant야.
주어진 context에 있는 내용만 사용해서 한국어로 답해.
숫자, 기간, 사업부 이름은 원문과 맞게 정확히 적고, 근거 페이지를 함께 표시해.
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


def inspect_pdf(pdf_path: Path):
    documents = load_pdf(pdf_path)
    chunks = split_documents(documents)
    print(f"PDF pages: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print("\nFirst chunk preview:")
    print(chunks[0].page_content[:700])


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
    parser.add_argument("--rebuild", action="store_true", help="Rebuild Chroma DB from the PDF")
    parser.add_argument("--inspect", action="store_true", help="Only load and split the PDF")
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    if args.inspect:
        inspect_pdf(args.pdf)
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
    )
    print(answer)


if __name__ == "__main__":
    main()
