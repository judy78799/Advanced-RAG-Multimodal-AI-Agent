# Advanced RAG Multimodal AI Agent

## 4. Naive RAG 시작하기

이 예제는 `docs/1Q26_Naver_Earnings_KOR_vF.pdf`를 대상으로 가장 기본적인 RAG 흐름을 실행합니다.

1. PDF 문서를 로드합니다.
2. 문서를 청크로 나눕니다.
3. 청크에 OpenAI embedding을 적용합니다.
4. Chroma Vector DB를 생성합니다.
5. Retriever로 관련 청크를 검색합니다.
6. 검색된 문맥을 바탕으로 LLM이 답변합니다.

## 준비

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`.env`에 `OPENAI_API_KEY`를 넣어주세요.

## PDF 로드와 청크 분할 확인

```bash
python src/naive_rag.py --inspect
```

문서 유형을 자동 감지해서 청킹 전략을 고릅니다.

```bash
python src/naive_rag.py --inspect --pdf docs/1Q26_Naver_Earnings_KOR_vF.pdf
python src/naive_rag.py --inspect --pdf docs/KorQuAD1.0KoreanQADatasetforMachineReadingComprehension.pdf
python src/naive_rag.py --inspect --pdf docs/social_culture_doc.pdf
```

전략을 직접 지정할 수도 있습니다.

```bash
python src/naive_rag.py --inspect --chunk-strategy slide
python src/naive_rag.py --inspect --chunk-strategy section --pdf docs/KorQuAD1.0KoreanQADatasetforMachineReadingComprehension.pdf
python src/naive_rag.py --inspect --chunk-strategy recursive --pdf docs/social_culture_doc.pdf
```

## Vector DB 생성 후 질문하기

```bash
python src/naive_rag.py --rebuild --chunk-strategy auto --question "네이버 2026년 1분기 매출과 영업이익은 얼마인가요?"
```

한 번 생성된 Chroma DB는 `data/chroma_naver_1q26`에 저장됩니다. 이후에는 `--rebuild` 없이 질문하면 저장된 DB를 재사용합니다.

문서나 청킹 전략을 바꾸면 기존 Vector DB와 섞이지 않도록 `--rebuild`와 별도 `--db-dir`을 같이 사용하는 것을 권장합니다.

```bash
python src/naive_rag.py \
  --pdf docs/KorQuAD1.0KoreanQADatasetforMachineReadingComprehension.pdf \
  --db-dir data/chroma_korquad \
  --rebuild \
  --chunk-strategy auto \
  --question "KorQuAD 데이터셋의 목적은 무엇인가요?"
```

## 청킹 전략

현재 코드는 문서 유형을 보고 다음 전략 중 하나를 선택합니다.

| 전략 | 적용 문서 | 방식 |
|---|---|---|
| `slide` | 실적 발표, PPT형 PDF | 한 페이지를 하나의 slide parent로 보고 `slide_summary`, `chart_or_table`, `bullet` 청크 생성 |
| `section` | 논문, 리포트형 PDF | heading 후보를 잡고 섹션 단위 문서를 만든 뒤 recursive split |
| `recursive` | 일반 본문 문서 | 문자 길이 기반 recursive split |
| `auto` | 기본값 | 페이지 수, 페이지당 글자 수, 실적/차트 키워드로 전략 자동 선택 |

Slide-aware chunk에는 다음 metadata가 붙습니다.

```text
page
chunk_strategy
section_type
slide_title
unit
periods
source
```

이 metadata는 retriever가 가져온 context 안에서 LLM이 페이지, 단위, chart/table/bullet 여부를 확인하는 데 사용됩니다.

## Agent / Pipeline Diagram

```text
┌──────────────────────┐
│      User Query       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Document Loader     │
│   PyPDFLoader         │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────┐
│        Chunk Strategy Router         │
│                                      │
│  pages, avg chars, keywords inspect  │
└───────┬─────────────┬───────────────┘
        │             │
        │             │
        ▼             ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Slide-aware  │  │ Section-aware│  │  Recursive   │
│ chunking     │  │ chunking     │  │  chunking    │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └─────────────────┴─────────────────┘
                         │
                         ▼
┌──────────────────────────────────────┐
│        Structured Chunk Metadata      │
│ page, title, section_type, unit       │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────┐
│      Embeddings       │
│ text-embedding-3-small│
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│       Chroma DB       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│      Retriever        │
│ top-k relevant chunks │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│       LLM Answer      │
│ grounded Korean QA    │
└──────────────────────┘
```
