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

## Vector DB 생성 후 질문하기

```bash
python src/naive_rag.py --rebuild --question "네이버 2026년 1분기 매출과 영업이익은 얼마인가요?"
```

한 번 생성된 Chroma DB는 `data/chroma_naver_1q26`에 저장됩니다. 이후에는 `--rebuild` 없이 질문하면 저장된 DB를 재사용합니다.
