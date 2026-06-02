# Week5: Hybrid Search + Re-ranking

---

# 주제
제목: Retrieval 고도화 Hybrid Search + Re-ranking: 왜 필요한가, 어떻게 구현/검증했나  
목표: Retrieval 문제 정의 → 파이프라인 설계 → 실험(ablations) 결과 해석 및 권고

---

# 1) 문제 정의 — Retrieval이 어려운 이유 
핵심: Retrieval 실패는 downstream LLM 오류와 hallucination의 주원인.
- 사용자 질의가 모호함 → intent 불확실성
- 숫자/연도/코드 같은 정밀 정보 검색 실패
- 의미는 유사하지만 키워드 불일치 (lexical vs semantic)
- chunk 경계 문제(중요 문맥이 잘려 나감)
- 긴 문서에서 핵심 정보 누락 → Top-K에 정답 문서 미포함

표: 문제 유형 vs 기존 방식 한계
- 키워드 기반 검색 → 의미 유사성 약함
- Vector Search → exact keyword 약함
- 긴 문서 → chunk 분리 문제
- 애매한 질의 → retrieval drift

---

# 2) Retrieval Pipeline 
파이프라인 (권장 플로우):
User Query → Query Normalization → Hybrid Retrieval (BM25 + Dense) → Metadata Filtering → Reranker → Context Compression → LLM

각 단계 역할:
- Query Normalization: 숫자/단위 정규화, 키워드 표준화
- Hybrid Retrieval: lexical + semantic 후보 회수 (recall 확보)
- Metadata Filtering: 페이지 범위/타입 필터링으로 precision 보강
- Reranker: cross-encoder로 정밀 정렬
- Context Compression: LLM 토큰 예산 내로 필수 근거만 압축

---

# 3) 전처리 표현 전략 (chunking · embedding · metadata) 
- Chunking: recursive / semantic / parent-child 옵션. 크기/overlap의 trade-off 설명.
- Embedding: 고정된 임베딩 모델 사용으로 실험 통제.
- Metadata: 페이지, 표/그림 태그, 표 좌표 등 보존 권장.
- Indexing: BM25 인덱스와 vector index 병행.

---

# 4) PDF → Document AI 
문제: PyPDFLoader는 텍스트 섞임, 표·그래프 정보 손실.
도구 추천(계층화):
- 1단계: PyPDFLoader (빠름, 기본)
- 2단계: pdfplumber (표 추출 강화)
- 3단계: LayoutParser + PaddleOCR (레이아웃 + 그래프 숫자 OCR)
- 4단계: Docling / Multimodal RAG (표·캡션·그래프를 구조화)


---

# 5) Hybrid Search 원리
아이디어: BM25(lexical) + Dense(semantic) 결합 → recall과 lexical 정확성 동시 확보.
결합 방식:
- Linear score sum: 쉽지만 스케일 문제 존재
- Reciprocal Rank Fusion (RRF): rank 기반이라 스케일 민감도 낮음 (권장)
LangChain 예시:
```
from langchain.retrievers import EnsembleRetriever  
from langchain_community.retrievers import BM25Retriever

bm25 = BM25Retriever.from_documents(chunks); bm25.k = 5  
dense = vectorstore.as_retriever(search_kwargs={"k": 5})

ensemble = EnsembleRetriever(  
    retrievers=[bm25, dense],  
    weights=[0.5, 0.5],  
)
```

---

# 6) Re-ranking 원리 · 위치 · 모델 선택
2-stage 구조:
1) Fast recall (BM25 / Dense) → Top-K (예: 20)  
2) Precise rerank (Cross-Encoder) → Top-N (예: 5)

Bi-Encoder vs Cross-Encoder:
- Bi-Encoder: 속도・확장성 우수
- Cross-Encoder: 정밀도 우수, latency·비용 큼

ContextualCompression + CrossEncoder 예:

```
from langchain_classic.retrievers import ContextualCompressionRetriever  
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker  
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")  
compressor = CrossEncoderReranker(model=model, top_n=5)  
retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble)
```

---

# 7) Ablation 실험 설계 원칙 
통제: chunking·embedding 고정 → retrieval만 변경.  
구성: Dense / BM25 / Hybrid / Hybrid+Rerank.  
평가: RAGAS (context_precision, faithfulness, answer_relevancy) + latency + 정성적 top-chunk 검토.  
로그: contexts, 답변, latency, metadata, 단위 감지 결과.  

---

# 8) 5주차 결과 해석 가이드 
- context_precision ↑ → retrieval 개선이 핵심 병목 해소 신호
- faithfulness 소폭 ↑ → LLM prompt/format 개선 여지
- Hybrid 성능 하락 시 → 가중치(RRF) 혹은 데이터 특성(도메인 lexical 비중) 점검
- Re-rank 후 latency 증가 → production 대비 모델/파라미터 튜닝 필요
- 단위 불일치(십억 vs 조) 발견 시: unit-detection → 정규화 모듈 권장

스피커 노트:
- 결과가 기대와 다르면 '무엇이 바뀌었나'를 단계별로 점검하라(쿼리 유형, context 단위, 페이지 분포).

---

# 9) 데모 & 체크리스트
데모 제안:
- 동일 질의로 Dense / Hybrid / Hybrid+Rerank top-3 청크와 LLM 답변 비교.
터미널(노트북) 명령:
- jupyter notebook presentation.md  # 또는 reveal.js 변환

중요 산출물:
- week5_ablation_structured.csv (질문별 단위·latency 확인)
- docs/week5_retrospective.md, docs/adr/week5_retrieval_strategy.md

---

# 10) Next Step
우선 순위:
1. 단위/숫자 정규화 파이프라인 추가  
2. PDF → Document AI(표·그래프 보존) 도입(ppl: pdfplumber → LayoutParser+PaddleOCR → Docling)  
3. RRF 가중치 스윕 + re-ranker 경량 모델 실험  
4. error case 중심(이미지·표·숫자) 개선 로드맵 작성
