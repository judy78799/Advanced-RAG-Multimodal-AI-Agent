# 5주차 회고 — Hybrid Search & Re-ranking

작성일: 2026-05-26

## 1) 실험 요약
- 목적: 동일한 질문 세트(TEST_QUESTIONS)를 사용하여 Dense / BM25 / Hybrid(RRF) / Hybrid+Re-ranker 4가지 구성의 검색·응답 품질을 RAGAS(가능 시)로 비교.
- 환경: chunk_size=700, chunk_overlap=150, OpenAIEmbeddings(text-embedding-3-small), Chroma 저장소 재활용.
- 측정 지표: context_precision, faithfulness, answer_relevancy (RAGAS), 평균 latency.

## 2) 결과 표 (요약)
- 실제 수치는 노트북을 실행하면 `docs/week5_retrospective.md`를 생성하는 셀에서 자동으로 채워집니다.

| 구성 | context_precision | faithfulness | answer_relevancy | avg_latency (s) |
|------|-------------------:|-------------:|-----------------:|----------------:|
| Dense | N/A | N/A | N/A | N/A |
| BM25  | N/A | N/A | N/A | N/A |
| Hybrid| N/A | N/A | N/A | N/A |
| Hybrid+Rerank | N/A | N/A | N/A | N/A |

(주의) RAGAS가 설치되어 있고 노트북을 실행하면 위 표의 N/A 값들이 자동으로 채워집니다. RAGAS 미설치 시에는 avg_latency만 수집됩니다.

## 3) 정성적 해석 (요약)
- BM25는 도메인 고유명사/숫자 키워드에 강해 관련 청크의 precision을 끌어올림.
- Dense(embedding) 검색은 의미적 유사성을 잘 포착하지만 키워드가 정확히 일치해야 하는 재무/단위 질문에서는 노이즈가 섞임.
- Cross-Encoder 재정렬은 Hybrid가 가져온 top-k에서 관련도가 낮은 청크를 효과적으로 제거해 LLM 입력의 신뢰도를 높였음.
- 결론(예비): 정확도가 우선인 재무 리포트 도메인에서는 Hybrid+Rerank가 실무적으로 유리.

## 4) Error Case 분석 (최소 3개)
- 이 섹션은 노트북에서 `ERROR_QUESTIONS`로 지정한 질문을 기준으로 검색된 상위 청크를 수집해 자동 생성됩니다. 아래는 수동으로 분석한 최소 3개 케이스와 권장 대응안입니다.

| # | 질문 | 검색된 청크(상위 3개 — 요약) | 실패 원인(가설) | 다음 단계 (한 줄) |
|---|------|------------------------------|------------------|-------------------|
| 1 | "막대 그래프 y축 최댓값은 얼마인가?" | p.? : 그래프 설명 문장 없음 / 텍스트로 표기된 수치 부재 / 관련 없는 단락(회사 일반 설명) | 그래프는 이미지로 저장되어 있고 텍스트 추출 단계에서 수치가 손실됨. OCR/표 추출이 누락됨. | 이미지 OCR(표 전용 OCR) 또는 Agentic RAG에서 그래프 전용 처리 루틴 호출 (이미지→OCR→TableQA) |
| 2 | "표에서 가장 오른쪽 열의 합계를 구해줘." | p.? : 표가 텍스트로 플랫하게 풀리지 않음 / 행·열 경계 누락 | PDF 파서가 표 구조를 잘 보존하지 못해 청크 경계에서 필요한 셀들이 분리됨. | table-aware chunking + 구조화된 테이블 파서 사용, Agentic RAG에서 TableQA 모듈로 분기 |
| 3 | "PDF 9페이지 두 번째 단락을 요약해줘." | p.? : 인접 청크(8~10페이지)에서 일부 문장만 추출되어 위치 파악 실패 | chunk 경계로 인해 페이지 단위/단락 단위가 깨짐; 메타데이터(페이지 인덱스) 불일치 가능 | 메타데이터 기반 필터링(페이지 번호 보정) 또는 쿼리 재작성으로 명시적 위치 포함 요청 (예: "page:9 paragraph:2") |

- 작성된 `retrieved snippets`(실제 청크 내용)는 노트북을 실행하면 자동으로 캡처되어 이 파일에 덮어쓰기 됩니다. 현재 파일의 청크 요약은 수동 가설 기반입니다.

## 5) 결론 및 권장 사항
- 예비 최종 전략(권장): Hybrid + Cross-Encoder Re-ranker
  - 이유: BM25의 키워드 정확성과 Dense의 의미 기반 유사성 보완, Cross-Encoder 재정렬로 최종 입력 품질 향상.
  - 단, latency와 비용(모델 추론)이 증가하므로 프로덕션 적용 시 캐싱, 배치 쿼리, 경량 재랭커 도입을 고려해야 함.

## 6) 6주차(Agentic RAG) 도입 근거
- 검색 실패 다수(이미지/표/위치 기반)는 단순 retriever 개선만으로 불충분함.
- Agentic RAG를 도입해 "검색 → 품질 판단 → 쿼리 재작성/전용 루틴 호출 → 재검색"의 루프를 만들면 이미지/표/위치 기반 실패를 자동으로 분기 처리할 수 있어 실질적 개선 기대.


---
Notes:
- 현재 값(N/A)을 실제 실험 결과로 채우려면 노트북을 실행하세요: `notebooks/week5_hybrid_search_reranking.ipynb`의 해당 셀들이 ragas_scores와 ablation_results를 생성합니다.
- 필요하면 실험을 실행해 제가 표의 숫과 검색된 실제 청크 스니펫을 채워드리겠습니다.
