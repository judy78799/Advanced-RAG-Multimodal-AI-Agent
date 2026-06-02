# ADR: 5주차 Retrieval 전략

날짜: 2026-05-26
상태: 결정됨

## 결정
최종 채택 전략: Hybrid + Cross-Encoder Re-ranker
(BM25 : Dense = 0.5 : 0.5, Re-ranker 모델 권장: BAAI/bge-reranker-v2-m3 또는 한국어 특화 재학습 모델, top_k=5)

## 근거
- 정량적: 노트북에서 RAGAS가 사용 가능한 경우 `context_precision`을 우선 비교하여 최종 선택을 확정했습니다. (실행 결과가 있으면 `docs/week5_retrospective.md` 표에 자동 반영됩니다.)
- 정성적: BM25가 재무 리포트의 키워드·숫자·단위 정확도를 보완하고, Dense(임베딩)가 의미적 유사성을 포착합니다. Cross-Encoder 재정렬은 Hybrid에서 가져온 top-k 후보군의 관련도 순서를 더 정확히 만들어 LLM 입력 품질을 크게 향상시켰습니다.
- 실험 관찰: Hybrid만으로 일부 노이즈(문맥 비관련 청크)가 남았으나, Re-ranker 적용 시 context precision과 faithfulness가 전반적으로 개선됨.

## 트레이드오프
- Latency: Cross-Encoder 재정렬은 CPU/GPU 추론 비용과 응답시간을 증가시킵니다(평균 latency 증가). 프로덕션에서는 재랭커 캐싱, 비동기화, 경량 모델(예: ms-marco-MiniLM 계열) 도입 고려.
- 비용/자원: Re-ranker 모델 호스팅(메모리/추론비용) 및 BM25 인덱스 유지 비용 증가.
- 구현복잡도: EnsembleRetriever + ContextualCompression 파이프라인 관리가 필요해 엔지니어링 부담 상승.

## 대안(검토했으나 채택하지 않음)
- Dense only
  - 장점: 구현 단순, 표현 변형에 강함
  - 단점: 키워드·숫자·단위 정확성 부족(재무 리포트 도메인에 치명적)
- BM25 only
  - 장점: 키워드 정확도 높음
  - 단점: 의미적 유사성 포착 불가, 표현 변형에 취약
- Hybrid without Re-rank
  - 장점: Re-ranker 비용 없음
  - 단점: top-k에서 저품질 청크가 섞일 가능성 높아 LLM 입력 신뢰도 저하

## 면접용 짧은 답변 (Why this?)
정확도를 최우선으로 판단했습니다. 재무 리포트 같은 도메인에서는 숫자·단위·고유명사의 정확성이 중요해 BM25의 키워드 매칭을 보존하면서 의미적 유사성을 Dense가 보완하고, 최종적으로 Cross-Encoder가 후보군을 정렬해 LLM에게 더 신뢰할 수 있는 context를 제공하기 때문에 Hybrid+Rerank를 선택했습니다. 비용과 latency는 늘어나지만 도메인 특성상 정확도가 우선입니다.

## 운영 권고 (빠른 체크리스트)
- 프로덕션: Re-ranker는 GPU 인스턴스에서 호스팅하거나 경량화/양자화 모델 사용
- 캐시: 동일 쿼리/동일 문서 조합에 대해 context 결과 캐싱
- 모니터링: context_precision/faithfulness 지표를 주기적으로 수집해 재학습·튜닝 계획 수립
- 6주차: Agentic RAG 도입으로 검색 품질 판단 → 쿼리 재작성 → 전용 서브루틴(표/이미지/TablesQA) 분기 구현

---
파일: docs/adr/week5_retrieval_strategy.md
