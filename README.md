# KB 부동산 보고서 챗봇

"2024 KB 부동산 보고서" PDF를 기반으로 질문에 답하는 RAG(검색 증강 생성) 챗봇입니다. Streamlit UI에서 대화형으로 질문하면, 보고서 내용을 검색해 근거로 삼아 답변합니다.

## 동작 방식

1. `data/2024_KB_부동산_보고서_최종.pdf`를 로드해 문서를 청크 단위로 분할
2. OpenAI 임베딩으로 벡터화한 뒤 FAISS 벡터스토어에 저장(`faiss_db/`, 최초 실행 시 생성 후 재사용)
3. 질문이 들어오면 관련 청크를 검색해 컨텍스트로 삼아 `gpt-4o-mini`가 답변 생성
4. 대화 이력을 반영해 후속 질문에도 맥락을 유지

## 기술 스택

Streamlit · LangChain · FAISS · OpenAI(gpt-4o-mini, text-embedding)

## 실행

```bash
pip install -r requirements.txt
```

OpenAI API 키가 필요합니다. `data/.env`에 아래처럼 설정하거나(로컬 실행) Streamlit Cloud의 secrets에 등록하세요.

```
OPENAI_API_KEY=sk-...
```

```bash
streamlit run app.py
```

`data/` 폴더에 다른 PDF 보고서로 교체하면 해당 문서 기준으로 질의응답하도록 재사용할 수 있습니다.
