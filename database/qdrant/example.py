"""
VectorStore 예제 - FastAPI/OpenAI 임베딩 지원

사용 전 준비:
1. .env 파일에서 EMBEDDING_TYPE 설정 (fastapi 또는 openai)
2. FastAPI 임베딩 서버 실행 (EMBEDDING_TYPE=fastapi인 경우)
3. Qdrant 서버 실행 확인
4. 이 스크립트 실행
"""

from vector_store_teamh import VectorStore
from langchain_core.documents import Document
import os
import dotenv
dotenv.load_dotenv()

# ========================================
# 예제 1: 기본 사용법 (.env 설정 사용)
# ========================================
def example_basic():
    print("\n" + "="*60)
    print("예제 1: 기본 사용법 (.env 파일의 EMBEDDING_TYPE 사용)")
    print("="*60)

    # VectorStore 초기화 (.env의 EMBEDDING_TYPE 사용)
    vs = VectorStore(
        url="http://localhost:6333",
        api_key=os.getenv('QDRANT_PASSWORD'),
        collection_name="team_h_example",
        recreate_collection=True,
        # embedding_type은 .env 파일에서 자동 로드됨
    )
    
    # 문서 생성
    docs = [
        Document(page_content="FastAPI는 빠르고 현대적인 웹 프레임워크입니다.",
                 metadata={"category": "tech", "lang": "ko"}),
        Document(page_content="Python은 데이터 과학에 널리 사용됩니다.",
                 metadata={"category": "tech", "lang": "ko"}),
        Document(page_content="Docker는 컨테이너 기술을 사용합니다.",
                 metadata={"category": "devops", "lang": "ko"}),
    ]

    print(f"\n총 {len(docs)}개 문서 추가 중...")
    vs.add_documents(docs)

    # 검색
    print("\n검색 수행...")
    results = vs.search("웹 프레임워크", k=2)

    print(f"\n검색 결과: {len(results)}개")
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] {doc.page_content}")
        print(f"    메타데이터: {doc.metadata}")


# ========================================
# 예제 2: OpenAI 임베딩 명시적 사용
# ========================================
def example_openai():
    print("\n" + "="*60)
    print("예제 2: OpenAI 임베딩 명시적 사용")
    print("="*60)

    # VectorStore 초기화 (OpenAI 명시)
    vs = VectorStore(
        url="http://localhost:6333",
        api_key=os.getenv('QDRANT_PASSWORD'),
        collection_name="team_h_openai",
        recreate_collection=True,
        embedding_type="openai",  # OpenAI 명시
        # openai_api_key는 .env에서 자동 로드됨
        # dense_size=3072 (자동 설정됨)
    )

    # 문서 생성
    docs = [
        Document(page_content="OpenAI의 text-embedding-3-large는 강력한 임베딩 모델입니다.",
                 metadata={"source": "openai"}),
        Document(page_content="벡터 검색은 의미론적 유사도를 기반으로 합니다.",
                 metadata={"source": "general"}),
    ]

    print(f"\n{len(docs)}개 문서 추가 중...")
    vs.add_documents(docs)

    # 검색
    results = vs.search_with_score("임베딩 모델", k=2)

    print(f"\n검색 결과 (점수 포함):")
    for doc, score in results:
        print(f"  - Score: {score:.4f} | {doc.page_content}")


# ========================================
# 예제 3: FastAPI 임베딩 명시적 사용
# ========================================
def example_fastapi():
    print("\n" + "="*60)
    print("예제 3: FastAPI 로컬 임베딩 명시적 사용")
    print("="*60)

    # VectorStore 초기화 (FastAPI 명시)
    vs = VectorStore(
        url="http://localhost:6333",
        api_key=os.getenv('QDRANT_PASSWORD'),
        collection_name="team_h_fastapi",
        recreate_collection=True,
        embedding_type="fastapi",  # FastAPI 명시
        embedder_url="http://192.168.0.101:8000",
        # dense_size=1024 (자동 설정됨)
    )

    # 문서 생성
    docs = [
        Document(page_content="로컬 임베딩 서버는 비용 효율적입니다.",
                 metadata={"type": "local"}),
        Document(page_content="BAAI/bge-m3 모델은 다국어를 지원합니다.",
                 metadata={"type": "model"}),
    ]

    print(f"\n{len(docs)}개 문서 추가 중...")
    vs.add_documents(docs)

    # 검색
    results = vs.search("로컬 모델", k=2)

    print(f"\n검색 결과:")
    for i, doc in enumerate(results, 1):
        print(f"  [{i}] {doc.page_content}")


# ========================================
# 메인 실행
# ========================================
if __name__ == "__main__":
    print("\n🚀 VectorStore 임베딩 예제")
    print(f"현재 .env 설정: EMBEDDING_TYPE={os.getenv('EMBEDDING_TYPE', 'fastapi')}")

    # 실행할 예제 선택 (주석 해제하여 실행)
    example_basic()
    # example_openai()
    # example_fastapi()

    print("\n" + "="*60)
    print("✅ 예제 실행 완료!")
    print("="*60)