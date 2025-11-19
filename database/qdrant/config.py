"""
Manager M Memory 설정 관리
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)


class MemoryConfig:
    """메모리 설정 클래스"""

    # Qdrant 설정
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_PASSWORD: str = os.getenv("QDRANT_PASSWORD", "")
    MANAGER_M_COLLECTION: str = os.getenv("MANAGER_M_COLLECTION", "manager_m_memories")

    # 임베딩 타입 설정 ("fastapi" 또는 "openai")
    EMBEDDING_TYPE: str = os.getenv("EMBEDDING_TYPE", "fastapi")

    # FastAPI 임베딩 서버 (EMBEDDING_TYPE="fastapi"일 때 사용)
    EMBEDDER_URL: str = os.getenv("EMBEDDER_URL", "http://192.168.0.101:8000")
    FASTAPI_EMBEDDING_DIMS: int = int(os.getenv("FASTAPI_EMBEDDING_DIMS", "1024"))  # BAAI/bge-m3

    # OpenAI 임베딩 (EMBEDDING_TYPE="openai"일 때 사용)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_EMBEDDING_DIMS: int = int(os.getenv("OPENAI_EMBEDDING_DIMS", "3072"))  # text-embedding-3-large

    @classmethod
    def validate(cls) -> bool:
        """
        필수 설정값 검증

        Returns:
            모든 필수 설정이 있으면 True
        """
        if not cls.QDRANT_PASSWORD:
            print("[⚠️] Warning: QDRANT_PASSWORD is not set in .env")
            return False

        # EMBEDDING_TYPE 검증
        if cls.EMBEDDING_TYPE not in ["fastapi", "openai"]:
            print(f"[⚠️] Warning: EMBEDDING_TYPE must be 'fastapi' or 'openai', got '{cls.EMBEDDING_TYPE}'")
            return False

        # FastAPI 임베딩 사용 시
        if cls.EMBEDDING_TYPE == "fastapi":
            if not cls.EMBEDDER_URL:
                print("[⚠️] Warning: EMBEDDER_URL is not set in .env")
                return False

        # OpenAI 임베딩 사용 시
        elif cls.EMBEDDING_TYPE == "openai":
            if not cls.OPENAI_API_KEY:
                print("[⚠️] Warning: OPENAI_API_KEY is not set in .env (required for EMBEDDING_TYPE='openai')")
                return False

        return True

    @classmethod
    def print_config(cls):
        """현재 설정 출력 (디버깅용)"""
        print("=== Manager M Memory Configuration ===")
        print(f"QDRANT_URL: {cls.QDRANT_URL}")
        print(f"QDRANT_PASSWORD: {'*' * len(cls.QDRANT_PASSWORD) if cls.QDRANT_PASSWORD else 'Not set'}")
        print(f"MANAGER_M_COLLECTION: {cls.MANAGER_M_COLLECTION}")
        print(f"EMBEDDING_TYPE: {cls.EMBEDDING_TYPE}")
        if cls.EMBEDDING_TYPE == "fastapi":
            print(f"EMBEDDER_URL: {cls.EMBEDDER_URL}")
            print(f"FASTAPI_EMBEDDING_DIMS: {cls.FASTAPI_EMBEDDING_DIMS}")
        elif cls.EMBEDDING_TYPE == "openai":
            print(f"OPENAI_API_KEY: {'Set' if cls.OPENAI_API_KEY else 'Not set'}")
            print(f"OPENAI_EMBEDDING_DIMS: {cls.OPENAI_EMBEDDING_DIMS}")
        print("=" * 40)


# 모듈 로드 시 검증
if __name__ == "__main__":
    MemoryConfig.print_config()
    if MemoryConfig.validate():
        print("[✅] Configuration is valid")
    else:
        print("[❌] Configuration has issues")
