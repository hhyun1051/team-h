"""
Manager M Memory 설정 관리

Pydantic BaseSettings 기반 설정을 사용합니다.
타입 안전성과 자동 검증을 제공합니다.

마이그레이션 가이드:
    기존: MemoryConfig.QDRANT_URL
    변경: qdrant_config.qdrant_url

    기존: MemoryConfig.EMBEDDING_TYPE
    변경: embedding_config.embedding_type (Literal 타입으로 자동 검증)

    기존: MemoryConfig.FASTAPI_EMBEDDING_DIMS (str을 int로 변환)
    변경: embedding_config.fastapi_embedding_dims (자동으로 int)
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Pydantic 기반 설정 import
from config import qdrant_config, embedding_config, api_config


class MemoryConfig:
    """
    메모리 설정 클래스 (하위 호환성을 위해 유지)

    ⚠️ DEPRECATED: 새로운 코드에서는 qdrant_config와 embedding_config를 직접 사용하세요.

    예시:
        # 기존 방식 (deprecated)
        url = MemoryConfig.QDRANT_URL

        # 새로운 방식 (권장)
        url = qdrant_config.qdrant_url
    """

    # Qdrant 설정 (qdrant_config에서 가져옴)
    QDRANT_URL: str = qdrant_config.qdrant_url
    QDRANT_PASSWORD: str = qdrant_config.qdrant_password
    MANAGER_M_COLLECTION: str = qdrant_config.manager_m_collection

    # 임베딩 설정 (embedding_config에서 가져옴)
    EMBEDDING_TYPE: str = embedding_config.embedding_type
    EMBEDDER_URL: str = embedding_config.embedder_url
    FASTAPI_EMBEDDING_DIMS: int = embedding_config.fastapi_embedding_dims
    OPENAI_EMBEDDING_DIMS: int = embedding_config.openai_embedding_dims

    # OpenAI API Key (api_config에서 가져옴)
    OPENAI_API_KEY: str = api_config.openai_api_key

    @classmethod
    def validate(cls) -> bool:
        """
        필수 설정값 검증

        ⚠️ DEPRECATED: Pydantic이 자동으로 검증하므로 불필요

        Returns:
            항상 True (Pydantic이 이미 검증했음)
        """
        # Pydantic이 이미 필수값 검증을 완료했으므로
        # 여기까지 도달했다면 모든 설정이 유효함
        return True

    @classmethod
    def print_config(cls):
        """현재 설정 출력 (디버깅용)"""
        print("=== Manager M Memory Configuration ===")
        print(f"QDRANT_URL: {cls.QDRANT_URL}")
        print(f"QDRANT_PASSWORD: {'*' * len(cls.QDRANT_PASSWORD)}")
        print(f"MANAGER_M_COLLECTION: {cls.MANAGER_M_COLLECTION}")
        print(f"EMBEDDING_TYPE: {cls.EMBEDDING_TYPE}")
        if cls.EMBEDDING_TYPE == "fastapi":
            print(f"EMBEDDER_URL: {cls.EMBEDDER_URL}")
            print(f"FASTAPI_EMBEDDING_DIMS: {cls.FASTAPI_EMBEDDING_DIMS}")
        elif cls.EMBEDDING_TYPE == "openai":
            print(f"OPENAI_API_KEY: Set")
            print(f"OPENAI_EMBEDDING_DIMS: {cls.OPENAI_EMBEDDING_DIMS}")
        print("=" * 40)


# 새로운 코드에서 사용할 export
__all__ = [
    "qdrant_config",  # ✅ 권장
    "embedding_config",  # ✅ 권장
    "MemoryConfig",  # ⚠️ 하위 호환성 (deprecated)
]


# 모듈 로드 시 검증
if __name__ == "__main__":
    MemoryConfig.print_config()
    if MemoryConfig.validate():
        print("[✅] Configuration is valid")
    else:
        print("[❌] Configuration has issues")
