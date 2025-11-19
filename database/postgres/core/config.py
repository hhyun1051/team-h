"""
데이터베이스 설정 관리

Pydantic BaseSettings 기반 설정을 사용합니다.
타입 안전성과 자동 검증을 제공합니다.

마이그레이션 가이드:
    기존: POSTGRES_PORT (str)
    변경: db_config.postgres_port (int)

    기존: DATABASE_URL
    변경: db_config.database_url
"""
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Pydantic 기반 설정 import
from config import db_config

# 하위 호환성을 위한 변수 (기존 코드가 계속 작동하도록)
POSTGRES_USER = db_config.postgres_user
POSTGRES_PASSWORD = db_config.postgres_password
POSTGRES_DB = db_config.postgres_db
POSTGRES_HOST = db_config.postgres_host
POSTGRES_PORT = db_config.postgres_port  # ⚠️ 이제 int 타입!
DATABASE_URL = db_config.database_url

# 새로운 코드에서는 db_config를 직접 사용하세요
__all__ = [
    "db_config",  # ✅ 권장
    # 하위 호환성 (deprecated)
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DB",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "DATABASE_URL",
]
