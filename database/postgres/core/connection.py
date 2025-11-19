"""
데이터베이스 연결 및 세션 관리
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.postgres.core.config import DATABASE_URL
from database.postgres.models import Base

# 엔진 생성
engine = create_engine(
    DATABASE_URL,
    echo=False,  # 실제 쿼리 로깅이 필요할 경우 True로 변경
    pool_pre_ping=True,  # 연결 유효성 검사
)

# 세션 팩토리 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """데이터베이스 세션 생성 (의존성 주입용)"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()