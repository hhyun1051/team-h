"""
데이터베이스 및 애플리케이션 설정 관리
.env 파일로부터 환경 변수를 로드합니다.
"""
import os
from dotenv import load_dotenv

# .env 파일이 존재할 경우, 해당 파일의 환경 변수를 로드합니다.
load_dotenv()

# 데이터베이스 연결 정보
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
POSTGRES_DB = os.getenv("POSTGRES_DB", "manager_g")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

# 데이터베이스 연결 URL
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
