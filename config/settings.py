"""
Team-H 프로젝트 통합 설정 관리

Pydantic BaseSettings를 사용하여 타입 안전성과 자동 검증을 제공합니다.

사용 예:
    from config.settings import settings, db_config, qdrant_config

    # 타입 안전한 접근
    db_url = db_config.database_url
    port = db_config.postgres_port  # int 타입 보장
"""

from typing import Literal, Optional, Dict
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """PostgreSQL 데이터베이스 설정"""

    postgres_user: str = Field(default="postgres", description="PostgreSQL 사용자명")
    postgres_password: str = Field(..., description="PostgreSQL 비밀번호 (필수)")
    postgres_db: str = Field(default="manager_g", description="데이터베이스 이름")
    postgres_host: str = Field(default="localhost", description="PostgreSQL 호스트")
    postgres_port: int = Field(default=5432, description="PostgreSQL 포트")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    @property
    def database_url(self) -> str:
        """데이터베이스 연결 URL 생성"""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


class QdrantConfig(BaseSettings):
    """Qdrant 벡터 데이터베이스 설정"""

    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant 서버 URL"
    )
    qdrant_password: str = Field(..., description="Qdrant 비밀번호 (필수)")
    manager_m_collection: str = Field(
        default="manager_m_memories",
        description="Manager M 컬렉션 이름"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


class EmbeddingConfig(BaseSettings):
    """임베딩 설정"""

    embedding_type: Literal["openai", "fastapi"] = Field(
        default="openai",
        description="임베딩 타입: 'openai' 또는 'fastapi'"
    )
    embedder_url: str = Field(
        default="http://localhost:8000",
        description="FastAPI 임베딩 서버 URL"
    )
    fastapi_embedding_dims: int = Field(
        default=1024,
        description="FastAPI 임베딩 차원"
    )
    openai_embedding_dims: int = Field(
        default=3072,
        description="OpenAI 임베딩 차원 (text-embedding-3-large)"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    @property
    def embedding_dims(self) -> int:
        """현재 선택된 임베딩 타입의 차원 반환"""
        return (
            self.openai_embedding_dims
            if self.embedding_type == "openai"
            else self.fastapi_embedding_dims
        )


class SmartThingsConfig(BaseSettings):
    """SmartThings IoT 설정 (Manager I용)"""

    smartthings_token: str = Field(..., description="SmartThings API 토큰 (필수)")
    smartthings_living_room_light: Optional[str] = Field(
        default=None,
        description="거실 조명 장치 ID"
    )
    smartthings_bedroom_light: Optional[str] = Field(
        default=None,
        description="안방 조명 장치 ID"
    )
    smartthings_bathroom_light: Optional[str] = Field(
        default=None,
        description="화장실 조명 장치 ID"
    )
    smartthings_living_room_speaker_outlet: Optional[str] = Field(
        default=None,
        description="거실 스피커 콘센트 장치 ID"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    def get_device_config(self) -> Dict[str, str]:
        """테스트에서 사용할 장치 설정 딕셔너리 반환"""
        config = {}
        if self.smartthings_living_room_light:
            config["living_room_light"] = self.smartthings_living_room_light
        if self.smartthings_bedroom_light:
            config["bedroom_light"] = self.smartthings_bedroom_light
        if self.smartthings_bathroom_light:
            config["bathroom_light"] = self.smartthings_bathroom_light
        if self.smartthings_living_room_speaker_outlet:
            config["living_room_speaker_outlet"] = self.smartthings_living_room_speaker_outlet
        return config


class GoogleCalendarConfig(BaseSettings):
    """Google Calendar API 설정 (Manager T용)"""

    google_calendar_credentials_path: Path = Field(
        default=Path("/root/credential/credentials.json"),
        description="Google OAuth 2.0 credentials.json 경로"
    )
    google_calendar_token_path: Path = Field(
        default=Path("/root/team-h/.credentials/calendar_token.json"),
        description="Google Calendar API 토큰 저장 경로"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    @field_validator("google_calendar_credentials_path", "google_calendar_token_path", mode="before")
    @classmethod
    def convert_to_path(cls, v):
        """문자열을 Path 객체로 변환"""
        if isinstance(v, str):
            return Path(v)
        return v


class APIConfig(BaseSettings):
    """외부 API 키 설정"""

    openai_api_key: str = Field(..., description="OpenAI API Key (필수)")
    tavily_api_key: str = Field(..., description="Tavily Search API Key (필수)")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


class LangfuseConfig(BaseSettings):
    """Langfuse Observability 설정"""

    langfuse_secret_key: str = Field(..., description="Langfuse Secret Key (필수)")
    langfuse_public_key: str = Field(..., description="Langfuse Public Key (필수)")
    langfuse_base_url: str = Field(
        default="http://192.168.0.151:3000",
        description="Langfuse 서버 URL (self-hosted)"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


class Settings:
    """
    통합 설정 클래스

    모든 설정 카테고리를 하나로 묶어 제공합니다.
    """

    def __init__(self):
        """설정 초기화 - 앱 시작 시 모든 필수값 검증"""
        self.api = APIConfig()
        self.database = DatabaseConfig()
        self.qdrant = QdrantConfig()
        self.embedding = EmbeddingConfig()
        self.smartthings = SmartThingsConfig()
        self.google_calendar = GoogleCalendarConfig()
        self.langfuse = LangfuseConfig()

    def validate_all(self) -> bool:
        """모든 설정 검증"""
        try:
            # 각 설정 객체가 생성되면서 자동으로 검증됨
            return True
        except Exception as e:
            print(f"❌ 설정 검증 실패: {e}")
            return False

    def print_config(self):
        """설정 정보 출력 (민감 정보 마스킹)"""
        print("=" * 60)
        print("Team-H 설정 정보")
        print("=" * 60)

        print("\n[API Keys]")
        print(f"  OpenAI API Key: {'*' * 20}...{self.api.openai_api_key[-4:]}")
        print(f"  Tavily API Key: {'*' * 20}...{self.api.tavily_api_key[-4:]}")

        print("\n[Database]")
        print(f"  Host: {self.database.postgres_host}:{self.database.postgres_port}")
        print(f"  Database: {self.database.postgres_db}")
        print(f"  User: {self.database.postgres_user}")

        print("\n[Qdrant]")
        print(f"  URL: {self.qdrant.qdrant_url}")
        print(f"  Collection: {self.qdrant.manager_m_collection}")

        print("\n[Embedding]")
        print(f"  Type: {self.embedding.embedding_type}")
        print(f"  Dimensions: {self.embedding.embedding_dims}")

        print("\n[SmartThings]")
        print(f"  Token: {'*' * 20}...{self.smartthings.smartthings_token[-4:]}")

        print("\n[Google Calendar]")
        print(f"  Credentials: {self.google_calendar.google_calendar_credentials_path}")
        print(f"  Token Path: {self.google_calendar.google_calendar_token_path}")

        print("\n[Langfuse]")
        print(f"  Public Key: {'*' * 20}...{self.langfuse.langfuse_public_key[-4:]}")
        print(f"  Secret Key: {'*' * 20}...{self.langfuse.langfuse_secret_key[-4:]}")
        print(f"  Base URL: {self.langfuse.langfuse_base_url}")

        print("=" * 60)


# 전역 설정 인스턴스 (앱 시작 시 한 번만 생성)
# 필수 환경변수가 없으면 여기서 즉시 에러 발생!
settings = Settings()

# 개별 설정 접근을 위한 편의 변수
api_config = settings.api
db_config = settings.database
qdrant_config = settings.qdrant
embedding_config = settings.embedding
smartthings_config = settings.smartthings
google_calendar_config = settings.google_calendar
langfuse_config = settings.langfuse


if __name__ == "__main__":
    # 설정 테스트
    if settings.validate_all():
        settings.print_config()
        print("\n✅ 모든 설정이 올바르게 로드되었습니다!")
    else:
        print("\n❌ 설정 로드 실패!")
