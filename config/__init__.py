"""
Team-H Config Package

Pydantic 기반 통합 설정 관리

사용 예:
    from config import settings, db_config, qdrant_config

    # 타입 안전한 접근
    connection_url = db_config.database_url
    port = db_config.postgres_port  # int 타입 보장
"""

from config.settings import (
    Settings,
    DatabaseConfig,
    QdrantConfig,
    EmbeddingConfig,
    SmartThingsConfig,
    GoogleCalendarConfig,
    APIConfig,
    settings,
    api_config,
    db_config,
    qdrant_config,
    embedding_config,
    smartthings_config,
    google_calendar_config,
)

__all__ = [
    # 설정 클래스
    "Settings",
    "DatabaseConfig",
    "QdrantConfig",
    "EmbeddingConfig",
    "SmartThingsConfig",
    "GoogleCalendarConfig",
    "APIConfig",
    # 전역 인스턴스
    "settings",
    "api_config",
    "db_config",
    "qdrant_config",
    "embedding_config",
    "smartthings_config",
    "google_calendar_config",
]

__version__ = "0.1.0"
