"""
Streamlit 공통 설정 및 상수

모든 Streamlit 앱에서 공유하는 설정값과 상수:
- 페이지 설정
- 아바타 이모지
- 기본값
- 가이드 텍스트

Note: 실제 환경 변수 설정은 /root/team-h/config/settings.py를 참조합니다.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# 중앙 설정 import
try:
    from config.settings import (
        api_config,
        smartthings_config,
        google_calendar_config,
        embedding_config,
        qdrant_config,
    )
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False
    print("⚠️ Warning: config.settings를 import할 수 없습니다. 환경변수에서 직접 로드합니다.")
    import os

# ============================================================================
# 페이지 설정
# ============================================================================

PAGE_CONFIGS = {
    "team_h": {
        "page_title": "Team-H",
        "page_icon": "💫",
        "layout": "wide",
        "title": "💫 Team-H",
        "caption": "Team-H",
    },
}


# ============================================================================
# 아바타 이모지
# ============================================================================

AGENT_AVATARS = {
    "team_h": "🤖",
    "manager_s": "🔍",
    "manager_m": "🧠",
    "manager_i": "🏠",
    "manager_t": "📅",
    "assistant": "🤖",
    "user": "👤",
}


# ============================================================================
# 기본값
# ============================================================================

DEFAULT_VALUES = {
    "user_id": "hhyun",
    "thread_id_suffix": {
        "team_h": "streamlit_teamh_thread",
        "manager_s": "streamlit_search_thread",
        "manager_m": "streamlit_memory_thread",
        "manager_i": "streamlit_iot_thread",
        "manager_t": "streamlit_calendar_thread",
    },
    "model_name": "gpt-4o-mini",
    "temperature": 0.7,
    "max_search_results": 5,
}


# ============================================================================
# 환경변수 기본값
# ============================================================================

def get_env_defaults() -> Dict[str, Any]:
    """
    환경변수에서 기본값 로드

    중앙 설정(config.settings)을 우선 사용하고,
    불가능한 경우 환경변수에서 직접 로드합니다.
    """
    if SETTINGS_AVAILABLE:
        # config.settings에서 로드
        return {
            "smartthings_token": smartthings_config.smartthings_token,
            "tavily_api_key": api_config.tavily_api_key,
            "device_config": smartthings_config.get_device_config(),
            "google_credentials_path": str(google_calendar_config.google_calendar_credentials_path),
            "google_token_path": str(google_calendar_config.google_calendar_token_path),
            # Manager M (Qdrant + Embedding) 설정 추가
            "embedding_type": embedding_config.embedding_type,
            "embedder_url": embedding_config.embedder_url,
            "openai_api_key": api_config.openai_api_key,
            "embedding_dims": embedding_config.embedding_dims,
            "qdrant_url": qdrant_config.qdrant_url,
            "qdrant_api_key": qdrant_config.qdrant_password,  # qdrant_password → qdrant_api_key
            "m_collection_name": qdrant_config.manager_m_collection,
        }
    else:
        # 환경변수에서 직접 로드 (폴백)
        import os
        return {
            "smartthings_token": os.getenv("SMARTTHINGS_TOKEN", ""),
            "tavily_api_key": os.getenv("TAVILY_API_KEY", ""),
            "device_config": {
                "living_room_speaker_outlet": os.getenv("SPEAKER_ID", ""),
                "living_room_light": os.getenv("PROJECTOR_ID", ""),
                "bedroom_light": os.getenv("VERTICAL_MONITOR_ID", ""),
                "bathroom_light": os.getenv("AIR_PURIFIER_ID", ""),
            },
            "google_credentials_path": os.getenv("GOOGLE_CALENDAR_CREDENTIALS_PATH"),
            "google_token_path": os.getenv("GOOGLE_CALENDAR_TOKEN_PATH"),
            # Manager M (Qdrant + Embedding) 설정 추가
            "embedding_type": os.getenv("EMBEDDING_TYPE", "openai"),
            "embedder_url": os.getenv("EMBEDDER_URL", "http://localhost:8000"),
            "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
            "embedding_dims": int(os.getenv("OPENAI_EMBEDDING_DIMS", "3072")),
            "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
            "qdrant_api_key": os.getenv("QDRANT_PASSWORD", ""),
            "m_collection_name": os.getenv("MANAGER_M_COLLECTION", "manager_m_memories"),
        }


# ============================================================================
# 사용 가이드 텍스트
# ============================================================================

USAGE_GUIDES = {
    "manager_s": """
**Manager S란?**
- 웹 검색 에이전트
- Tavily Search API를 사용한 실시간 정보 검색
- 뉴스, 일반 웹 검색 지원

**사용 방법:**
1. .env 파일에 TAVILY_API_KEY 설정
2. '에이전트 초기화' 버튼 클릭
3. 아래 채팅창에서 Manager S와 대화

**검색 기능:**
- 일반 웹 검색: "파이썬 최신 버전은?"
- 뉴스 검색: "오늘 AI 관련 뉴스 찾아줘"
- 실시간 정보: "현재 환율은?"

**예시 명령:**
- "LangChain 최신 소식 검색해줘"
- "2024년 AI 트렌드 찾아줘"
- "파이썬 3.12 새로운 기능은?"
- "최근 OpenAI 뉴스 검색해줘"

**특징:**
- 실시간 웹 정보 접근
- 검색 결과 요약 및 정리
- 출처 URL 제공
""",
    "manager_m": """
**Manager M이란?**
- 일반 기억 관리 에이전트
- 사용자 선호도, 습관, 대화 컨텍스트 등을 기억

**사용 방법:**
1. 먼저 '에이전트 초기화' 버튼 클릭
2. 사용자 ID 설정 (선택사항)
3. 아래 채팅창에서 Manager M과 대화

**메모리 작업:**
- 기억 검색: "내 선호도 찾아줘"
- 기억 추가: "나는 커피를 좋아해"
- 기억 업데이트: "ID xxx의 기억을 수정해줘"
- 기억 삭제: "ID xxx 기억 삭제해줘"

**Human-in-the-Loop:**
- 메모리 추가/수정/삭제 시 자동으로 승인 요청
- yes, no, edit 중 선택 가능
""",
    "manager_i": """
**Manager I란?**
- IoT 제어 에이전트
- SmartThings를 통해 스마트 기기 제어
- 거실/안방/화장실 불, 스피커, 미니PC 제어

**사용 방법:**
1. SmartThings Token 입력
2. '에이전트 초기화' 버튼 클릭
3. 아래 채팅창에서 Manager I와 대화

**제어 가능한 장치:**
- 거실 불 (프로젝터)
- 안방 불 (세로모니터 콘센트)
- 화장실 불 (공기청정기)
- 거실 스피커 (스마트 콘센트)
- 미니PC (종료만 가능)

**예시 명령:**
- "거실 불 켜줘"
- "안방 불 꺼줘"
- "거실 스피커 꺼줘"
- "미니PC 종료해줘" (승인 필요)

**Human-in-the-Loop:**
- 위험한 작업(미니PC 종료)만 승인 요청
- 일반 불 제어는 즉시 실행
""",
    "manager_t": """
**Manager T란?**
- 캘린더 및 시간 관리 에이전트
- Google Calendar 연동으로 일정 관리
- 자연어 시간 파싱 지원

**사용 방법:**
1. Google Calendar API 설정 (credentials.json)
2. '에이전트 초기화' 버튼 클릭
3. 아래 채팅창에서 Manager T와 대화

**일정 관리:**
- 일정 생성: "내일 오후 3시에 회의 잡아줘"
- 일정 조회: "오늘 일정 보여줘"
- 일정 수정: "내일 회의 시간 4시로 변경해줘"
- 일정 삭제: "내일 회의 취소해줘"

**예시 명령:**
- "다음주 월요일 오전 10시에 팀 회의 일정 추가해줘"
- "이번 주 일정 알려줘"
- "내일 일정 있어?"
- "금요일 저녁 7시 저녁 약속 추가"

**특징:**
- 자연어 시간 이해 ("내일", "다음주 월요일" 등)
- Google Calendar 실시간 동기화
- 일정 충돌 확인
- 스마트 알림 설정

**Human-in-the-Loop:**
- 일정 생성/수정/삭제 시 승인 요청
- yes, no, edit 중 선택 가능
""",
}


# ============================================================================
# 사이드바 정보 텍스트 생성 함수
# ============================================================================

def format_sidebar_info(
    thread_id: str,
    message_count: int,
    additional_info: Dict[str, str] = None
) -> str:
    """
    사이드바 정보 텍스트 포맷팅

    Args:
        thread_id: 스레드 ID
        message_count: 메시지 개수
        additional_info: 추가 정보 딕셔너리

    Returns:
        포맷팅된 정보 문자열
    """
    info = {
        "Thread ID": f"`{thread_id}`",
        "메시지 수": str(message_count),
    }

    if additional_info:
        info.update(additional_info)

    return info
