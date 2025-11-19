"""
Streamlit Agent Factory - 에이전트 캐싱 및 생성

⚠️ DEPRECATED: 이 파일은 더 이상 사용되지 않습니다.

대신 streamlits/components.py의 create_cached_agent() 함수를 사용하세요.

마이그레이션 가이드:
-----------------
Before (agent_factory.py):
    from streamlits.agent_factory import create_cached_manager_s_agent
    agent = create_cached_manager_s_agent(model_name="gpt-4o-mini", ...)

After (components.py):
    from streamlits.components import create_cached_agent
    from agents import ManagerS
    agent = create_cached_agent(ManagerS, model_name="gpt-4o-mini", ...)

장점:
- 더 간단한 API (하나의 범용 함수)
- 모든 에이전트 타입 지원 (ManagerS, ManagerM, ManagerI, ManagerT, TeamHGraph)
- 중복 코드 제거
- 유지보수성 향상

이 파일은 참고용으로만 남겨두며, 향후 삭제될 예정입니다.
"""

import streamlit as st
from typing import Dict, Optional


@st.cache_resource
def create_cached_team_h_agent(**config):
    """
    Team-H 에이전트 캐싱 생성

    Args:
        **config: TeamHAgent 초기화 파라미터

    Returns:
        TeamHAgent 인스턴스
    """
    from agents.team_h import TeamHAgent

    print("[🔄] Creating cached Team-H agent...")
    agent = TeamHAgent(**config)
    print("[✅] Cached Team-H agent created successfully")
    return agent


@st.cache_resource
def create_cached_manager_s_agent(**config):
    """
    Manager S 에이전트 캐싱 생성

    Args:
        **config: ManagerS 초기화 파라미터

    Returns:
        ManagerS 인스턴스
    """
    from agents.manager_s import ManagerS

    print("[🔄] Creating cached Manager S agent...")
    agent = ManagerS(**config)
    print("[✅] Cached Manager S agent created successfully")
    return agent


@st.cache_resource
def create_cached_manager_m_agent(**config):
    """
    Manager M 에이전트 캐싱 생성

    Args:
        **config: ManagerM 초기화 파라미터

    Returns:
        ManagerM 인스턴스
    """
    from agents.manager_m import ManagerM

    print("[🔄] Creating cached Manager M agent...")
    agent = ManagerM(**config)
    print("[✅] Cached Manager M agent created successfully")
    return agent


@st.cache_resource
def create_cached_manager_i_agent(**config):
    """
    Manager I 에이전트 캐싱 생성

    Args:
        **config: ManagerI 초기화 파라미터

    Returns:
        ManagerI 인스턴스
    """
    from agents.manager_i import ManagerI

    print("[🔄] Creating cached Manager I agent...")
    agent = ManagerI(**config)
    print("[✅] Cached Manager I agent created successfully")
    return agent


def validate_team_h_config(config: Dict) -> tuple[bool, Optional[str]]:
    """
    Team-H 설정 검증

    Args:
        config: 설정 딕셔너리

    Returns:
        (is_valid, error_message) 튜플
    """
    # 최소 하나의 매니저가 활성화되어야 함
    enable_i = config.get("enable_manager_i", False)
    enable_m = config.get("enable_manager_m", False)
    enable_s = config.get("enable_manager_s", False)

    if not (enable_i or enable_m or enable_s):
        return False, "최소 하나의 매니저를 활성화해야 합니다."

    # Manager I가 활성화되어 있으면 SmartThings Token 필요
    if enable_i and not config.get("smartthings_token"):
        return False, "Manager I를 사용하려면 SmartThings Token이 필요합니다."

    # Manager S가 활성화되어 있으면 Tavily API Key 필요
    if enable_s and not config.get("tavily_api_key"):
        return False, "Manager S를 사용하려면 Tavily API Key가 필요합니다."

    return True, None


def validate_manager_s_config(config: Dict) -> tuple[bool, Optional[str]]:
    """
    Manager S 설정 검증

    Args:
        config: 설정 딕셔너리

    Returns:
        (is_valid, error_message) 튜플
    """
    if not config.get("tavily_api_key"):
        return False, "Tavily API Key가 필요합니다."

    return True, None


def validate_manager_m_config(config: Dict) -> tuple[bool, Optional[str]]:
    """
    Manager M 설정 검증

    Args:
        config: 설정 딕셔너리

    Returns:
        (is_valid, error_message) 튜플
    """
    # Manager M은 기본적으로 환경변수에서 설정을 로드하므로 별도 검증 불필요
    return True, None


def validate_manager_i_config(config: Dict) -> tuple[bool, Optional[str]]:
    """
    Manager I 설정 검증

    Args:
        config: 설정 딕셔너리

    Returns:
        (is_valid, error_message) 튜플
    """
    if not config.get("smartthings_token"):
        return False, "SmartThings Token이 필요합니다."

    # 장치 설정 확인
    device_config = config.get("device_config", {})
    required_devices = [
        "living_room_light",
        "bedroom_light",
        "bathroom_light",
        "living_room_speaker_outlet"
    ]

    missing_devices = [d for d in required_devices if d not in device_config]
    if missing_devices:
        return False, f"일부 장치가 설정되지 않았습니다: {', '.join(missing_devices)}"

    return True, None


def create_agent_with_validation(agent_type: str, config: Dict):
    """
    검증 후 에이전트 생성

    Args:
        agent_type: 에이전트 타입 ("team_h", "manager_s", "manager_m", "manager_i")
        config: 설정 딕셔너리

    Returns:
        에이전트 인스턴스 또는 None (검증 실패 시)
    """
    # 설정 검증
    if agent_type == "team_h":
        is_valid, error_msg = validate_team_h_config(config)
        create_func = create_cached_team_h_agent
    elif agent_type == "manager_s":
        is_valid, error_msg = validate_manager_s_config(config)
        create_func = create_cached_manager_s_agent
    elif agent_type == "manager_m":
        is_valid, error_msg = validate_manager_m_config(config)
        create_func = create_cached_manager_m_agent
    elif agent_type == "manager_i":
        is_valid, error_msg = validate_manager_i_config(config)
        create_func = create_cached_manager_i_agent
    else:
        st.error(f"❌ 알 수 없는 에이전트 타입: {agent_type}")
        return None

    if not is_valid:
        st.error(f"❌ 설정 검증 실패: {error_msg}")
        return None

    # 에이전트 생성
    try:
        with st.spinner(f"{agent_type.upper()} 에이전트 초기화 중..."):
            agent = create_func(**config)
        st.success(f"✅ {agent_type.upper()} 에이전트 초기화 완료!")
        return agent
    except Exception as e:
        st.error(f"❌ 에이전트 초기화 실패: {str(e)}")
        with st.expander("상세 에러 정보"):
            import traceback
            st.code(traceback.format_exc())
        return None


def clear_agent_cache():
    """모든 캐시된 에이전트 삭제"""
    st.cache_resource.clear()
    st.success("✅ 모든 에이전트 캐시가 삭제되었습니다.")