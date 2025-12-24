"""
Streamlit HITL UI - Team-H Graph

통합 에이전트 시스템 with Human-in-the-Loop
"""

import sys
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
import uuid

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# 프로젝트 루트의 .env 로드
load_dotenv(project_root / ".env")

# Note: TeamHGraph import 제거됨 (백엔드 분리 원칙)
# FastAPI (api/main.py)가 TeamHGraph를 관리하고,
# 이 Streamlit 앱은 FastAPI 클라이언트로만 동작합니다.

# 공통 컴포넌트 import
from streamlits.ui.components import (
    display_chat_message,
    create_session_state_defaults,
    render_error_expander,
    render_audio_input_widget,
)
from streamlits.ui.approval import render_approval_ui_refactored
from streamlits.core.config import (
    PAGE_CONFIGS,
    DEFAULT_VALUES,
    get_env_defaults,
)
from streamlits.core.auth import simple_auth, show_auth_status
from config.settings import auth_config

# FastAPI 클라이언트 import
from streamlits.utils.fastapi_client import FastAPIClient

# 페이지 설정
page_config = PAGE_CONFIGS["team_h"]
st.set_page_config(
    page_title=page_config["page_title"],
    page_icon=page_config["page_icon"],
    layout=page_config["layout"]
)

# ============================================================================
# 기기 인증 (외부 접속 보호)
# ============================================================================
# .env 파일에서 STREAMLIT_AUTH_ENABLED=true로 설정하면 활성화
# STREAMLIT_AUTH_PASSWORD에 비밀번호 설정
if auth_config.streamlit_auth_enabled and auth_config.streamlit_auth_password:
    if not simple_auth(
        password=auth_config.streamlit_auth_password,
        expiry_days=auth_config.streamlit_auth_expiry_days
    ):
        st.stop()

st.title(page_config["title"])
st.caption(page_config["caption"])


# ============================================================================
# 세션 상태 초기화
# ============================================================================

def initialize_session_state():
    """세션 상태 초기화 (FastAPI 클라이언트 사용)"""
    # 브라우저 세션당 고유 session_id 생성 (통합 ID 전략)
    # session_id = PostgreSQL thread_id = Langfuse session_id
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        print(f"[🆕] New session created: {st.session_state.session_id}")

    create_session_state_defaults(
        messages=[],
        user_id=DEFAULT_VALUES["user_id"],
        thread_id=st.session_state.session_id,  # session_id를 thread_id로 사용
        pending_approval=None,
        approval_decisions={},  # HITL 승인 결정 저장
        # UI 설정
        view_mode="💬 채팅",  # 화면 모드 (채팅/옵션)
        input_mode="💬 텍스트",  # 입력 방식 (텍스트/음성)
        # FastAPI 클라이언트
        api_client=FastAPIClient(base_url=os.getenv("FASTAPI_URL", "http://localhost:8000")),
    )


# ============================================================================
# Agent는 FastAPI 서버에서 관리
# Streamlit은 단순 UI 클라이언트로 동작
# ============================================================================


# ============================================================================
# 응답 처리 - FastAPI 클라이언트를 통한 스트리밍으로 처리
# ============================================================================
# 응답은 SSE 스트림으로 실시간 수신됩니다


# ============================================================================
# 메인
# ============================================================================

initialize_session_state()

# ============================================================================
# 사이드바
# ============================================================================
with st.sidebar:
    st.header("⚙️ 설정")

    # 인증 상태 표시
    if auth_config.streamlit_auth_enabled:
        show_auth_status()
        st.divider()

    # 화면 모드 선택
    st.subheader("📱 화면 모드")
    view_mode = st.radio(
        "표시할 화면 선택",
        ["💬 채팅", "⚙️ 옵션"],
        index=0 if st.session_state.view_mode == "💬 채팅" else 1,
        horizontal=True,
        label_visibility="collapsed"
    )

    # 모드 변경 시 세션 상태 업데이트
    if view_mode != st.session_state.view_mode:
        st.session_state.view_mode = view_mode

    st.divider()

    # 세션 정보
    st.info(f"""
**📊 세션 정보**
- Session ID: `{st.session_state.session_id[:8]}...`
- User ID: `{st.session_state.user_id}`
- 메시지 수: {len(st.session_state.messages)}
    """)

    st.session_state.user_id = st.text_input(
        "사용자 ID",
        value=st.session_state.user_id
    )

# ============================================================================
# 메인 화면 - 사이드바 선택에 따라 조건부 렌더링
# ============================================================================

# 채팅 화면
if st.session_state.view_mode == "💬 채팅":
    # 승인 대기 중이면 먼저 표시
    if render_approval_ui_refactored():
        st.info("👆 위의 작업을 승인 또는 거부해주세요")
        st.stop()

    # 채팅 히스토리 표시
    for msg in st.session_state.messages:
        display_chat_message(
            msg["role"],
            msg["content"],
            agent_name=msg.get("agent_name")
        )
        # 로그가 있으면 표시
        if "logs" in msg and msg["logs"]:
            with st.expander("📜 과정 로그 보기", expanded=False):
                for log in msg["logs"]:
                    st.markdown(log)

    # 입력 영역 (항상 화면 하단에 고정)
    # 텍스트 입력은 항상 표시 (고정 위치)
    prompt = st.chat_input("메시지 입력...")

# ============================================================================
# 옵션 화면
# ============================================================================
elif st.session_state.view_mode == "⚙️ 옵션":
    st.header("⚙️ 설정 및 옵션")

    # ========================================================================
    # 세션 관리
    # ========================================================================
    st.subheader("🔧 세션 관리")

    if st.button("🆕 새 대화 시작", use_container_width=True):
        # 새 session_id 생성
        old_session = st.session_state.session_id
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.thread_id = st.session_state.session_id
        st.session_state.messages = []
        st.session_state.pending_approval = None
        st.session_state.approval_decisions = {}
        print(f"[🔄] Session changed: {old_session[:8]}... → {st.session_state.session_id[:8]}...")
        st.success("새 대화를 시작했습니다!")
        st.rerun()

    st.caption("채팅 내역을 초기화하고 새 세션으로 시작합니다")

    st.divider()

    # ========================================================================
    # 입력 방식 설정
    # ========================================================================
    st.subheader("📝 입력 방식")

    # 입력 방식 선택
    input_mode_option = st.radio(
        "메시지 입력 방식 선택",
        ["💬 텍스트", "🎤 음성"],
        index=0 if st.session_state.input_mode == "💬 텍스트" else 1,
        horizontal=True
    )

    # 선택 변경 시 세션 상태 업데이트
    if input_mode_option != st.session_state.input_mode:
        st.session_state.input_mode = input_mode_option
        st.success(f"입력 방식이 {input_mode_option}(으)로 변경되었습니다!")

    # 음성 입력 모드일 때 음성 입력 위젯 표시
    if st.session_state.input_mode == "🎤 음성":
        st.divider()
        st.caption("🎤 음성 입력 테스트")
        audio_text = render_audio_input_widget("options_test")
        if audio_text:
            st.success(f"인식된 텍스트: {audio_text}")
            st.info("💡 채팅 탭에서 음성 입력을 사용할 수 있습니다.")

    st.divider()

    # ========================================================================
    # Manager 활성화 설정 (백엔드에서 관리)
    # ========================================================================
    st.subheader("🤖 Manager 활성화")

    st.info("""
    ℹ️ Manager 설정은 FastAPI 백엔드에서 환경 변수로 관리됩니다.

    `.env` 파일에서 다음 설정을 변경하세요:
    - `HOMEASSISTANT_TOKEN` (Manager I)
    - `TAVILY_API_KEY` (Manager S)
    - `GOOGLE_CALENDAR_CREDENTIALS_PATH` (Manager T)
    - Manager M은 항상 활성화됩니다.

    변경 후 FastAPI 서버를 재시작해야 합니다.
    """)

# ============================================================================
# 입력 처리 (화면 모드와 무관하게 실행)
# ============================================================================

# 음성 입력 처리 (채팅 화면이고 음성 모드일 때만)
if st.session_state.view_mode == "💬 채팅" and st.session_state.input_mode == "🎤 음성":
    st.caption("🎤 아래 녹음 버튼을 눌러 음성을 입력하세요")
    audio_text = render_audio_input_widget("main_chat")
    if audio_text:
        prompt = audio_text

# 입력이 있을 때 처리 (채팅 화면에서만)
# prompt는 채팅 화면의 st.chat_input() 또는 음성 입력에서 정의됨
if st.session_state.view_mode == "💬 채팅" and 'prompt' in locals() and prompt:
    # 사용자 메시지
    st.session_state.messages.append({"role": "user", "content": prompt})
    display_chat_message("user", prompt)

    # FastAPI를 통한 스트리밍 실행
    try:
        # 툴/라우터 로그를 위한 컨테이너를 먼저 생성 (사용자 메시지와 AI 응답 사이)
        logs_container = st.container()

        # 응답을 실시간으로 표시할 placeholder (st.empty()로 내용 교체)
        message_placeholder = st.empty()
        full_response = ""
        current_node = "unknown"
        current_agent_name = "Assistant"
        avatar = "🤖"  # 기본 아이콘

        # Agent별 응답 저장 (handoff 시 여러 agent가 응답)
        agent_responses = []  # [(agent_name, avatar, response_text), ...]

        # Agent code → 이름/아이콘 매핑 (state의 current_agent 값 사용)
        agent_to_info = {
            "i": ("Manager I", "🏠"),
            "m": ("Manager M", "🧠"),
            "s": ("Manager S", "🔍"),
            "t": ("Manager T", "📅"),
        }

        # FastAPI SSE 스트림
        for event in st.session_state.api_client.chat_stream(
            message=prompt,
            thread_id=st.session_state.session_id,
            user_id=st.session_state.user_id,
        ):
            event_type = event.get("event")

            # Agent 시작 이벤트 (초기 agent 정보)
            if event_type == "agent_start":
                agent_code = event.get("current_agent")
                if agent_code and agent_code in agent_to_info:
                    current_agent_name, avatar = agent_to_info[agent_code]
                    current_node = agent_code

            # Agent 변경 이벤트 (handoff 발생)
            elif event_type == "agent_change":
                # 이전 agent의 응답 저장
                if full_response:
                    agent_responses.append((current_agent_name, avatar, full_response))
                    full_response = ""  # 새 agent를 위해 리셋

                # 새 agent 정보 설정
                agent_code = event.get("current_agent")
                if agent_code and agent_code in agent_to_info:
                    current_agent_name, avatar = agent_to_info[agent_code]
                    current_node = agent_code

            # 토큰 스트리밍 (실시간 표시)
            elif event_type == "token":
                full_response += event.get("content", "")

                # 전체 메시지 렌더링 (이전 agent들 + 현재 agent)
                with message_placeholder.container():
                    # 이전에 완료된 agent들의 응답 표시
                    for prev_agent_name, prev_avatar, prev_response in agent_responses:
                        st.markdown(f"**{prev_avatar} {prev_agent_name}**")
                        st.markdown(prev_response)
                        st.markdown("---")  # 구분선

                    # 현재 agent의 응답 표시 (스트리밍 중)
                    st.markdown(f"**{avatar} {current_agent_name}**")
                    st.markdown(full_response + "▌")  # 커서 표시

            # LLM 완료
            elif event_type == "llm_end":
                full_response = event.get("full_message", full_response)
                # llm_end는 표시 업데이트 없이 full_message만 저장

            # 라우터 결정
            elif event_type == "router_decision":
                target = event.get("target_agent", "unknown")
                reason = event.get("reason", "No reason provided")

                # Agent 이름 매핑
                agent_names = {"i": "Manager I", "m": "Manager M", "s": "Manager S", "t": "Manager T"}
                target_name = agent_names.get(target, target)

                with logs_container:
                    with st.status(f"🔀 라우팅: {target_name}", state="complete", expanded=False):
                        st.write(f"**사유:** {reason}")

            # 툴 실행
            elif event_type == "tool_start":
                tool_name = event.get("tool_name")
                with logs_container:
                    with st.status(f"🛠️ {tool_name} 실행 중...", expanded=False):
                        st.write(f"입력: {event.get('tool_input', {})}")

            # 인터럽트 (HITL)
            elif event_type == "interrupt":
                # 마지막 agent 응답도 저장
                if full_response:
                    agent_responses.append((current_agent_name, avatar, full_response))

                st.session_state.pending_approval = {
                    "interrupt": event.get("interrupt"),
                    "thread_id": st.session_state.session_id,
                    "user_id": st.session_state.user_id,
                    "session_id": st.session_state.session_id,
                }
                st.warning("⏸️ 승인이 필요한 작업이 있습니다")
                st.rerun()

            # 완료
            elif event_type == "done":
                # 마지막 agent의 응답 저장
                if full_response:
                    agent_responses.append((current_agent_name, avatar, full_response))

                # 최종 메시지 표시 (모든 agent 응답)
                if agent_responses:
                    with message_placeholder.container():
                        for idx, (agent_name, agent_avatar, response) in enumerate(agent_responses):
                            st.markdown(f"**{agent_avatar} {agent_name}**")
                            st.markdown(response)
                            # 마지막이 아니면 구분선
                            if idx < len(agent_responses) - 1:
                                st.markdown("---")

            # 오류
            elif event_type == "error":
                st.error(f"❌ 오류: {event.get('error')}")
                with st.expander("상세 오류"):
                    st.code(event.get("traceback", ""))

        # 메시지 저장 (각 agent별로 개별 메시지)
        for agent_name, agent_avatar, response in agent_responses:
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "agent_name": agent_name,
            })

    except Exception as e:
        error_msg = f"❌ FastAPI 연결 오류: {e}"
        st.error(error_msg)
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_msg,
        })
        render_error_expander("상세 오류")

st.divider()
st.caption("Team-H for hhyun")
