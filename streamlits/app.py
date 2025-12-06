"""
Streamlit HITL UI - Team-H Graph

통합 에이전트 시스템 with Human-in-the-Loop
"""

import sys
from pathlib import Path
import streamlit as st
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import json
import uuid
from openai import OpenAI

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# 프로젝트 루트의 .env 로드
load_dotenv(project_root / ".env")

# Agents import
try:
    from agents.graph import TeamHGraph
    from langchain_core.messages import AIMessage
    from langgraph.types import Command
except ImportError as e:
    st.error(f"""
    ❌ TeamHGraph import 실패!

    필요한 패키지를 설치하세요:
    ```bash
    pip install langfuse langgraph
    ```

    에러: {e}
    """)
    st.stop()

# 공통 컴포넌트 import
from streamlits.ui.components import (
    display_chat_message,
    create_session_state_defaults,
    render_error_expander,
    create_cached_agent,
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
    """세션 상태 초기화 및 에이전트 자동 생성"""
    env_defaults = get_env_defaults()

    # 브라우저 세션당 고유 session_id 생성 (통합 ID 전략)
    # session_id = PostgreSQL thread_id = Langfuse session_id
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        print(f"[🆕] New session created: {st.session_state.session_id}")

    create_session_state_defaults(
        messages=[],
        agent=None,
        user_id=DEFAULT_VALUES["user_id"],
        thread_id=st.session_state.session_id,  # session_id를 thread_id로 사용
        routing_history=[],
        pending_approval=None,
        # Home Assistant 설정 (Manager I용)
        homeassistant_url=env_defaults["homeassistant_url"],
        homeassistant_token=env_defaults["homeassistant_token"],
        tavily_api_key=env_defaults["tavily_api_key"],
        google_credentials_path=env_defaults["google_credentials_path"],
        google_token_path=env_defaults["google_token_path"],
        # Manager M (Qdrant + Embedding) 설정
        embedding_type=env_defaults["embedding_type"],
        embedder_url=env_defaults["embedder_url"],
        openai_api_key=env_defaults["openai_api_key"],
        embedding_dims=env_defaults["embedding_dims"],
        qdrant_url=env_defaults["qdrant_url"],
        qdrant_api_key=env_defaults["qdrant_api_key"],
        m_collection_name=env_defaults["m_collection_name"],
        enable_manager_i=True,
        enable_manager_m=True,
        enable_manager_s=True,
        enable_manager_t=True,
        agent_initialized=False,  # 자동 초기화 완료 플래그
    )

    # 에이전트 자동 초기화 (첫 실행 시에만)
    if st.session_state.agent is None and not st.session_state.agent_initialized:
        st.session_state.agent = create_agent()
        st.session_state.agent_initialized = True


# ============================================================================
# 에이전트 생성
# ============================================================================

def create_agent():
    """Team-H Graph 에이전트 생성 (캐싱 적용)"""
    try:
        with st.spinner("초기화 중..."):
            agent = create_cached_agent(
                TeamHGraph,
                enable_manager_i=st.session_state.enable_manager_i,
                enable_manager_m=st.session_state.enable_manager_m,
                enable_manager_s=st.session_state.enable_manager_s,
                enable_manager_t=st.session_state.enable_manager_t,
                # Home Assistant 설정 (Manager I용)
                homeassistant_url=st.session_state.homeassistant_url,
                homeassistant_token=st.session_state.homeassistant_token or None,
                # Manager M (Qdrant + Embedding) 설정
                embedding_type=st.session_state.embedding_type,
                embedder_url=st.session_state.embedder_url,
                openai_api_key=st.session_state.openai_api_key,
                embedding_dims=st.session_state.embedding_dims,
                qdrant_url=st.session_state.qdrant_url,
                qdrant_api_key=st.session_state.qdrant_api_key,
                m_collection_name=st.session_state.m_collection_name,
                # Manager S 설정
                tavily_api_key=st.session_state.tavily_api_key or None,
                max_search_results=5,
                # Manager T 설정
                google_credentials_path=st.session_state.google_credentials_path,
                google_token_path=st.session_state.google_token_path,
                # 공통 설정
                model_name=DEFAULT_VALUES["model_name"],
                temperature=DEFAULT_VALUES["temperature"],
            )
        st.success("✅ 초기화 완료!")
        return agent
    except Exception as e:
        st.error(f"❌ 초기화 실패: {e}")
        return None


# ============================================================================
# 응답 처리
# ============================================================================

def extract_response(response: Dict[str, Any]) -> tuple[str, Optional[str]]:
    """응답에서 메시지 추출"""
    messages = response.get("messages", [])

    # current_agent 또는 last_active_manager에서 agent_name 추출
    current_agent = response.get("current_agent") or response.get("last_active_manager")

    # agent_name 매핑 (i, m, s, t -> Manager I, Manager M 등)
    agent_name_map = {
        "i": "Manager I",
        "m": "Manager M",
        "s": "Manager S",
        "t": "Manager T",
    }
    agent_name = agent_name_map.get(current_agent) if current_agent else None

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content, agent_name

    return "응답을 받지 못했습니다.", agent_name


# ============================================================================
# HITL 승인 UI (Legacy - 사용 안 함)
# ============================================================================
# 새로운 approval_ui_refactored.py 사용
# 기존 코드는 백업용으로 보관

def render_approval_ui_legacy():
    """HITL 승인 UI"""
    if not st.session_state.pending_approval:
        return False

    approval_data = st.session_state.pending_approval
    interrupt = approval_data["interrupt"]
    config = approval_data["config"]

    st.divider()
    st.warning("⏸️ 승인이 필요한 작업이 있습니다", icon="✋")

    # 전체 interrupt 구조 확인 (디버깅용)
    with st.expander("🐛 디버그: 전체 구조", expanded=False):
        st.code(f"Type: {type(interrupt.value).__name__}")
        try:
            st.code(json.dumps(interrupt.value, indent=2, default=str))
        except:
            st.text(str(interrupt.value))

    # action_requests 안전하게 추출
    try:
        action_requests = interrupt.value.get("action_requests", [])
        review_configs = interrupt.value.get("review_configs", [])

        if not action_requests:
            st.error("❌ action_requests가 비어있습니다")
            st.session_state.pending_approval = None
            return False

        # 각 작업 표시
        for idx, (action, review) in enumerate(zip(action_requests, review_configs)):
            with st.expander(f"🔍 작업 {idx + 1}: {action.get('name', 'Unknown')}", expanded=True):
                # 설명
                st.markdown(f"**설명:**")
                st.text(action.get('description', 'N/A'))

                # 전체 action 정보 (디버깅)
                with st.expander("상세 정보", expanded=False):
                    st.json(action)

                # 승인 가능한 결정 타입
                allowed = review.get("allowed_decisions", ["approve", "reject"])
                st.caption(f"허용된 결정: {', '.join(allowed)}")

                # Edit 모드 체크
                edit_mode_key = f"edit_mode_{idx}"
                if edit_mode_key not in st.session_state:
                    st.session_state[edit_mode_key] = False

                # Edit 모드가 활성화된 경우
                if st.session_state[edit_mode_key]:
                    st.info("✏️ 편집 모드: 아래에서 tool arguments를 수정하세요")

                    # Arguments 표시 및 수정
                    original_args = action.get('arguments', {})
                    tool_name = action.get('name', '')

                    st.markdown(f"**Tool Name:** `{tool_name}`")

                    # JSON 형태로 arguments 편집
                    args_json = json.dumps(original_args, indent=2, ensure_ascii=False)
                    edited_args_json = st.text_area(
                        "Arguments (JSON 형식):",
                        value=args_json,
                        height=200,
                        key=f"edit_args_{idx}"
                    )

                    col1, col2 = st.columns(2)

                    # 편집 완료 버튼
                    if col1.button("✅ 편집 완료", key=f"submit_edit_{idx}", use_container_width=True):
                        try:
                            edited_args = json.loads(edited_args_json)
                            edited_tool_name = st.session_state.get(f"edit_tool_name_{idx}", tool_name)

                            # 모든 action_requests에 대해 decisions 생성
                            # 현재 편집 중인 것은 edit, 나머지는 거부
                            num_actions = len(action_requests)
                            decisions = []
                            for i in range(num_actions):
                                if i == idx:
                                    decisions.append({
                                        "type": "edit",
                                        "edited_action": {
                                            "name": edited_tool_name,
                                            "args": edited_args
                                        }
                                    })
                                else:
                                    decisions.append({"type": "reject", "message": "사용자가 다른 작업만 편집함"})

                            result = st.session_state.agent.invoke_command(
                                Command(resume={"decisions": decisions}),
                                config=approval_data["config"],
                                user_id=approval_data["user_id"],
                                thread_id=approval_data["thread_id"],
                                session_id=approval_data["session_id"]
                            )

                            msg, agent_name = extract_response(result)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": msg,
                                "agent_name": agent_name,
                            })

                            st.session_state[edit_mode_key] = False
                            st.session_state.pending_approval = None
                            st.success("✅ 편집 완료 및 실행!")
                            st.rerun()
                        except json.JSONDecodeError as e:
                            st.error(f"❌ JSON 파싱 오류: {e}")
                        except Exception as e:
                            st.error(f"편집 완료 중 오류: {e}")
                            render_error_expander()

                    # 편집 취소 버튼
                    if col2.button("↩️ 취소", key=f"cancel_edit_{idx}", use_container_width=True):
                        st.session_state[edit_mode_key] = False
                        st.rerun()

                # 일반 모드 (버튼들)
                else:
                    num_buttons = sum([
                        "approve" in allowed,
                        "edit" in allowed,
                        "reject" in allowed
                    ])

                    if num_buttons == 3:
                        col1, col2, col3 = st.columns(3)
                    elif num_buttons == 2:
                        col1, col2 = st.columns(2)
                        col3 = None
                    else:
                        col1 = st
                        col2 = None
                        col3 = None

                    # 승인 버튼
                    if "approve" in allowed:
                        target_col = col1 if num_buttons >= 1 else st
                        if target_col.button("✅ 승인", key=f"approve_{idx}", use_container_width=True):
                            try:
                                # 현재 작업만 승인, 나머지는 거부
                                num_actions = len(action_requests)
                                decisions = []
                                for i in range(num_actions):
                                    if i == idx:
                                        decisions.append({"type": "approve"})
                                    else:
                                        decisions.append({"type": "reject", "message": "사용자가 다른 작업만 승인함"})
                                result = st.session_state.agent.invoke_command(
                                    Command(resume={"decisions": decisions}),
                                    config=approval_data["config"],
                                    user_id=approval_data["user_id"],
                                    thread_id=approval_data["thread_id"],
                                    session_id=approval_data["session_id"]
                                )

                                msg, agent_name = extract_response(result)
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": msg,
                                    "agent_name": agent_name,
                                })

                                st.session_state.pending_approval = None
                                st.success("✅ 승인 완료!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"승인 중 오류: {e}")

                    # 편집 버튼
                    if "edit" in allowed:
                        target_col = col2 if num_buttons >= 2 else col1 if num_buttons >= 1 else st
                        if target_col.button("✏️ 편집", key=f"edit_{idx}", use_container_width=True):
                            st.session_state[edit_mode_key] = True
                            st.rerun()

                    # 거부 버튼
                    if "reject" in allowed:
                        if num_buttons == 3:
                            target_col = col3
                        elif num_buttons == 2 and "edit" not in allowed:
                            target_col = col2
                        elif num_buttons == 2 and "approve" not in allowed:
                            target_col = col2
                        else:
                            target_col = col1 if num_buttons >= 1 else st

                        if target_col.button("❌ 거부", key=f"reject_{idx}", use_container_width=True):
                            reject_reason = st.session_state.get(f"reject_reason_{idx}", "사용자가 거부했습니다")

                            try:
                                # 현재 작업만 거부, 나머지는 거부 (모두 거부)
                                num_actions = len(action_requests)
                                decisions = []
                                for i in range(num_actions):
                                    if i == idx:
                                        decisions.append({"type": "reject", "message": reject_reason})
                                    else:
                                        decisions.append({"type": "reject", "message": "사용자가 다른 작업만 처리함"})
                                result = st.session_state.agent.invoke_command(
                                    Command(resume={"decisions": decisions}),
                                    config=approval_data["config"],
                                    user_id=approval_data["user_id"],
                                    thread_id=approval_data["thread_id"],
                                    session_id=approval_data["session_id"]
                                )

                                msg, agent_name = extract_response(result)
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": msg,
                                    "agent_name": agent_name,
                                })

                                st.session_state.pending_approval = None
                                st.info("ℹ️ 거부 완료")
                                st.rerun()
                            except Exception as e:
                                st.error(f"거부 중 오류: {e}")

        st.divider()
        return True

    except Exception as e:
        st.error(f"❌ 승인 UI 렌더링 오류: {e}")
        render_error_expander("상세 오류")
        st.session_state.pending_approval = None
        return False


# ============================================================================
# 메인
# ============================================================================

initialize_session_state()

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")

    # 인증 상태 표시
    if auth_config.streamlit_auth_enabled:
        show_auth_status()
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

    st.divider()

    # 새 대화 시작 버튼
    if st.button("🆕 새 대화 시작", use_container_width=True):
        # 새 session_id 생성
        old_session = st.session_state.session_id
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.thread_id = st.session_state.session_id
        st.session_state.messages = []
        st.session_state.routing_history = []
        st.session_state.pending_approval = None
        # 에이전트는 유지 (캐싱된 인스턴스 재사용)
        print(f"[🔄] Session changed: {old_session[:8]}... → {st.session_state.session_id[:8]}...")
        st.success("새 대화를 시작했습니다!")
        st.rerun()

    st.divider()

    # 에이전트 상태 표시 및 재시작 버튼
    if st.session_state.agent is not None:
        st.success("✅ 에이전트 활성화됨")
    else:
        st.warning("⏳ 에이전트 초기화 중...")

    if st.button("🔄 에이전트 재시작", use_container_width=True):
        st.session_state.agent = create_agent()
        st.session_state.agent_initialized = True
        st.rerun()

    st.divider()

    if st.button("🗑️ 채팅 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.routing_history = []
        st.session_state.pending_approval = None
        st.rerun()

    st.divider()

    st.caption(f"""
**상태:**
- 승인 대기: {'있음' if st.session_state.pending_approval else '없음'}

**Langfuse 추적:**
[이 세션 보기](http://192.168.0.151:3000/sessions/{st.session_state.session_id})
""")

# 메인
st.divider()

# 승인 대기 중이면 먼저 표시
if render_approval_ui_refactored():
    st.info("👆 위의 작업을 승인 또는 거부해주세요")
    st.stop()

# 채팅 히스토리
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

# 입력 방식 선택
input_mode = st.radio(
    "입력 방식 선택",
    ["💬 텍스트", "🎤 음성"],
    horizontal=True,
    label_visibility="collapsed"
)

# 입력 처리
prompt = None

if input_mode == "💬 텍스트":
    # 텍스트 입력
    prompt = st.chat_input("메시지 입력...")
else:
    # 음성 입력
    st.caption("🎤 녹음 버튼을 눌러 음성을 입력하세요")
    audio_text = render_audio_input_widget("main_chat")
    if audio_text:
        prompt = audio_text

# 입력이 있을 때 처리
if prompt:
    if st.session_state.agent is None:
        st.error("❌ 에이전트 초기화에 실패했습니다. 사이드바에서 '에이전트 재시작' 버튼을 눌러주세요.")
        st.stop()

    # 사용자 메시지
    st.session_state.messages.append({"role": "user", "content": prompt})
    display_chat_message("user", prompt)

    # 에이전트 실행
    with st.spinner("생각 중..."):
        try:
            # 통합 ID 전략: session_id를 thread_id와 session_id 모두로 사용
            config = {"configurable": {"thread_id": st.session_state.session_id}}
            
            # 스트리밍을 위한 상태 컨테이너
            execution_logs = []  # 로그 수집용 리스트
            
            with st.status("🤔 생각 중...", expanded=True) as status:
                # Stream 실행
                for chunk in st.session_state.agent.stream(
                    message=prompt,
                    user_id=st.session_state.user_id,
                    thread_id=st.session_state.session_id,
                    session_id=st.session_state.session_id,
                ):
                    # 청크 처리 및 로그 표시
                    for node_name, updates in chunk.items():
                        # Router 로그
                        if node_name == "router":
                            reason = updates.get("routing_reason", "Unknown reason")
                            target = updates.get("current_agent", "unknown")
                            log_msg = f"🔄 **Router:** {target.upper()}로 전달 ({reason})"
                            status.write(log_msg)
                            execution_logs.append(log_msg)
                        
                        # Manager 로그
                        elif node_name.startswith("manager_"):
                            agent_key = node_name.replace("manager_", "")
                            msgs = updates.get("messages", [])
                            
                            # 새로 생성된 메시지 중 ToolMessage(핸드오프) 확인
                            for msg in msgs:
                                if hasattr(msg, "type") and msg.type == "tool":
                                    # 핸드오프 메시지
                                    log_msg = f"🤝 **{agent_key.upper()}:** 핸드오프 실행 - {msg.content}"
                                    status.write(log_msg)
                                    execution_logs.append(log_msg)
                                elif hasattr(msg, "type") and msg.type == "ai" and msg.tool_calls:
                                    # 툴 호출
                                    for tool_call in msg.tool_calls:
                                        log_msg = f"🛠️ **{agent_key.upper()}:** 툴 호출 - `{tool_call['name']}`"
                                        status.write(log_msg)
                                        execution_logs.append(log_msg)
                
                status.update(label="✅ 완료!", state="complete", expanded=False)

            # 최종 상태 가져오기
            snapshot = st.session_state.agent.graph.get_state(config)
            result = snapshot.values

            # Interrupt 확인 (Next가 있으면 interrupt 상태)
            if snapshot.next:
                # snapshot.tasks에서 interrupt 추출
                interrupts = []
                for task in snapshot.tasks:
                    interrupts.extend(task.interrupts)

                if interrupts:
                    # HITL 승인 대기 상태로 전환 (context 정보도 함께 저장)
                    st.session_state.pending_approval = {
                        "interrupt": interrupts[0],
                        "config": config,
                        "user_id": st.session_state.user_id,
                        "thread_id": st.session_state.session_id,
                        "session_id": st.session_state.session_id
                    }
                    status.update(label="⏸️ 승인 대기 중", state="running", expanded=False)
                    st.rerun()  # UI를 즉시 갱신하여 승인 UI 표시

            # 정상 응답 (interrupt가 없을 때만 실행됨)
            msg, agent_name = extract_response(result)
            active = result.get("active_agent")

            if active:
                st.session_state.routing_history.append(active)

            # 올바른 아바타 선택 (순수 이모지만 사용)
            avatar_map = {
                "Manager I": "🏠",
                "Manager M": "🧠",
                "Manager S": "🔍",
                "Manager T": "📅",
            }
            avatar = avatar_map.get(agent_name, "🤖")

            # 아바타와 함께 메시지 표시
            with st.chat_message("assistant", avatar=avatar):
                if agent_name:
                    st.caption(f"**{agent_name}**")
                st.markdown(msg)

            st.session_state.messages.append({
                "role": "assistant",
                "content": msg,
                "agent_name": agent_name,
                "logs": execution_logs,  # 로그 저장
            })

        except Exception as e:
            error_msg = f"❌ 오류: {e}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
            })

            render_error_expander("상세 오류")

st.divider()
st.caption("Team-H for hhyun")
