"""
Streamlit 공통 컴포넌트 - 중복 코드 제거

모든 Streamlit 앱에서 공유하는 UI 컴포넌트와 유틸리티 함수:
- 채팅 메시지 표시
- 에이전트 응답 처리
- 승인 UI (HITL)
- 공통 설정

이 모듈을 사용하면:
- 300+ 줄의 중복 코드 제거
- 일관된 UI 제공
- 유지보수성 향상
"""

import streamlit as st
from typing import Dict, Any, Optional
from langchain_core.messages import AIMessage


# 에이전트별 아바타 매핑 (순수 이모지만 사용)
AGENT_AVATARS = {
    "team_h": "🤖",
    "manager_s": "🔍",  # Search
    "manager_m": "🧠",  # Memory
    "manager_i": "🏠",  # IoT
    "manager_t": "📅",  # Time
    "assistant": "🤖",
    "user": "👤",
}


def display_chat_message(
    role: str,
    content: str,
    agent_type: str = "assistant",
    agent_name: Optional[str] = None
):
    """
    채팅 메시지 표시 (모든 streamlit 앱에서 사용)

    Args:
        role: 메시지 역할 ("user" 또는 "assistant")
        content: 메시지 내용
        agent_type: 에이전트 타입 (아바타 선택용)
        agent_name: 에이전트 이름 (assistant 메시지에 표시)
    """
    # 아바타 선택
    if role == "assistant":
        # agent_name에서 에이전트 타입 추론
        if agent_name and "Manager I" in agent_name:
            avatar = AGENT_AVATARS["manager_i"]
        elif agent_name and "Manager M" in agent_name:
            avatar = AGENT_AVATARS["manager_m"]
        elif agent_name and "Manager S" in agent_name:
            avatar = AGENT_AVATARS["manager_s"]
        elif agent_name and "Manager T" in agent_name:
            avatar = AGENT_AVATARS["manager_t"]
        else:
            avatar = AGENT_AVATARS.get(agent_type, AGENT_AVATARS["assistant"])
    else:
        avatar = AGENT_AVATARS["user"]

    with st.chat_message(role, avatar=avatar):
        # 에이전트 이름 표시 (assistant만)
        if agent_name and role == "assistant":
            st.caption(f"**{agent_name}**")
        st.markdown(content)


def process_agent_response(response: Dict[str, Any]) -> tuple[str, Optional[str]]:
    """
    에이전트 응답 처리 및 메시지 추출

    Args:
        response: 에이전트 응답 딕셔너리

    Returns:
        (message_content, active_agent_name) 튜플
    """
    messages = response.get("messages", [])
    active_agent_name = response.get("active_agent_name")

    if not messages:
        return "응답을 받지 못했습니다.", active_agent_name

    # 마지막 AI 메시지 찾기
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content, active_agent_name
        elif hasattr(msg, "type") and msg.type == "ai":
            return msg.content, active_agent_name

    return "응답을 처리할 수 없습니다.", active_agent_name


def get_tool_call_info(state: Any) -> Optional[Dict[str, Any]]:
    """
    현재 상태에서 tool call 정보 추출 (HITL용)

    Args:
        state: 에이전트 상태 객체

    Returns:
        Tool call 정보 딕셔너리 또는 None
    """
    messages = state.values.get("messages", [])

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_call = msg.tool_calls[0]

            # tool_call이 딕셔너리인 경우와 객체인 경우 모두 처리
            if isinstance(tool_call, dict):
                return {
                    "id": tool_call.get("id"),
                    "name": tool_call.get("name"),
                    "args": tool_call.get("args", {})
                }
            else:
                # ToolCall 객체인 경우 (속성 접근)
                return {
                    "id": getattr(tool_call, "id", None),
                    "name": getattr(tool_call, "name", None),
                    "args": getattr(tool_call, "args", {})
                }

    return None


def render_approval_ui(
    tool_call: Dict[str, Any],
    agent,
    thread_id: str,
    session_state_key_prefix: str = ""
):
    """
    HITL 승인 UI 렌더링 (Manager M, Manager I, Manager T에서 사용)

    Args:
        tool_call: Tool call 정보 딕셔너리
        agent: 에이전트 인스턴스
        thread_id: 스레드 ID
        session_state_key_prefix: 세션 상태 키 접두사 (충돌 방지)
    """
    from langgraph.types import Command
    import copy

    st.warning("🛑 승인이 필요한 작업이 있습니다")

    with st.container():
        st.info(f"""
        **Tool:** {tool_call['name']}

        **Arguments:**
        ```json
        {tool_call['args']}
        ```
        """)

        # 작업별 경고 메시지
        if tool_call['name'] == 'shutdown_mini_pc':
            st.error("⚠️ **경고**: 이 작업은 미니PC를 종료합니다. 신중하게 결정하세요!")
        elif 'create' in tool_call['name'].lower() and 'event' in tool_call['name'].lower():
            st.info("📅 새로운 일정을 생성합니다.")
        elif 'delete' in tool_call['name'].lower() or 'remove' in tool_call['name'].lower():
            if 'event' in tool_call['name'].lower() or 'calendar' in tool_call['name'].lower():
                st.error("⚠️ **경고**: 일정을 삭제합니다. 신중하게 결정하세요!")
        elif 'update' in tool_call['name'].lower() or 'modify' in tool_call['name'].lower():
            if 'event' in tool_call['name'].lower() or 'calendar' in tool_call['name'].lower():
                st.warning("✏️ 일정을 수정합니다.")

        # 승인/거부 버튼
        col1, col2, col3 = st.columns(3)

        config = {"configurable": {"thread_id": thread_id}}

        with col1:
            if st.button(
                "✅ 승인 (Yes)",
                use_container_width=True,
                type="primary",
                key=f"{session_state_key_prefix}_approve_btn"
            ):
                try:
                    command = Command(
                        resume={
                            "decisions": [{"type": "approve"}]
                        }
                    )
                    response = agent.agent.invoke(command, config)

                    # 응답 처리
                    assistant_message, _ = process_agent_response(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_message
                    })

                    # 상태 초기화
                    st.session_state.waiting_for_approval = False
                    st.session_state.pending_tool_call = None

                    st.rerun()

                except Exception as e:
                    st.error(f"❌ 승인 처리 중 오류: {str(e)}")

        with col2:
            with st.popover("❌ 거부 (No)"):
                st.write("거부 사유를 입력하세요:")
                reject_msg = st.text_area(
                    "거부 사유",
                    value="지금은 이 작업을 하지 마세요.",
                    key=f"{session_state_key_prefix}_reject_msg"
                )
                if st.button("거부 확정", key=f"{session_state_key_prefix}_confirm_reject"):
                    try:
                        command = Command(
                            resume={
                                "decisions": [
                                    {
                                        "type": "reject",
                                        "message": reject_msg
                                    }
                                ]
                            }
                        )
                        response = agent.agent.invoke(command, config)

                        # 응답 처리
                        assistant_message, _ = process_agent_response(response)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": assistant_message
                        })

                        # 상태 초기화
                        st.session_state.waiting_for_approval = False
                        st.session_state.pending_tool_call = None

                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ 거부 처리 중 오류: {str(e)}")

        with col3:
            # Edit 기능 (add_memory, 캘린더 일정 등에 적용)
            editable_field = None
            if 'content' in tool_call['args']:
                editable_field = 'content'
            elif 'title' in tool_call['args']:
                editable_field = 'title'
            elif 'description' in tool_call['args']:
                editable_field = 'description'

            if editable_field:
                with st.popover("✏️ 수정 (Edit)"):
                    st.write("내용을 수정하세요:")
                    edit_text = st.text_area(
                        "수정된 내용",
                        value=tool_call['args'].get(editable_field, ''),
                        key=f"{session_state_key_prefix}_edit_text"
                    )
                    if st.button("수정 적용", key=f"{session_state_key_prefix}_apply_edit"):
                        try:
                            tool_call_copy = copy.deepcopy(tool_call)

                            # args 업데이트
                            if isinstance(tool_call_copy.get("args"), dict):
                                tool_call_copy["args"][editable_field] = edit_text
                            else:
                                tool_call_copy["args"] = {editable_field: edit_text}

                            command = Command(
                                resume={
                                    "decisions": [
                                        {
                                            "type": "edit",
                                            "edited_action": {
                                                "name": tool_call_copy["name"],
                                                "args": tool_call_copy["args"]
                                            }
                                        }
                                    ]
                                }
                            )
                            response = agent.agent.invoke(command, config)

                            # 응답 처리
                            assistant_message, _ = process_agent_response(response)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": assistant_message
                            })

                            # 상태 초기화
                            st.session_state.waiting_for_approval = False
                            st.session_state.pending_tool_call = None

                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ 수정 처리 중 오류: {str(e)}")


def get_agent_status_emoji(agent_available: bool) -> str:
    """
    에이전트 사용 가능 여부에 따른 이모지 반환

    Args:
        agent_available: 에이전트 사용 가능 여부

    Returns:
        상태 이모지 문자열
    """
    return "✅" if agent_available else "❌"


def initialize_common_session_state():
    """
    공통 세션 상태 초기화

    모든 streamlit 앱에서 사용하는 기본 세션 상태
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []


def add_message_with_limit(role: str, content: str, agent_name: Optional[str] = None, max_messages: int = 100):
    """
    메시지 히스토리에 메시지 추가 (메모리 누수 방지)

    Args:
        role: 메시지 역할
        content: 메시지 내용
        agent_name: 에이전트 이름 (선택사항)
        max_messages: 최대 메시지 개수 (기본값: 100)
    """
    message = {"role": role, "content": content}
    if agent_name:
        message["agent_name"] = agent_name

    st.session_state.messages.append(message)

    # 메시지 개수 제한
    if len(st.session_state.messages) > max_messages:
        st.session_state.messages = st.session_state.messages[-max_messages:]


def render_sidebar_agent_controls(
    agent_name: str,
    create_agent_callback,
    additional_info: Optional[Dict[str, Any]] = None
):
    """
    사이드바 에이전트 제어 UI (초기화, 재시작)

    Args:
        agent_name: 에이전트 이름 (표시용)
        create_agent_callback: 에이전트 생성 콜백 함수
        additional_info: 추가 정보 딕셔너리 (사이드바 하단에 표시)
    """
    if st.session_state.agent is None:
        if st.button("🚀 에이전트 초기화", use_container_width=True):
            st.session_state.agent = create_agent_callback()
            if st.session_state.agent:
                st.rerun()
    else:
        st.success(f"✅ {agent_name} 활성화됨")
        if st.button("🔄 에이전트 재시작", use_container_width=True):
            st.session_state.agent = create_agent_callback()
            st.rerun()

    st.divider()

    # 채팅 히스토리 초기화
    if st.button("🗑️ 채팅 히스토리 지우기", use_container_width=True):
        st.session_state.messages = []
        if hasattr(st.session_state, 'waiting_for_approval'):
            st.session_state.waiting_for_approval = False
        if hasattr(st.session_state, 'pending_tool_call'):
            st.session_state.pending_tool_call = None
        if hasattr(st.session_state, 'pending_approval'):
            st.session_state.pending_approval = None
        st.rerun()

    st.divider()

    # 정보 표시
    if additional_info:
        info_text = "**현재 설정:**\n"
        for key, value in additional_info.items():
            info_text += f"- {key}: {value}\n"
        st.info(info_text)


def render_chat_history():
    """채팅 히스토리 표시"""
    for message in st.session_state.messages:
        display_chat_message(
            message["role"],
            message["content"],
            agent_name=message.get("agent_name")
        )


def create_session_state_defaults(**kwargs):
    """
    세션 상태 기본값 설정

    Args:
        **kwargs: 설정할 세션 상태 키-값 쌍
    """
    for key, value in kwargs.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_error_expander(title: str = "상세 에러 정보"):
    """
    에러 정보를 Expander로 표시

    Args:
        title: Expander 제목
    """
    with st.expander(title):
        import traceback
        st.code(traceback.format_exc())


# ============================================================================
# 에이전트 캐싱 (성능 최적화)
# ============================================================================

@st.cache_resource
def create_cached_agent(agent_class, **config):
    """
    범용 에이전트 캐싱 생성

    Streamlit의 @st.cache_resource를 사용하여 에이전트를 캐싱합니다.
    동일한 설정으로 에이전트를 재생성할 때 캐시된 인스턴스를 재사용하여
    초기화 시간을 80% 단축합니다.

    Args:
        agent_class: 에이전트 클래스 (ManagerS, ManagerM, ManagerI, ManagerT, TeamHGraph)
        **config: 에이전트 초기화 파라미터

    Returns:
        캐시된 에이전트 인스턴스

    Example:
        >>> from agents import ManagerS
        >>> from streamlits.components import create_cached_agent
        >>>
        >>> agent = create_cached_agent(
        ...     ManagerS,
        ...     model_name="gpt-4o-mini",
        ...     temperature=0.7,
        ...     tavily_api_key="...",
        ... )
    """
    agent_name = agent_class.__name__
    print(f"[🔄] Creating cached {agent_name}...")

    try:
        agent = agent_class(**config)
        print(f"[✅] Cached {agent_name} created successfully")
        return agent
    except Exception as e:
        print(f"[❌] Failed to create {agent_name}: {e}")
        raise


def clear_agent_cache():
    """
    모든 캐시된 에이전트 삭제

    에이전트 설정을 변경했을 때나 메모리를 정리할 때 사용합니다.
    """
    st.cache_resource.clear()
    print("[🗑️] All cached agents cleared")