"""
Streamlit HITL UI - 간단 버전

에러 방지를 위해 안전하게 작성된 버전
"""

import sys
from pathlib import Path
import streamlit as st
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.team_h_graph import TeamHGraph
from langchain_core.messages import AIMessage
from langgraph.types import Command

load_dotenv()

st.set_page_config(
    page_title="Team-H HITL",
    page_icon="✋",
    layout="wide"
)

st.title("✋ Team-H - Human-in-the-Loop")
st.caption("승인이 필요한 작업은 사용자 확인 후 실행됩니다")


def initialize_session_state():
    """세션 상태 초기화"""
    defaults = {
        "messages": [],
        "agent": None,
        "user_id": "default_user",
        "thread_id": "streamlit_teamh_thread",
        "routing_history": [],
        "pending_approval": None,  # HITL
        "smartthings_token": os.getenv("SMARTTHINGS_TOKEN", ""),
        "tavily_api_key": os.getenv("TAVILY_API_KEY", ""),
        "device_config": {
            "living_room_speaker_outlet": os.getenv("SPEAKER_ID", ""),
            "living_room_light": os.getenv("PROJECTOR_ID", ""),
            "bedroom_light": os.getenv("VERTICAL_MONITOR_ID", ""),
            "bathroom_light": os.getenv("AIR_PURIFIER_ID", ""),
        },
        "enable_manager_i": True,
        "enable_manager_m": True,
        "enable_manager_s": True,
        "enable_manager_t": True,
        "google_credentials_path": os.getenv("GOOGLE_CALENDAR_CREDENTIALS_PATH"),
        "google_token_path": os.getenv("GOOGLE_CALENDAR_TOKEN_PATH"),
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def create_agent():
    """에이전트 생성"""
    try:
        with st.spinner("초기화 중..."):
            agent = TeamHGraph(
                enable_manager_i=st.session_state.enable_manager_i,
                enable_manager_m=st.session_state.enable_manager_m,
                enable_manager_s=st.session_state.enable_manager_s,
                enable_manager_t=st.session_state.enable_manager_t,
                smartthings_token=st.session_state.smartthings_token or None,
                device_config=st.session_state.device_config,
                tavily_api_key=st.session_state.tavily_api_key or None,
                max_search_results=5,
                google_credentials_path=st.session_state.google_credentials_path,
                google_token_path=st.session_state.google_token_path,
                model_name="gpt-4o-mini",
                temperature=0.7,
            )
        st.success("✅ 초기화 완료!")
        return agent
    except Exception as e:
        st.error(f"❌ 초기화 실패: {e}")
        return None


def display_message(role: str, content: str, agent_name: Optional[str] = None):
    """메시지 표시"""
    avatar = {
        "user": "👤",
        "assistant": "🤖"
    }.get(role, "💬")
    
    if agent_name:
        if "Manager I" in agent_name:
            avatar = "🏠"
        elif "Manager M" in agent_name:
            avatar = "🧠"
        elif "Manager S" in agent_name:
            avatar = "🔍"
        elif "Manager T" in agent_name:
            avatar = "📅"
    
    with st.chat_message(role, avatar=avatar):
        if agent_name:
            st.caption(f"**{agent_name}**")
        st.markdown(content)


def extract_response(response: Dict[str, Any]) -> tuple[str, Optional[str]]:
    """응답에서 메시지 추출"""
    messages = response.get("messages", [])
    agent_name = response.get("active_agent_name")
    
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content, agent_name
    
    return "응답을 받지 못했습니다.", agent_name


def render_approval_ui():
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
            import json
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
                    import json
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
                            # JSON 파싱
                            edited_args = json.loads(edited_args_json)

                            # Tool name도 수정 가능하게 (선택적)
                            edited_tool_name = st.session_state.get(f"edit_tool_name_{idx}", tool_name)

                            # Command로 전송
                            result = st.session_state.agent.invoke_command(
                                Command(resume={
                                    "decisions": [{
                                        "type": "edit",
                                        "edited_action": {
                                            "name": edited_tool_name,
                                            "args": edited_args
                                        }
                                    }]
                                }),
                                config=config
                            )

                            msg, agent_name = extract_response(result)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": msg,
                                "agent_name": agent_name,
                            })

                            # Edit 모드 해제 및 pending_approval 초기화
                            st.session_state[edit_mode_key] = False
                            st.session_state.pending_approval = None
                            st.success("✅ 편집 완료 및 실행!")
                            st.rerun()
                        except json.JSONDecodeError as e:
                            st.error(f"❌ JSON 파싱 오류: {e}")
                        except Exception as e:
                            st.error(f"편집 완료 중 오류: {e}")
                            import traceback
                            st.code(traceback.format_exc())

                    # 편집 취소 버튼
                    if col2.button("↩️ 취소", key=f"cancel_edit_{idx}", use_container_width=True):
                        st.session_state[edit_mode_key] = False
                        st.rerun()

                # 일반 모드 (버튼들)
                else:
                    # 3개 버튼을 columns로 배치
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
                                result = st.session_state.agent.invoke_command(
                                    Command(resume={"decisions": [{"type": "approve"}]}),
                                    config=config
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
                            # 거부 이유를 입력받을 수 있도록 modal 또는 text_input 추가 (선택적)
                            reject_reason = st.session_state.get(f"reject_reason_{idx}", "사용자가 거부했습니다")

                            try:
                                result = st.session_state.agent.invoke_command(
                                    Command(resume={
                                        "decisions": [{
                                            "type": "reject",
                                            "message": reject_reason
                                        }]
                                    }),
                                    config=config
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
        with st.expander("상세 오류"):
            import traceback
            st.code(traceback.format_exc())
        st.session_state.pending_approval = None
        return False


# 초기화
initialize_session_state()

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")
    
    st.session_state.user_id = st.text_input(
        "사용자 ID",
        value=st.session_state.user_id
    )
    
    st.divider()
    
    if st.session_state.agent is None:
        if st.button("🚀 에이전트 초기화", use_container_width=True):
            st.session_state.agent = create_agent()
            if st.session_state.agent:
                st.rerun()
    else:
        st.success("✅ 에이전트 활성화됨")
        
        if st.button("🔄 재시작", use_container_width=True):
            st.session_state.agent = create_agent()
            st.rerun()
    
    st.divider()
    
    if st.button("🗑️ 채팅 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.routing_history = []
        st.session_state.pending_approval = None
        st.rerun()
    
    st.divider()
    
    st.info(f"""
    **상태:**
    - 메시지: {len(st.session_state.messages)}
    - 승인 대기: {'있음' if st.session_state.pending_approval else '없음'}
    """)

# 메인
st.divider()

# 승인 대기 중이면 먼저 표시
if render_approval_ui():
    st.info("👆 위의 작업을 승인 또는 거부해주세요")
    st.stop()

# 채팅 히스토리
for msg in st.session_state.messages:
    display_message(
        msg["role"],
        msg["content"],
        msg.get("agent_name")
    )

# 입력
if prompt := st.chat_input("메시지 입력..."):
    if st.session_state.agent is None:
        st.warning("⚠️ 먼저 에이전트를 초기화하세요")
        st.stop()
    
    # 사용자 메시지
    st.session_state.messages.append({"role": "user", "content": prompt})
    display_message("user", prompt)
    
    # 에이전트 실행
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("생각 중..."):
            try:
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                
                result = st.session_state.agent.invoke(
                    message=prompt,
                    user_id=st.session_state.user_id,
                    thread_id=st.session_state.thread_id,
                )
                
                # Interrupt 확인
                if "__interrupt__" in result:
                    st.session_state.pending_approval = {
                        "interrupt": result["__interrupt__"][0],
                        "config": config,
                    }
                    st.info("⏸️ 승인이 필요합니다")
                    st.rerun()
                
                # 정상 응답
                msg, agent_name = extract_response(result)
                active = result.get("active_agent")
                
                if active:
                    st.session_state.routing_history.append(active)
                
                if agent_name:
                    st.caption(f"**{agent_name}**")
                
                st.markdown(msg)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": msg,
                    "agent_name": agent_name,
                })
                
            except Exception as e:
                error_msg = f"❌ 오류: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })
                
                with st.expander("상세 오류"):
                    import traceback
                    st.code(traceback.format_exc())

st.divider()
st.caption("Team-H with Human-in-the-Loop")