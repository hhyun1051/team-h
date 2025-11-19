"""
Streamlit 채팅 앱 - Manager M과 대화하기

Manager M은 일반 기억 관리 에이전트입니다.
메모리 작업 시 Human-in-the-Loop을 통해 승인을 요청합니다.
"""

import sys
from pathlib import Path
import streamlit as st
from typing import Dict, Any
from dotenv import load_dotenv

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from agents.manager_m import ManagerM
from langchain_core.messages import AIMessage
from langgraph.types import Command

# .env 로드
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="Manager M Chat",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Manager M - 기억 관리 에이전트")
st.caption("Manager M과 대화하면서 당신의 기억을 관리하세요")


# 세션 상태 초기화
def initialize_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "agent" not in st.session_state:
        st.session_state.agent = None

    if "user_id" not in st.session_state:
        st.session_state.user_id = "default_user"

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "streamlit_thread"

    if "waiting_for_approval" not in st.session_state:
        st.session_state.waiting_for_approval = False

    if "pending_tool_call" not in st.session_state:
        st.session_state.pending_tool_call = None

    # 자동 에이전트 초기화
    if "agent" in st.session_state and st.session_state.agent is None:
        st.session_state.agent = create_agent()


def create_agent():
    """Manager M 에이전트 생성"""
    try:
        with st.spinner("Manager M 에이전트 초기화 중..."):
            agent = ManagerM(
                model_name="gpt-4.1-mini",
                temperature=0.7,
            )
        st.success("✅ Manager M 에이전트 초기화 완료!")
        return agent
    except Exception as e:
        st.error(f"❌ 에이전트 초기화 실패: {str(e)}")
        st.info("💡 .env 파일에 필수 환경변수(OPENAI_API_KEY, QDRANT_PASSWORD 등)가 설정되어 있는지 확인하세요.")
        return None


def display_chat_message(role: str, content: str):
    """채팅 메시지 표시"""
    avatar = "🤖" if role == "assistant" else "👤"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)


def process_agent_response(response: Dict[str, Any]) -> str:
    """에이전트 응답 처리 및 메시지 추출"""
    messages = response.get("messages", [])

    if not messages:
        return "응답을 받지 못했습니다."

    # 마지막 AI 메시지 찾기
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content
        elif hasattr(msg, "type") and msg.type == "ai":
            return msg.content

    return "응답을 처리할 수 없습니다."


def get_tool_call_info(state: Any) -> Dict[str, Any]:
    """현재 상태에서 tool call 정보 추출"""
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


def handle_approval_response(approval_choice: str, edit_text: str = None):
    """승인 응답 처리 - HITL Middleware 형식에 맞춤"""
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    try:
        if approval_choice == "yes":
            # 승인 - approve decision
            command = Command(
                resume={
                    "decisions": [
                        {
                            "type": "approve"
                        }
                    ]
                }
            )
            response = st.session_state.agent.agent.invoke(command, config)

        elif approval_choice == "no":
            # 거부 - reject decision with message
            command = Command(
                resume={
                    "decisions": [
                        {
                            "type": "reject",
                            "message": "User rejected this action."
                        }
                    ]
                }
            )
            response = st.session_state.agent.agent.invoke(command, config)

        elif approval_choice == "edit" and edit_text:
            # 수정 - edit decision with edited_action
            import copy
            tool_call = copy.deepcopy(st.session_state.pending_tool_call)

            # args 업데이트
            if isinstance(tool_call.get("args"), dict):
                tool_call["args"]["content"] = edit_text
            else:
                tool_call["args"] = {"content": edit_text}

            command = Command(
                resume={
                    "decisions": [
                        {
                            "type": "edit",
                            "edited_action": {
                                "name": tool_call["name"],
                                "args": tool_call["args"]
                            }
                        }
                    ]
                }
            )
            response = st.session_state.agent.agent.invoke(command, config)

        # 응답 처리
        assistant_message = process_agent_response(response)
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
        # 디버그 정보 출력
        with st.expander("디버그 정보"):
            import traceback
            st.code(traceback.format_exc())
            st.write("Pending tool call:", st.session_state.pending_tool_call)


# 세션 상태 초기화
initialize_session_state()

# 사이드바: 설정
with st.sidebar:
    st.header("⚙️ 설정")

    # 사용자 ID 입력
    user_id = st.text_input(
        "사용자 ID",
        value=st.session_state.user_id,
        help="메모리 저장 시 사용할 사용자 ID"
    )
    if user_id != st.session_state.user_id:
        st.session_state.user_id = user_id

    st.divider()

    # 에이전트 초기화 버튼
    if st.session_state.agent is None:
        if st.button("🚀 에이전트 초기화", use_container_width=True):
            st.session_state.agent = create_agent()
    else:
        st.success("✅ 에이전트 활성화됨")
        if st.button("🔄 에이전트 재시작", use_container_width=True):
            st.session_state.agent = create_agent()

    st.divider()

    # 채팅 히스토리 초기화
    if st.button("🗑️ 채팅 히스토리 지우기", use_container_width=True):
        st.session_state.messages = []
        st.session_state.waiting_for_approval = False
        st.session_state.pending_tool_call = None
        st.rerun()

    st.divider()

    # 정보 표시
    st.info(f"""
    **현재 설정:**
    - 사용자 ID: `{st.session_state.user_id}`
    - Thread ID: `{st.session_state.thread_id}`
    - 메시지 수: {len(st.session_state.messages)}
    """)

    st.divider()

    # 사용 가이드
    with st.expander("📖 사용 가이드"):
        st.markdown("""
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
        """)

# 메인 채팅 영역
st.divider()

# 채팅 히스토리 표시
for message in st.session_state.messages:
    display_chat_message(message["role"], message["content"])

# 승인 대기 중이면 승인 UI 표시
if st.session_state.waiting_for_approval and st.session_state.pending_tool_call:
    tool_call = st.session_state.pending_tool_call

    st.warning("🛑 승인이 필요한 작업이 있습니다")

    with st.container():
        st.info(f"""
        **Tool:** {tool_call['name']}

        **Arguments:**
        ```json
        {tool_call['args']}
        ```
        """)

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("✅ 승인 (Yes)", use_container_width=True, type="primary"):
                handle_approval_response("yes")

        with col2:
            if st.button("❌ 거부 (No)", use_container_width=True):
                handle_approval_response("no")

        with col3:
            with st.popover("✏️ 수정 (Edit)"):
                st.write("내용을 수정하세요:")
                edit_text = st.text_area(
                    "수정된 내용",
                    value=tool_call['args'].get('content', ''),
                    key="edit_text_area"
                )
                if st.button("수정 적용", key="apply_edit"):
                    handle_approval_response("edit", edit_text)

# 채팅 입력
if not st.session_state.waiting_for_approval:
    if prompt := st.chat_input("Manager M에게 메시지를 입력하세요..."):
        # 에이전트가 초기화되지 않았으면 경고
        if st.session_state.agent is None:
            st.warning("⚠️ 먼저 사이드바에서 '에이전트 초기화' 버튼을 클릭하세요.")
            st.stop()

        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)

        # 에이전트 응답 생성
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("생각하는 중..."):
                try:
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}

                    # 에이전트 실행
                    response = st.session_state.agent.invoke(
                        message=prompt,
                        user_id=st.session_state.user_id,
                        thread_id=st.session_state.thread_id,
                    )

                    # 상태 확인 - interrupt가 있는지 체크
                    state = st.session_state.agent.agent.get_state(config)

                    if state.next:  # interrupt가 있음
                        # Tool call 정보 추출
                        tool_call_info = get_tool_call_info(state)

                        if tool_call_info:
                            st.session_state.waiting_for_approval = True
                            st.session_state.pending_tool_call = tool_call_info

                            st.info("🔔 승인이 필요한 작업이 있습니다. 위의 승인 UI를 확인하세요.")
                            st.rerun()
                    else:
                        # interrupt가 없으면 정상 응답 처리
                        assistant_message = process_agent_response(response)
                        st.markdown(assistant_message)

                        # 응답 저장
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": assistant_message
                        })

                        # 디버그: 전체 응답 확인 (개발용)
                        with st.expander("🔍 디버그: 전체 응답 보기"):
                            st.json({
                                "messages": str(response.get("messages", [])),
                                "state_next": state.next,
                            })

                except Exception as e:
                    error_msg = f"❌ 오류 발생: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

                    # 상세 에러 정보
                    with st.expander("상세 에러 정보"):
                        import traceback
                        st.code(traceback.format_exc())

# 푸터
st.divider()
st.caption("Built with Streamlit + LangChain + Manager M | HITL Enabled")
