"""
Streamlit 채팅 앱 - Manager I와 대화하기

Manager I는 IoT 제어 에이전트입니다.
위험한 작업(미니PC 종료) 시 Human-in-the-Loop을 통해 승인을 요청합니다.
"""

import sys
from pathlib import Path
import streamlit as st
from typing import Dict, Any
from dotenv import load_dotenv
import os

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.manager_i import ManagerI
from langchain_core.messages import AIMessage
from langgraph.types import Command

# .env 로드
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="Manager I Chat",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Manager I - IoT 제어 에이전트")
st.caption("Manager I와 대화하면서 집안의 스마트 기기를 제어하세요")


# 세션 상태 초기화
def initialize_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "agent" not in st.session_state:
        st.session_state.agent = None

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "streamlit_iot_thread"

    if "waiting_for_approval" not in st.session_state:
        st.session_state.waiting_for_approval = False

    if "pending_tool_call" not in st.session_state:
        st.session_state.pending_tool_call = None

    if "smartthings_token" not in st.session_state:
        st.session_state.smartthings_token = os.getenv("SMARTTHINGS_TOKEN", "")

    if "device_config" not in st.session_state:
        # 기본 장치 설정 (환경변수나 기본값 사용)
        st.session_state.device_config = {
            "living_room_speaker_outlet": os.getenv("SPEAKER_ID", "d5ae3413-10a4-4a03-b5e3-eaa0bee64db4"),
            "living_room_light": os.getenv("PROJECTOR_ID", "f28bb22f-4768-685b-076b-b9514941498c"),
            "bedroom_light": os.getenv("VERTICAL_MONITOR_ID", "55ca4824-3237-411b-88fd-efb549927553"),
            "bathroom_light": os.getenv("AIR_PURIFIER_ID", "0897d30e-5cb2-5566-13d5-7de7394061d1"),
        }

    # 자동 에이전트 초기화
    if "agent" in st.session_state and st.session_state.agent is None:
        st.session_state.agent = create_agent()


def create_agent():
    """Manager I 에이전트 생성"""
    try:
        if not st.session_state.smartthings_token:
            st.error("❌ SmartThings Token이 설정되지 않았습니다.")
            return None

        with st.spinner("Manager I 에이전트 초기화 중..."):
            agent = ManagerI(
                model_name="gpt-4o-mini",
                temperature=0.7,
                smartthings_token=st.session_state.smartthings_token,
                device_config=st.session_state.device_config,
            )
        st.success("✅ Manager I 에이전트 초기화 완료!")
        return agent
    except Exception as e:
        st.error(f"❌ 에이전트 초기화 실패: {str(e)}")
        st.info("💡 .env 파일에 SMARTTHINGS_TOKEN이 설정되어 있는지 확인하세요.")
        return None


def display_chat_message(role: str, content: str):
    """채팅 메시지 표시"""
    avatar = "🏠" if role == "assistant" else "👤"
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


def handle_approval_response(approval_choice: str, reject_message: str = None):
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
            message = reject_message or "User rejected this action."
            command = Command(
                resume={
                    "decisions": [
                        {
                            "type": "reject",
                            "message": message
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

    # SmartThings Token 입력
    token = st.text_input(
        "SmartThings Token",
        value=st.session_state.smartthings_token,
        type="password",
        help="SmartThings API 토큰"
    )
    if token != st.session_state.smartthings_token:
        st.session_state.smartthings_token = token

    st.divider()

    # 장치 설정 (확장 가능)
    with st.expander("🔧 장치 설정"):
        st.caption("현재 설정된 장치:")
        for device_key, device_id in st.session_state.device_config.items():
            device_name = device_key.replace("_", " ").title()
            st.text(f"{device_name}:")
            st.caption(f"  {device_id[:8]}...")

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
    - Thread ID: `{st.session_state.thread_id}`
    - 메시지 수: {len(st.session_state.messages)}
    - 장치 수: {len(st.session_state.device_config)}
    """)

    st.divider()

    # 사용 가이드
    with st.expander("📖 사용 가이드"):
        st.markdown("""
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
        **작업:** {tool_call['name']}

        **인수:**
        ```json
        {tool_call['args']}
        ```
        """)

        # 위험 경고 (shutdown인 경우)
        if tool_call['name'] == 'shutdown_mini_pc':
            st.error("⚠️ **경고**: 이 작업은 미니PC를 종료합니다. 신중하게 결정하세요!")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ 승인 (Yes)", use_container_width=True, type="primary"):
                handle_approval_response("yes")

        with col2:
            with st.popover("❌ 거부 (No)"):
                st.write("거부 사유를 입력하세요:")
                reject_msg = st.text_area(
                    "거부 사유",
                    value="지금은 이 작업을 하지 마세요.",
                    key="reject_message"
                )
                if st.button("거부 확정", key="confirm_reject"):
                    handle_approval_response("no", reject_msg)

# 채팅 입력
if not st.session_state.waiting_for_approval:
    if prompt := st.chat_input("Manager I에게 메시지를 입력하세요..."):
        # 에이전트가 초기화되지 않았으면 경고
        if st.session_state.agent is None:
            st.warning("⚠️ 먼저 사이드바에서 '에이전트 초기화' 버튼을 클릭하세요.")
            st.stop()

        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)

        # 에이전트 응답 생성
        with st.chat_message("assistant", avatar="🏠"):
            with st.spinner("생각하는 중..."):
                try:
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}

                    # 에이전트 실행
                    response = st.session_state.agent.invoke(
                        message=prompt,
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
st.caption("Built with Streamlit + LangChain + Manager I | Smart Home Control")
