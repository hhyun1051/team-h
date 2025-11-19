"""
Streamlit 채팅 앱 - Manager S와 대화하기

Manager S는 웹 검색 에이전트입니다.
Tavily Search API를 사용하여 실시간 정보를 검색합니다.
"""

import sys
from pathlib import Path
import streamlit as st
from typing import Dict, Any
from dotenv import load_dotenv
import os

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.manager_s import ManagerS
from langchain_core.messages import AIMessage

# .env 로드
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="Manager S Chat",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Manager S - 웹 검색 에이전트")
st.caption("Manager S와 대화하면서 웹에서 정보를 검색하세요")


# 세션 상태 초기화
def initialize_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "agent" not in st.session_state:
        st.session_state.agent = None

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "streamlit_search_thread"

    if "tavily_api_key" not in st.session_state:
        st.session_state.tavily_api_key = os.getenv("TAVILY_API_KEY", "")

    if "max_results" not in st.session_state:
        st.session_state.max_results = 5

    # 자동 에이전트 초기화
    if "agent" in st.session_state and st.session_state.agent is None:
        st.session_state.agent = create_agent()


def create_agent():
    """Manager S 에이전트 생성"""
    try:
        if not st.session_state.tavily_api_key:
            st.error("❌ Tavily API Key가 설정되지 않았습니다.")
            return None

        with st.spinner("Manager S 에이전트 초기화 중..."):
            agent = ManagerS(
                model_name="gpt-4o-mini",
                temperature=0.7,
                tavily_api_key=st.session_state.tavily_api_key,
                max_results=st.session_state.max_results,
            )
        st.success("✅ Manager S 에이전트 초기화 완료!")
        return agent
    except Exception as e:
        st.error(f"❌ 에이전트 초기화 실패: {str(e)}")
        st.info("💡 .env 파일에 TAVILY_API_KEY가 설정되어 있는지 확인하세요.")
        return None


def display_chat_message(role: str, content: str):
    """채팅 메시지 표시"""
    avatar = "🔍" if role == "assistant" else "👤"
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


# 세션 상태 초기화
initialize_session_state()

# 사이드바: 설정
with st.sidebar:
    st.header("⚙️ 설정")

    # API Key 상태 표시
    if st.session_state.tavily_api_key:
        st.success("✅ Tavily API Key 설정됨")
    else:
        st.error("❌ Tavily API Key가 설정되지 않았습니다")
        st.caption("💡 .env 파일에 TAVILY_API_KEY를 설정하세요")

    st.divider()

    # 검색 결과 최대 개수 설정
    max_results = st.slider(
        "검색 결과 최대 개수",
        min_value=1,
        max_value=10,
        value=st.session_state.max_results,
        help="검색 시 반환할 최대 결과 개수"
    )
    if max_results != st.session_state.max_results:
        st.session_state.max_results = max_results

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
        st.rerun()

    st.divider()

    # 정보 표시
    api_status = "✅ 설정됨" if st.session_state.tavily_api_key else "❌ 미설정"
    st.info(f"""
    **현재 설정:**
    - Tavily API: {api_status}
    - Thread ID: `{st.session_state.thread_id}`
    - 메시지 수: {len(st.session_state.messages)}
    - 최대 결과 수: {st.session_state.max_results}
    """)

    st.divider()

    # 사용 가이드
    with st.expander("📖 사용 가이드"):
        st.markdown("""
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
        """)

# 메인 채팅 영역
st.divider()

# 채팅 히스토리 표시
for message in st.session_state.messages:
    display_chat_message(message["role"], message["content"])

# 채팅 입력
if prompt := st.chat_input("Manager S에게 메시지를 입력하세요..."):
    # 에이전트가 초기화되지 않았으면 경고
    if st.session_state.agent is None:
        st.warning("⚠️ 먼저 사이드바에서 '에이전트 초기화' 버튼을 클릭하세요.")
        st.stop()

    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    display_chat_message("user", prompt)

    # 에이전트 응답 생성
    with st.chat_message("assistant", avatar="🔍"):
        with st.spinner("검색 중..."):
            try:
                # 에이전트 실행
                response = st.session_state.agent.invoke(
                    message=prompt,
                    thread_id=st.session_state.thread_id,
                )

                # 응답 처리
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
st.caption("Built with Streamlit + LangChain + Manager S | Web Search Powered by Tavily")
