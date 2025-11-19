"""
Streamlit + PostgreSQL + Langfuse 통합 예제

이 예제는 Streamlit session_state, PostgreSQL checkpoint, Langfuse session_id를
통합하여 사용하는 방법을 보여줍니다.

실행 방법:
    streamlit run streamlit_integration_example.py
"""

import streamlit as st
import uuid
from pathlib import Path
import sys

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents import TeamHGraph


# ============================================================================
# 세션 상태 초기화
# ============================================================================

def init_session_state():
    """Streamlit 세션 상태 초기화"""

    # 1. Session ID 생성 (Streamlit 세션 = PostgreSQL thread = Langfuse session)
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        print(f"[🆕] New session created: {st.session_state.session_id}")

    # 2. User ID (로그인 시스템이 있다면 여기서 설정)
    if "user_id" not in st.session_state:
        st.session_state.user_id = "default_user"
        # 실제 환경에서는:
        # st.session_state.user_id = get_logged_in_user_id()

    # 3. 대화 히스토리 (UI 표시용)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 4. TeamHGraph 인스턴스 (세션당 하나)
    if "graph" not in st.session_state:
        st.session_state.graph = TeamHGraph(
            enable_manager_t=True,
            # PostgreSQL checkpoint 자동 활성화
            use_postgres_checkpoint=True,
        )
        print(f"[✅] TeamHGraph initialized for session {st.session_state.session_id}")


# ============================================================================
# UI 구성
# ============================================================================

def main():
    st.set_page_config(
        page_title="Team-H Chat",
        page_icon="🤖",
        layout="wide"
    )

    # 세션 초기화
    init_session_state()

    # 타이틀
    st.title("🤖 Team-H Agent Chat")

    # 사이드바: 세션 정보
    with st.sidebar:
        st.header("📊 세션 정보")
        st.info(f"""
        **Session ID**: `{st.session_state.session_id[:8]}...`

        **User ID**: `{st.session_state.user_id}`

        **대화 수**: {len(st.session_state.messages) // 2}
        """)

        st.markdown("---")

        # 새 대화 시작 버튼
        if st.button("🆕 새 대화 시작", use_container_width=True):
            # 새 session_id 생성
            old_session = st.session_state.session_id
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []

            st.success(f"새 대화를 시작했습니다!")
            print(f"[🔄] Session changed: {old_session[:8]}... → {st.session_state.session_id[:8]}...")
            st.rerun()

        st.markdown("---")

        # 통합 ID 설명
        with st.expander("ℹ️ ID 통합 구조"):
            st.markdown("""
            **단일 session_id 전략:**

            ```
            session_id
              ├─ Streamlit: 브라우저 세션 식별
              ├─ PostgreSQL: thread_id (대화 저장)
              └─ Langfuse: session_id (추적)
            ```

            **장점:**
            - 모든 시스템에서 동일한 ID 사용
            - 대화 재개 가능 (PostgreSQL)
            - 전체 세션 추적 (Langfuse)
            - 간단하고 직관적
            """)

    # 메인: 대화 히스토리
    st.header("💬 대화")

    # 대화 히스토리 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력
    if prompt := st.chat_input("메시지를 입력하세요..."):
        # 사용자 메시지 표시
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                try:
                    # TeamHGraph 실행
                    # ✨ 핵심: session_id를 모든 곳에서 공유
                    result = st.session_state.graph.invoke(
                        message=prompt,
                        user_id=st.session_state.user_id,      # Langfuse user_id
                        thread_id=st.session_state.session_id, # PostgreSQL thread_id
                        session_id=st.session_state.session_id,# Langfuse session_id (동일 값)
                    )

                    # 응답 추출
                    if result and "messages" in result and len(result["messages"]) > 0:
                        last_message = result["messages"][-1]
                        response = last_message.content
                    else:
                        response = "죄송합니다. 응답을 생성할 수 없습니다."

                    # 응답 표시
                    st.markdown(response)

                    # 히스토리에 추가
                    st.session_state.messages.append({"role": "assistant", "content": response})

                    # 성공 로그
                    print(f"[✅] Response generated for session {st.session_state.session_id[:8]}...")

                except Exception as e:
                    error_msg = f"오류가 발생했습니다: {str(e)}"
                    st.error(error_msg)
                    print(f"[❌] Error in session {st.session_state.session_id[:8]}...: {e}")

    # 하단: Langfuse 추적 링크
    if st.session_state.messages:
        st.markdown("---")
        st.caption(f"🔍 [Langfuse에서 이 세션 보기](http://192.168.0.151:3000/sessions/{st.session_state.session_id})")


# ============================================================================
# 스트리밍 버전 (선택적)
# ============================================================================

def main_streaming():
    """스트리밍 응답 버전 (더 나은 UX)"""
    st.set_page_config(
        page_title="Team-H Chat (Streaming)",
        page_icon="🤖",
        layout="wide"
    )

    init_session_state()

    st.title("🤖 Team-H Agent Chat (Streaming)")

    # 사이드바는 동일...

    # 메인
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("메시지를 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            try:
                # 스트리밍 실행
                for chunk in st.session_state.graph.stream(
                    message=prompt,
                    user_id=st.session_state.user_id,
                    thread_id=st.session_state.session_id,
                    session_id=st.session_state.session_id,
                ):
                    # 마지막 노드의 응답 추출
                    if chunk:
                        node_name = list(chunk.keys())[0]
                        node_state = chunk[node_name]

                        if "messages" in node_state and len(node_state["messages"]) > 0:
                            last_msg = node_state["messages"][-1]
                            if hasattr(last_msg, 'content'):
                                full_response = last_msg.content
                                response_placeholder.markdown(full_response)

                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"오류: {str(e)}")


# ============================================================================
# 대화 재개 예제
# ============================================================================

def example_resume_conversation():
    """
    이전 대화 재개 예제

    PostgreSQL에 저장된 대화를 불러와서 계속할 수 있습니다.
    """
    st.title("🔄 대화 재개 예제")

    # 세션 ID 입력
    previous_session_id = st.text_input("이전 세션 ID를 입력하세요:")

    if st.button("대화 재개"):
        if previous_session_id:
            # 세션 ID 변경
            st.session_state.session_id = previous_session_id

            # 그래프 재초기화 (필요시)
            st.session_state.graph = TeamHGraph(
                enable_manager_t=True,
                use_postgres_checkpoint=True,
            )

            st.success(f"세션 {previous_session_id[:8]}...로 전환되었습니다!")
            st.info("이제 이전 대화를 이어서 계속할 수 있습니다.")

            # PostgreSQL에서 대화 히스토리 불러오기 (선택적)
            # checkpoint = graph.checkpointer.get_tuple({"configurable": {"thread_id": previous_session_id}})
            # if checkpoint:
            #     st.session_state.messages = extract_messages(checkpoint)
        else:
            st.warning("세션 ID를 입력하세요.")


# ============================================================================
# 실행
# ============================================================================

if __name__ == "__main__":
    # 기본 버전 실행
    main()

    # 스트리밍 버전을 사용하려면:
    # main_streaming()
