"""
개선된 HITL 승인 UI - 개별 결정 수집 후 일괄 제출 방식

사용자가 각 작업에 대해 독립적으로 결정을 내리고,
모든 결정을 확인한 후 "최종 제출" 버튼으로 한 번에 전송합니다.

FastAPI 클라이언트를 통해 HITL resume 요청을 처리합니다.
"""

import streamlit as st
import json
from typing import Dict, List, Any, Optional


def initialize_approval_decisions(num_actions: int):
    """승인 결정 상태 초기화"""
    if "approval_decisions" not in st.session_state:
        st.session_state.approval_decisions = {}

    # 각 액션에 대한 기본 결정 (미결정 상태)
    for idx in range(num_actions):
        if idx not in st.session_state.approval_decisions:
            st.session_state.approval_decisions[idx] = {
                "type": None,  # None, "approve", "reject", "edit"
                "edited_args": None,
                "edited_tool_name": None,
                "reject_message": None,
            }


def fetch_memory_content(memory_id: str) -> Optional[str]:
    """
    메모리 ID로 실제 메모리 내용을 가져옴

    Args:
        memory_id: 조회할 메모리 ID

    Returns:
        메모리 내용 문자열, 실패 시 None
    """
    try:
        # 디버깅: 시작
        print(f"[DEBUG] fetch_memory_content called with memory_id: {memory_id}")

        # TeamHGraph의 manager_m 인스턴스에 접근
        if hasattr(st.session_state, 'agent'):
            print(f"[DEBUG] st.session_state.agent exists")
            agent = st.session_state.agent

            if hasattr(agent, 'manager_m'):
                print(f"[DEBUG] agent.manager_m exists: {agent.manager_m}")

                if agent.manager_m is not None:
                    print(f"[DEBUG] agent.manager_m is not None")

                    # get_memory_by_id 메서드 사용
                    memory = agent.manager_m.memory.get_memory_by_id(memory_id)
                    print(f"[DEBUG] Retrieved memory: {memory}")

                    if memory:
                        content = memory.get('content', 'No content')
                        memory_type = memory.get('type', 'unknown')
                        result = f"[{memory_type}] {content}"
                        print(f"[DEBUG] Returning: {result}")
                        return result
                    else:
                        print(f"[DEBUG] memory is None")
                else:
                    print(f"[DEBUG] agent.manager_m is None")
            else:
                print(f"[DEBUG] agent.manager_m does not exist")
        else:
            print(f"[DEBUG] st.session_state.agent does not exist")

    except Exception as e:
        import traceback
        print(f"[DEBUG] Exception occurred: {str(e)}")
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return f"⚠️ 메모리 조회 실패: {str(e)}"

    print(f"[DEBUG] Returning None")
    return None


def render_action_card(
    idx: int,
    action: Dict[str, Any],
    review: Dict[str, Any],
    total_actions: int,
    is_single_action: bool = False
) -> None:
    """개별 작업 카드 렌더링"""

    # 현재 결정 상태
    current_decision = st.session_state.approval_decisions.get(idx, {})
    decision_type = current_decision.get("type")

    # 카드 헤더 - 결정 상태에 따라 색상 변경
    if decision_type == "approve":
        status_emoji = "✅"
        status_text = "승인됨"
        status_color = "green"
    elif decision_type == "reject":
        status_emoji = "❌"
        status_text = "거부됨"
        status_color = "red"
    elif decision_type == "edit":
        status_emoji = "✏️"
        status_text = "편집됨"
        status_color = "orange"
    else:
        status_emoji = "⏳"
        status_text = "대기 중"
        status_color = "gray"

    tool_name = action.get('name', 'Unknown')

    with st.expander(
        f"{status_emoji} 작업 {idx + 1}/{total_actions}: {tool_name} - [{status_text}]",
        expanded=(decision_type is None)
    ):
        # 디버깅: tool_name 확인
        st.caption(f"🔍 Tool name debug: `{tool_name}` (type: {type(tool_name)})")

        # 작업 설명
        st.markdown(f"**📝 설명:**")
        st.info(action.get('description', 'N/A'))

        # delete_memory인 경우 실제 메모리 내용 표시
        if tool_name == 'delete_memory':
            # arguments 또는 args 키 모두 지원
            args = action.get('arguments') or action.get('args', {})
            memory_id = args.get('memory_id')

            if memory_id:
                st.markdown("**🗑️ 삭제할 메모리 내용:**")

                # 디버깅: UI에 표시
                with st.expander("🐛 디버그 정보", expanded=False):
                    st.write(f"memory_id: `{memory_id}`")
                    st.write(f"session_state.agent exists: {hasattr(st.session_state, 'agent')}")
                    if hasattr(st.session_state, 'agent'):
                        agent = st.session_state.agent
                        st.write(f"agent type: {type(agent)}")
                        st.write(f"agent.manager_m exists: {hasattr(agent, 'manager_m')}")
                        if hasattr(agent, 'manager_m'):
                            st.write(f"agent.manager_m: {agent.manager_m}")
                            st.write(f"agent.manager_m is None: {agent.manager_m is None}")

                memory_content = fetch_memory_content(memory_id)

                if memory_content:
                    st.warning(f"**{memory_content}**")
                else:
                    st.error(f"메모리 ID `{memory_id}`의 내용을 불러올 수 없습니다")

        # Arguments 표시
        st.markdown(f"**🔧 Arguments:**")
        with st.expander("상세 정보", expanded=False):
            # arguments 또는 args 키 모두 지원
            args_to_show = action.get('arguments') or action.get('args', {})
            st.json(args_to_show)

        # 허용된 결정 타입
        allowed = review.get("allowed_decisions", ["approve", "reject"])
        st.caption(f"허용된 결정: {', '.join(allowed)}")

        st.divider()

        # 편집 모드
        edit_mode_key = f"edit_mode_{idx}"
        if edit_mode_key not in st.session_state:
            st.session_state[edit_mode_key] = False

        if st.session_state[edit_mode_key]:
            render_edit_mode(idx, action, allowed, is_single_action)
        else:
            render_decision_buttons(idx, action, allowed, total_actions, is_single_action)


def render_edit_mode(idx: int, action: Dict[str, Any], allowed: List[str], is_single_action: bool = False):
    """편집 모드 UI"""
    st.info("✏️ 편집 모드: 아래에서 tool arguments를 수정하세요")

    original_args = action.get('arguments', {})
    tool_name = action.get('name', '')

    st.markdown(f"**Tool Name:** `{tool_name}`")

    # JSON 형태로 arguments 편집
    args_json = json.dumps(original_args, indent=2, ensure_ascii=False)
    edited_args_json = st.text_area(
        "Arguments (JSON 형식):",
        value=args_json,
        height=200,
        key=f"edit_args_input_{idx}"
    )

    col1, col2 = st.columns(2)

    # 편집 적용 버튼
    if col1.button("✅ 편집 적용", key=f"apply_edit_{idx}", use_container_width=True):
        try:
            edited_args = json.loads(edited_args_json)

            if is_single_action:
                # 단일 작업: 즉시 실행
                approval_data = st.session_state.pending_approval
                execute_single_decision("edit", action, approval_data,
                                      edited_args=edited_args,
                                      edited_tool_name=tool_name)
            else:
                # 다중 작업: 결정 저장
                st.session_state.approval_decisions[idx] = {
                    "type": "edit",
                    "edited_args": edited_args,
                    "edited_tool_name": tool_name,
                    "reject_message": None,
                }

                st.session_state[f"edit_mode_{idx}"] = False
                st.success(f"✅ 작업 {idx + 1} 편집 적용됨")
                st.rerun()
        except json.JSONDecodeError as e:
            st.error(f"❌ JSON 파싱 오류: {e}")

    # 편집 취소 버튼
    if col2.button("↩️ 취소", key=f"cancel_edit_{idx}", use_container_width=True):
        st.session_state[f"edit_mode_{idx}"] = False
        st.rerun()


def render_decision_buttons(idx: int, action: Dict[str, Any], allowed: List[str], total_actions: int = 1, is_single_action: bool = False):
    """결정 버튼 UI

    Args:
        idx: 작업 인덱스
        action: 작업 정보
        allowed: 허용된 결정 타입
        total_actions: 전체 작업 개수
        is_single_action: 단일 작업 여부 (True일 경우 즉시 실행)
    """

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
        if target_col.button("✅ 승인", key=f"btn_approve_{idx}", use_container_width=True):
            if is_single_action:
                # 단일 작업: 즉시 실행
                approval_data = st.session_state.pending_approval
                execute_single_decision("approve", action, approval_data)
            else:
                # 다중 작업: 결정만 저장
                st.session_state.approval_decisions[idx] = {
                    "type": "approve",
                    "edited_args": None,
                    "edited_tool_name": None,
                    "reject_message": None,
                }
                st.success(f"✅ 작업 {idx + 1} 승인됨")
                st.rerun()

    # 편집 버튼
    if "edit" in allowed:
        target_col = col2 if num_buttons >= 2 else col1 if num_buttons >= 1 else st
        if target_col.button("✏️ 편집", key=f"btn_edit_{idx}", use_container_width=True):
            st.session_state[f"edit_mode_{idx}"] = True
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

        if target_col.button("❌ 거부", key=f"btn_reject_{idx}", use_container_width=True):
            if is_single_action:
                # 단일 작업: 즉시 실행
                approval_data = st.session_state.pending_approval
                execute_single_decision("reject", action, approval_data, reject_message="사용자가 거부했습니다")
            else:
                # 다중 작업: 결정만 저장
                # 거부 사유 입력 (선택적)
                reject_reason = st.text_input(
                    "거부 사유 (선택):",
                    key=f"reject_reason_input_{idx}",
                    placeholder="거부 사유를 입력하세요..."
                )

                st.session_state.approval_decisions[idx] = {
                    "type": "reject",
                    "edited_args": None,
                    "edited_tool_name": None,
                    "reject_message": reject_reason or "사용자가 거부했습니다",
                }
                st.error(f"❌ 작업 {idx + 1} 거부됨")
                st.rerun()


def execute_single_decision(decision_type: str, action: Dict[str, Any], approval_data: Dict[str, Any], edited_args: Optional[Dict] = None, edited_tool_name: Optional[str] = None, reject_message: Optional[str] = None):
    """단일 작업 즉시 실행 (FastAPI 클라이언트 사용)"""
    try:
        # 단일 결정 페이로드 생성
        if decision_type == "approve":
            decisions = [{"type": "approve"}]
        elif decision_type == "edit":
            decisions = [{
                "type": "edit",
                "edited_action": {
                    "name": edited_tool_name,
                    "args": edited_args
                }
            }]
        elif decision_type == "reject":
            decisions = [{
                "type": "reject",
                "message": reject_message or "사용자가 거부했습니다"
            }]
        else:
            st.error(f"❌ 알 수 없는 결정 타입: {decision_type}")
            return

        # FastAPI 클라이언트로 resume 요청
        api_client = st.session_state.api_client
        thread_id = approval_data["thread_id"]
        user_id = approval_data.get("user_id", "default_user")
        session_id = approval_data.get("session_id")

        # SSE 스트림 표시
        with st.spinner("⏳ 작업 실행 중..."):
            full_response = ""
            agent_name = "Manager M"
            message_placeholder = st.empty()

            # SSE 스트림 처리
            for event in api_client.resume_stream(
                thread_id=thread_id,
                decisions=decisions,
                user_id=user_id,
                session_id=session_id,
            ):
                event_type = event.get("event")

                if event_type == "token":
                    # 실시간 토큰 스트리밍
                    full_response += event.get("content", "")
                    message_placeholder.markdown(full_response + "▌")

                elif event_type == "llm_end":
                    # LLM 완료 시 전체 메시지
                    full_msg = event.get("full_message", "")
                    if full_msg:
                        full_response = full_msg

                elif event_type == "done":
                    # 정상 완료
                    break

                elif event_type == "error":
                    # 오류 발생
                    st.error(f"❌ 서버 오류: {event.get('error')}")
                    return

            # 메시지 저장
            if full_response:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "agent_name": agent_name,
                })

            # 상태 정리
            st.session_state.pending_approval = None
            st.session_state.approval_decisions = {}

            st.success("✅ 작업 실행 완료!")
            st.rerun()

    except Exception as e:
        import traceback
        st.error(f"❌ 실행 중 오류: {e}")
        st.code(traceback.format_exc())


def build_decisions_payload(action_requests: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """
    세션 상태에서 decisions 페이로드 생성

    Returns:
        decisions 리스트, 또는 미결정 작업이 있으면 None
    """
    decisions = []

    for idx in range(len(action_requests)):
        decision = st.session_state.approval_decisions.get(idx, {})
        decision_type = decision.get("type")

        if decision_type == "approve":
            decisions.append({"type": "approve"})

        elif decision_type == "reject":
            decisions.append({
                "type": "reject",
                "message": decision.get("reject_message", "사용자가 거부했습니다")
            })

        elif decision_type == "edit":
            decisions.append({
                "type": "edit",
                "edited_action": {
                    "name": decision.get("edited_tool_name"),
                    "args": decision.get("edited_args")
                }
            })

        else:
            # 미결정 상태 - None 반환하여 제출 불가
            return None

    return decisions


def render_approval_summary(action_requests: List[Dict[str, Any]]):
    """승인 요약 및 최종 제출 버튼"""
    st.divider()

    # 통계 계산
    total = len(action_requests)
    approved = sum(1 for d in st.session_state.approval_decisions.values() if d.get("type") == "approve")
    rejected = sum(1 for d in st.session_state.approval_decisions.values() if d.get("type") == "reject")
    edited = sum(1 for d in st.session_state.approval_decisions.values() if d.get("type") == "edit")
    pending = sum(1 for d in st.session_state.approval_decisions.values() if d.get("type") is None)

    # 요약 표시
    st.markdown("### 📊 승인 요약")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("✅ 승인", approved)
    col2.metric("❌ 거부", rejected)
    col3.metric("✏️ 편집", edited)
    col4.metric("⏳ 대기", pending)

    # 경고 메시지
    if pending > 0:
        st.warning(f"⚠️ {pending}개의 작업이 아직 결정되지 않았습니다. 모든 작업에 대한 결정이 필요합니다.")

    st.divider()

    # 최종 제출 버튼
    col_submit, col_cancel = st.columns(2)

    # 미결정 작업이 있으면 비활성화
    submit_disabled = (pending > 0)

    if col_submit.button(
        "🚀 최종 제출",
        key="final_submit",
        use_container_width=True,
        type="primary",
        disabled=submit_disabled,
        help="모든 작업에 대한 결정이 필요합니다" if submit_disabled else "모든 결정을 제출합니다"
    ):
        return True  # 제출 트리거

    if col_cancel.button(
        "🔄 모두 초기화",
        key="reset_all",
        use_container_width=True,
    ):
        # 모든 결정 초기화
        st.session_state.approval_decisions = {}
        st.rerun()

    return False


def render_approval_ui_refactored():
    """
    개선된 HITL 승인 UI (메인 함수)

    Returns:
        bool: 승인 처리 완료 여부
    """
    if not st.session_state.pending_approval:
        return False

    approval_data = st.session_state.pending_approval
    interrupt = approval_data["interrupt"]

    st.divider()
    st.warning("⏸️ 승인이 필요한 작업이 있습니다", icon="✋")

    # 디버그 정보 (선택적)
    with st.expander("🐛 디버그: 전체 구조", expanded=False):
        st.code(f"Type: {type(interrupt).__name__}")
        try:
            st.code(json.dumps(interrupt, indent=2, default=str))
        except:
            st.text(str(interrupt))

    # action_requests 추출
    try:
        # FastAPI에서 받은 interrupt는 이미 dictionary 형태
        action_requests = interrupt.get("action_requests", [])
        review_configs = interrupt.get("review_configs", [])

        if not action_requests:
            st.error("❌ action_requests가 비어있습니다")
            st.session_state.pending_approval = None
            return False

        # 결정 상태 초기화
        initialize_approval_decisions(len(action_requests))

        # 단일 작업 여부 확인
        is_single_action = len(action_requests) == 1

        # 각 작업 카드 렌더링
        for idx, (action, review) in enumerate(zip(action_requests, review_configs)):
            render_action_card(idx, action, review, len(action_requests), is_single_action)

        # 다중 작업인 경우에만 요약 및 최종 제출 버튼 표시
        if not is_single_action and render_approval_summary(action_requests):
            # 최종 제출 처리 (FastAPI 클라이언트 사용)
            try:
                decisions = build_decisions_payload(action_requests)

                # 미결정 작업이 있으면 제출 불가
                if decisions is None:
                    st.error("❌ 모든 작업에 대한 결정이 필요합니다.")
                    return True

                # FastAPI 클라이언트로 resume 요청
                api_client = st.session_state.api_client
                thread_id = approval_data["thread_id"]
                user_id = approval_data.get("user_id", "default_user")
                session_id = approval_data.get("session_id")

                # SSE 스트림 표시
                with st.spinner("⏳ 모든 작업 실행 중..."):
                    full_response = ""
                    agent_name = "Manager M"
                    message_placeholder = st.empty()

                    # SSE 스트림 처리
                    for event in api_client.resume_stream(
                        thread_id=thread_id,
                        decisions=decisions,
                        user_id=user_id,
                        session_id=session_id,
                    ):
                        event_type = event.get("event")

                        if event_type == "token":
                            # 실시간 토큰 스트리밍
                            full_response += event.get("content", "")
                            message_placeholder.markdown(full_response + "▌")

                        elif event_type == "llm_end":
                            # LLM 완료 시 전체 메시지
                            full_msg = event.get("full_message", "")
                            if full_msg:
                                full_response = full_msg

                        elif event_type == "done":
                            # 정상 완료
                            break

                        elif event_type == "error":
                            # 오류 발생
                            st.error(f"❌ 서버 오류: {event.get('error')}")
                            return True

                    # 메시지 저장
                    if full_response:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_response,
                            "agent_name": agent_name,
                        })

                    # 상태 정리
                    st.session_state.pending_approval = None
                    st.session_state.approval_decisions = {}

                    st.success("✅ 승인 처리 완료!")
                    st.rerun()

            except Exception as e:
                st.error(f"❌ 승인 처리 중 오류: {e}")
                import traceback
                with st.expander("오류 상세", expanded=True):
                    st.code(traceback.format_exc())

        return True

    except Exception as e:
        st.error(f"❌ UI 렌더링 중 오류: {e}")
        import traceback
        st.code(traceback.format_exc())
        return False
