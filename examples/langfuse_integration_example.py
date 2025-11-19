"""
Langfuse 통합 예제 - TeamHGraph와 함께 사용하기

이 예제는 TeamHGraph에서 Langfuse를 사용하여 그래프 전체를 추적하는 방법을 보여줍니다.
"""

import os
from pathlib import Path
import sys

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from agents.team_h_graph import TeamHGraph
from langfuse.langchain import CallbackHandler
from langfuse import Langfuse


# ============================================================================
# 방법 1: CallbackHandler를 사용한 그래프 전체 추적 (권장)
# ============================================================================

def example_1_basic_tracing():
    """
    가장 간단한 방법: CallbackHandler를 graph.invoke()에 전달

    장점:
    - 그래프 전체가 하나의 trace로 기록됨
    - Router와 모든 Manager 노드가 자동으로 하위 span으로 기록
    - 각 Manager의 tool 호출도 모두 추적됨
    """
    print("\n" + "="*80)
    print("방법 1: CallbackHandler를 사용한 기본 추적")
    print("="*80)

    # Langfuse 환경 변수 설정 (또는 .env 파일에서 로드)
    # os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."
    # os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
    # os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"

    # CallbackHandler 초기화
    langfuse_handler = CallbackHandler()

    # TeamHGraph 초기화
    team_h_graph = TeamHGraph(
        enable_manager_i=True,
        enable_manager_m=True,
        enable_manager_s=True,
        enable_manager_t=True,
        # 각 Manager의 설정은 생략...
    )

    # 그래프 실행 - callbacks에 langfuse_handler 전달
    result = team_h_graph.invoke(
        message="거실 불 켜줘",
        user_id="user123",
        thread_id="conversation-001",
        callbacks=[langfuse_handler],  # 🔑 핵심: callbacks 전달
    )

    print(f"\n결과: {result['messages'][-1].content}")
    print(f"Trace ID: {langfuse_handler.get_trace_id()}")

    # 짧은 스크립트에서는 flush 필요
    langfuse_handler.flush()


# ============================================================================
# 방법 2: Metadata와 함께 사용 (사용자 정보, 세션 ID 등)
# ============================================================================

def example_2_with_metadata():
    """
    Metadata를 추가하여 trace를 더 풍부하게 만들기

    장점:
    - user_id, session_id, tags 등으로 trace 필터링 가능
    - Langfuse UI에서 사용자별/세션별 분석 가능
    """
    print("\n" + "="*80)
    print("방법 2: Metadata와 함께 사용")
    print("="*80)

    # CallbackHandler 초기화 시 metadata 전달
    langfuse_handler = CallbackHandler(
        session_id="session-abc-123",
        user_id="user-456",
        tags=["production", "team-h-graph"],
    )

    team_h_graph = TeamHGraph(
        enable_manager_i=True,
        enable_manager_m=True,
        enable_manager_s=True,
        enable_manager_t=True,
    )

    result = team_h_graph.invoke(
        message="오늘 날씨 어때?",
        user_id="user-456",
        callbacks=[langfuse_handler],
    )

    print(f"\n결과: {result['messages'][-1].content}")
    print(f"Trace URL: https://cloud.langfuse.com/trace/{langfuse_handler.get_trace_id()}")

    langfuse_handler.flush()


# ============================================================================
# 방법 3: @observe decorator와 함께 사용 (이미 적용됨)
# ============================================================================

def example_3_with_observe_decorator():
    """
    @observe decorator는 이미 TeamHGraph.invoke()에 적용되어 있음

    장점:
    - CallbackHandler와 @observe가 함께 작동하여 더 상세한 trace 생성
    - @observe는 최상위 레벨 trace를 생성하고
    - CallbackHandler는 LangChain/LangGraph 내부 동작을 추적
    """
    print("\n" + "="*80)
    print("방법 3: @observe decorator와 함께 사용")
    print("="*80)

    langfuse_handler = CallbackHandler(
        session_id="session-xyz-789",
        user_id="user-999",
    )

    team_h_graph = TeamHGraph(
        enable_manager_i=True,
        enable_manager_m=True,
        enable_manager_s=True,
        enable_manager_t=True,
    )

    # @observe decorator가 자동으로 작동
    # team-h-graph-invoke라는 이름으로 trace 생성
    result = team_h_graph.invoke(
        message="내일 일정 알려줘",
        user_id="user-999",
        callbacks=[langfuse_handler],
    )

    print(f"\n결과: {result['messages'][-1].content}")

    langfuse_handler.flush()


# ============================================================================
# 방법 4: Stream과 함께 사용
# ============================================================================

def example_4_streaming():
    """
    스트리밍 실행에서도 동일하게 추적 가능

    장점:
    - 실시간 스트리밍 응답도 모두 기록됨
    - 각 노드별 실행 시간 측정 가능
    """
    print("\n" + "="*80)
    print("방법 4: Streaming과 함께 사용")
    print("="*80)

    langfuse_handler = CallbackHandler(
        session_id="stream-session-001",
        user_id="streaming-user",
    )

    team_h_graph = TeamHGraph(
        enable_manager_i=True,
        enable_manager_m=True,
        enable_manager_s=True,
        enable_manager_t=True,
    )

    print("\n스트리밍 시작...")
    for chunk in team_h_graph.stream(
        message="거실 불 켜줘",
        user_id="streaming-user",
        callbacks=[langfuse_handler],
    ):
        print(f"Chunk: {chunk}")

    print("\n스트리밍 완료!")
    langfuse_handler.flush()


# ============================================================================
# 방법 5: 커스텀 trace ID 사용 (고급)
# ============================================================================

def example_5_custom_trace_id():
    """
    외부 시스템과 통합할 때 유용한 커스텀 trace ID 사용

    장점:
    - 외부 요청 ID와 trace를 연결 가능
    - 분산 추적(distributed tracing) 지원
    """
    print("\n" + "="*80)
    print("방법 5: 커스텀 Trace ID 사용")
    print("="*80)

    # 외부 시스템의 요청 ID
    external_request_id = "req-12345-abcde"

    # 결정론적 trace ID 생성
    custom_trace_id = Langfuse.create_trace_id(seed=external_request_id)

    langfuse_handler = CallbackHandler(
        trace_id=custom_trace_id,
        session_id="custom-session",
        user_id="custom-user",
    )

    team_h_graph = TeamHGraph(
        enable_manager_i=True,
        enable_manager_m=True,
    )

    result = team_h_graph.invoke(
        message="불 켜줘",
        callbacks=[langfuse_handler],
    )

    print(f"\n결과: {result['messages'][-1].content}")
    print(f"커스텀 Trace ID: {custom_trace_id}")
    print(f"외부 Request ID: {external_request_id}")

    langfuse_handler.flush()


# ============================================================================
# 방법 6: Score 추가 (사용자 피드백)
# ============================================================================

def example_6_with_scoring():
    """
    실행 후 사용자 피드백이나 평가 점수 추가

    장점:
    - 응답 품질 평가 및 추적
    - A/B 테스트 결과 분석
    """
    print("\n" + "="*80)
    print("방법 6: Score 추가")
    print("="*80)

    langfuse = Langfuse()
    langfuse_handler = CallbackHandler(
        user_id="feedback-user",
    )

    team_h_graph = TeamHGraph(
        enable_manager_i=True,
        enable_manager_m=True,
    )

    result = team_h_graph.invoke(
        message="거실 불 켜줘",
        callbacks=[langfuse_handler],
    )

    print(f"\n결과: {result['messages'][-1].content}")

    # Trace ID 가져오기
    trace_id = langfuse_handler.get_trace_id()

    # 사용자 피드백 점수 추가
    langfuse.score(
        trace_id=trace_id,
        name="user-feedback",
        value=1,  # 1 (좋음) 또는 0 (나쁨)
        comment="정확하게 불을 켰습니다!",
    )

    # 자동 평가 점수 추가 (예: 응답 시간)
    langfuse.score(
        trace_id=trace_id,
        name="response-time",
        value=0.95,  # 0-1 사이 점수
        comment="빠른 응답",
    )

    print(f"Score가 추가된 Trace ID: {trace_id}")

    langfuse_handler.flush()
    langfuse.flush()


# ============================================================================
# 실행 예제
# ============================================================================

if __name__ == "__main__":
    print("Langfuse 통합 예제")
    print("="*80)

    # 환경 변수 확인
    if not os.getenv("LANGFUSE_SECRET_KEY"):
        print("⚠️  경고: LANGFUSE_SECRET_KEY가 설정되지 않았습니다.")
        print("   .env 파일에 다음 변수를 설정하세요:")
        print("   - LANGFUSE_SECRET_KEY")
        print("   - LANGFUSE_PUBLIC_KEY")
        print("   - LANGFUSE_HOST (선택적)")
        print("\n   예제는 실행되지만 Langfuse에 데이터가 전송되지 않습니다.")

    # 원하는 예제 실행
    try:
        # example_1_basic_tracing()
        # example_2_with_metadata()
        # example_3_with_observe_decorator()
        # example_4_streaming()
        # example_5_custom_trace_id()
        # example_6_with_scoring()

        print("\n" + "="*80)
        print("모든 예제가 성공적으로 완료되었습니다!")
        print("="*80)

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
