"""
Manager I 통합 테스트 (End-to-End Scenarios)

실제 사용 시나리오를 전체적으로 테스트합니다.
"""

import sys
from pathlib import Path
import pytest
import time

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.manager_i import ManagerIAgent
from langchain_core.messages import AIMessage


@pytest.fixture
def manager_i(smartthings_token, device_config, openai_api_key):
    """Manager I Agent 인스턴스"""
    agent = ManagerIAgent(
        model_name="gpt-4o-mini",
        temperature=0.0,
        smartthings_token=smartthings_token,
        device_config=device_config,
    )
    return agent


@pytest.mark.integration
@pytest.mark.safe
@pytest.mark.slow
def test_scenario_bedtime_routine(manager_i):
    """
    시나리오: 잠들기 전 루틴

    1. 모든 불 상태 확인
    2. 거실 불 끄기
    3. 안방 불 끄기
    4. 스피커 끄기
    """
    thread_id = "scenario_bedtime"

    print("\n=== 잠들기 전 루틴 시나리오 ===")

    steps = [
        "거실 불 상태 확인해줘",
        "안방 불 상태 확인해줘",
        "거실 불 꺼줘",
        "안방 불 꺼줘",
        "거실 스피커 꺼줘",
    ]

    for i, step in enumerate(steps, 1):
        print(f"\n{i}. User: {step}")

        result = manager_i.invoke(message=step, thread_id=thread_id)

        # AI 응답 출력
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                print(f"   Assistant: {msg.content}")
                assert len(msg.content) > 0
                break

        # 다음 명령 전에 잠시 대기
        if i < len(steps):
            time.sleep(1)

    print("\n✅ 잠들기 전 루틴 완료")


@pytest.mark.integration
@pytest.mark.safe
def test_scenario_morning_routine(manager_i):
    """
    시나리오: 아침 루틴

    1. 안방 불 켜기
    2. 화장실 불 켜기
    3. 거실 불 켜기
    """
    thread_id = "scenario_morning"

    print("\n=== 아침 루틴 시나리오 ===")

    steps = [
        "안방 불 켜줘",
        "화장실 불 켜줘",
        "거실 불 켜줘",
    ]

    for i, step in enumerate(steps, 1):
        print(f"\n{i}. User: {step}")

        result = manager_i.invoke(message=step, thread_id=thread_id)

        # AI 응답
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                print(f"   Assistant: {msg.content}")
                break

        time.sleep(1)

    print("\n✅ 아침 루틴 완료")


@pytest.mark.integration
@pytest.mark.safe
def test_scenario_status_check_all(manager_i):
    """
    시나리오: 모든 장치 상태 확인

    모든 방의 불 상태를 차례대로 확인
    """
    thread_id = "scenario_status_all"

    print("\n=== 모든 장치 상태 확인 시나리오 ===")

    rooms = [
        ("거실", "living_room"),
        ("안방", "bedroom"),
        ("화장실", "bathroom"),
    ]

    for i, (room_kr, room_en) in enumerate(rooms, 1):
        message = f"{room_kr} 불 상태 알려줘"
        print(f"\n{i}. User: {message}")

        result = manager_i.invoke(message=message, thread_id=thread_id)

        # AI 응답
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                print(f"   Assistant: {msg.content}")
                break

        time.sleep(0.5)

    print("\n✅ 모든 장치 상태 확인 완료")


@pytest.mark.integration
@pytest.mark.safe
def test_scenario_conversational(manager_i):
    """
    시나리오: 대화형 제어

    자연스러운 대화를 통한 IoT 제어
    """
    thread_id = "scenario_conversation"

    print("\n=== 대화형 제어 시나리오 ===")

    conversation = [
        "안녕 Manager I",
        "지금 거실 불 켜져있어?",
        "그럼 꺼줄래?",
        "고마워!",
    ]

    for i, message in enumerate(conversation, 1):
        print(f"\n{i}. User: {message}")

        result = manager_i.invoke(message=message, thread_id=thread_id)

        # AI 응답
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                print(f"   Assistant: {msg.content}")
                break

        time.sleep(1)

    print("\n✅ 대화형 제어 완료")


@pytest.mark.integration
@pytest.mark.safe
def test_scenario_multiple_actions(manager_i):
    """
    시나리오: 한 번에 여러 동작 요청

    "모든 불 꺼줘" 같은 복합 명령 처리
    """
    thread_id = "scenario_multiple"

    print("\n=== 복합 명령 시나리오 ===")

    # 단일 메시지로 여러 동작 요청
    message = "거실하고 안방 불 둘 다 꺼줘"

    print(f"\nUser: {message}")

    result = manager_i.invoke(message=message, thread_id=thread_id)

    # AI 응답
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            print(f"Assistant: {msg.content}")
            break

    print("\n✅ 복합 명령 완료")


@pytest.mark.integration
def test_scenario_error_recovery(manager_i):
    """
    시나리오: 에러 복구

    잘못된 명령 후 올바른 명령으로 복구
    """
    thread_id = "scenario_error_recovery"

    print("\n=== 에러 복구 시나리오 ===")

    steps = [
        "주방 불 켜줘",  # 에러: 주방은 설정 안 됨
        "아 미안, 거실 불 켜줘",  # 정상 명령
    ]

    for i, step in enumerate(steps, 1):
        print(f"\n{i}. User: {step}")

        result = manager_i.invoke(message=step, thread_id=thread_id)

        # AI 응답
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                print(f"   Assistant: {msg.content}")
                assert len(msg.content) > 0
                break

        time.sleep(1)

    print("\n✅ 에러 복구 완료")


@pytest.mark.integration
@pytest.mark.safe
def test_scenario_context_aware(manager_i):
    """
    시나리오: 컨텍스트 인식

    이전 대화를 기억하고 대응하는지 테스트
    """
    thread_id = "scenario_context"

    print("\n=== 컨텍스트 인식 시나리오 ===")

    conversation = [
        "거실 불 상태 확인해줘",
        "켜져있으면 꺼줘",  # 이전 대화 컨텍스트 사용
        "스피커도 꺼줄까?",
        "응, 꺼줘",  # 이전 질문에 대한 답변
    ]

    for i, message in enumerate(conversation, 1):
        print(f"\n{i}. User: {message}")

        result = manager_i.invoke(message=message, thread_id=thread_id)

        # AI 응답
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                print(f"   Assistant: {msg.content}")
                break

        time.sleep(1)

    print("\n✅ 컨텍스트 인식 테스트 완료")


@pytest.mark.integration
@pytest.mark.safe
@pytest.mark.slow
def test_scenario_full_cycle(manager_i):
    """
    시나리오: 전체 사이클 테스트

    장치를 켜고 -> 상태 확인 -> 끄고 -> 다시 상태 확인
    """
    thread_id = "scenario_full_cycle"
    room = "안방"  # 세로모니터 (안전한 장치)

    print(f"\n=== {room} 전체 사이클 테스트 ===")

    steps = [
        f"{room} 불 켜줘",
        f"{room} 불 상태 확인해줘",
        f"{room} 불 꺼줘",
        f"{room} 불 다시 상태 확인해줘",
    ]

    for i, step in enumerate(steps, 1):
        print(f"\n{i}. User: {step}")

        result = manager_i.invoke(message=step, thread_id=thread_id)

        # AI 응답
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                print(f"   Assistant: {msg.content}")
                break

        # 장치가 상태를 변경할 시간 제공
        time.sleep(2)

    print(f"\n✅ {room} 전체 사이클 완료")


@pytest.mark.integration
def test_scenario_multilingual(manager_i):
    """
    시나리오: 다양한 표현 방식

    같은 의미를 다른 방식으로 표현
    """
    thread_id_base = "scenario_multilingual"

    print("\n=== 다양한 표현 방식 테스트 ===")

    variations = [
        "거실 불 꺼줘",
        "거실 조명 끄기",
        "living room light off",
        "거실 전등 소등해줘",
    ]

    for i, message in enumerate(variations, 1):
        thread_id = f"{thread_id_base}_{i}"
        print(f"\n{i}. User: {message}")

        result = manager_i.invoke(message=message, thread_id=thread_id)

        # AI 응답
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                print(f"   Assistant: {msg.content}")
                break

        time.sleep(0.5)

    print("\n✅ 다양한 표현 방식 테스트 완료")


@pytest.mark.integration
@pytest.mark.safe
def test_scenario_batch_status_check(manager_i):
    """
    시나리오: 일괄 상태 확인

    여러 장치의 상태를 한 번에 확인
    """
    thread_id = "scenario_batch_status"

    message = "모든 방 불 상태 알려줘"

    print("\n=== 일괄 상태 확인 시나리오 ===")
    print(f"User: {message}")

    result = manager_i.invoke(message=message, thread_id=thread_id)

    # AI 응답
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            print(f"Assistant: {msg.content}")
            assert len(msg.content) > 0
            break

    print("\n✅ 일괄 상태 확인 완료")


if __name__ == "__main__":
    # 개별 실행용
    pytest.main([__file__, "-v", "-s"])