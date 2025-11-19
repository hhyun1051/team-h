"""
Manager I Agent 테스트

Agent 전체 동작 및 Tools 테스트를 포함합니다.
- Agent 초기화 및 기본 동작
- 개별 Tools 테스트
- 자연어 명령 처리
- 실제 장치 제어
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import pytest
import time

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 프로젝트 루트의 .env 로드
load_dotenv(project_root / ".env")

from agents.manager_i import ManagerIAgent
from langchain_core.messages import AIMessage


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def smartthings_token():
    """SmartThings API 토큰"""
    return os.getenv("SMARTTHINGS_TOKEN")


@pytest.fixture
def device_config():
    """테스트용 장치 설정"""
    return {
        "living_room_speaker_outlet": "d5ae3413-10a4-4a03-b5e3-eaa0bee64db4",  # 스피커
        "bedroom_light": "55ca4824-3237-411b-88fd-efb549927553",  # 세로모니터
        "living_room_light": "f28bb22f-4768-685b-076b-b9514941498c",  # 프로젝터
        "bathroom_light": "0897d30e-5cb2-5566-13d5-7de7394061d1",  # 공기청정기
    }


@pytest.fixture
def agent(smartthings_token, device_config):
    """Manager I Agent 인스턴스"""
    return ManagerIAgent(
        model_name="gpt-4o-mini",
        temperature=0.0,
        smartthings_token=smartthings_token,
        device_config=device_config,
    )


# ============================================================================
# Agent 초기화 및 기본 동작 테스트
# ============================================================================

@pytest.mark.integration
def test_agent_initialization(smartthings_token, device_config):
    """Agent 초기화 테스트"""
    agent = ManagerIAgent(
        model_name="gpt-4o-mini",
        temperature=0.0,
        smartthings_token=smartthings_token,
        device_config=device_config,
    )

    assert agent is not None
    assert agent.smartthings_token == smartthings_token
    assert len(agent.tools) == 5
    assert agent.agent is not None

    print(f"\n✅ Agent initialized:")
    print(f"   - Model: {agent.model_name}")
    print(f"   - Tools: {len(agent.tools)}")
    print(f"   - Devices: {len(agent.device_config)}")


@pytest.mark.integration
def test_tools_exist(agent):
    """모든 Tools가 생성되었는지 확인"""
    tool_names = [tool.name for tool in agent.tools]
    expected_tools = [
        "shutdown_mini_pc",
        "turn_on_light",
        "turn_off_light",
        "turn_off_speaker",
        "get_device_status",
    ]

    for expected_tool in expected_tools:
        assert expected_tool in tool_names

    print(f"\n✅ All {len(expected_tools)} tools found:")
    for tool in tool_names:
        print(f"   - {tool}")


# ============================================================================
# 개별 Tools 테스트
# ============================================================================

@pytest.mark.integration
@pytest.mark.safe
def test_tool_turn_off_speaker(agent):
    """스피커 끄기 Tool 테스트"""
    turn_off_speaker_tool = next(
        (tool for tool in agent.tools if tool.name == "turn_off_speaker"), None
    )
    assert turn_off_speaker_tool is not None

    result = turn_off_speaker_tool.invoke({})
    print(f"\n🔧 Tool result: {result}")

    assert isinstance(result, str)
    assert "✅" in result or "거실 스피커" in result


@pytest.mark.integration
@pytest.mark.safe
def test_tool_turn_off_light(agent):
    """불 끄기 Tool 테스트"""
    turn_off_light_tool = next(
        (tool for tool in agent.tools if tool.name == "turn_off_light"), None
    )
    assert turn_off_light_tool is not None

    result = turn_off_light_tool.invoke({"room": "bedroom"})
    print(f"\n🔧 Tool result: {result}")

    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.integration
def test_tool_invalid_room(agent):
    """유효하지 않은 방 이름 테스트"""
    turn_on_light_tool = next(
        (tool for tool in agent.tools if tool.name == "turn_on_light"), None
    )

    result = turn_on_light_tool.invoke({"room": "invalid_room"})
    print(f"\n🔧 Tool result: {result}")

    assert "❌" in result
    assert "Unknown room" in result or "Available" in result


@pytest.mark.integration
@pytest.mark.safe
def test_tool_sequence(agent):
    """여러 Tool을 순차적으로 실행"""
    print("\n=== Tool 순차 실행 테스트 ===")

    # 1. 불 끄기
    turn_off_tool = next(
        (tool for tool in agent.tools if tool.name == "turn_off_light"), None
    )
    result1 = turn_off_tool.invoke({"room": "bedroom"})
    print(f"1. 끄기: {result1}")
    time.sleep(1)

    # 2. 상태 확인
    status_tool = next(
        (tool for tool in agent.tools if tool.name == "get_device_status"), None
    )
    result2 = status_tool.invoke({"room": "bedroom", "device_type": "light"})
    print(f"2. 상태: {result2}")
    time.sleep(1)

    # 3. 불 켜기
    turn_on_tool = next(
        (tool for tool in agent.tools if tool.name == "turn_on_light"), None
    )
    result3 = turn_on_tool.invoke({"room": "bedroom"})
    print(f"3. 켜기: {result3}")

    assert all(isinstance(r, str) for r in [result1, result2, result3])


# ============================================================================
# Agent 자연어 명령 처리 테스트
# ============================================================================

@pytest.mark.integration
@pytest.mark.safe
def test_agent_simple_command(agent):
    """간단한 자연어 명령 처리"""
    message = "거실 스피커 꺼줘"

    result = agent.invoke(message=message, thread_id="test_thread_1")

    assert result is not None
    assert "messages" in result

    print(f"\n사용자: {message}")
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            print(f"Agent: {msg.content}")
            assert len(msg.content) > 0
            break


@pytest.mark.integration
@pytest.mark.safe
def test_agent_multiple_commands(agent):
    """여러 명령을 순차적으로 실행"""
    commands = [
        "거실 스피커 꺼줘",
        "안방 불 꺼줘",
    ]

    print("\n=== 순차 명령 테스트 ===")

    for i, command in enumerate(commands):
        thread_id = f"test_multi_{i}"
        result = agent.invoke(message=command, thread_id=thread_id)

        print(f"\n{i+1}. 사용자: {command}")

        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                print(f"   Agent: {msg.content}")
                break

        time.sleep(1)


@pytest.mark.integration
def test_agent_unknown_command(agent):
    """IoT와 관련 없는 명령"""
    message = "내일 날씨 알려줘"

    result = agent.invoke(message=message, thread_id="test_unknown")

    assert result is not None

    print(f"\n사용자: {message}")
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            print(f"Agent: {msg.content}")
            assert len(msg.content) > 0
            break


@pytest.mark.integration
def test_agent_error_handling(agent):
    """Agent 에러 처리 테스트"""
    message = "주방 불 켜줘"  # 주방은 설정에 없음

    result = agent.invoke(message=message, thread_id="test_error")

    assert result is not None

    print(f"\n사용자: {message}")
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            print(f"Agent: {msg.content}")
            break


# ============================================================================
# Agent 실제 사용 테스트
# ============================================================================

@pytest.mark.integration
@pytest.mark.safe
def test_agent_conversation(agent):
    """대화형 제어 테스트"""
    thread_id = "test_conversation"

    conversation = [
        "안녕, Manager I",
        "거실 스피커 꺼줘",
        "고마워!",
    ]

    print("\n=== 대화형 제어 테스트 ===")

    for i, message in enumerate(conversation, 1):
        print(f"\n{i}. 사용자: {message}")

        result = agent.invoke(message=message, thread_id=thread_id)

        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                print(f"   Agent: {msg.content}")
                break

        time.sleep(1)


@pytest.mark.integration
@pytest.mark.safe
def test_agent_final():
    """최종 통합 테스트 - Agent 전체 동작 확인"""
    smartthings_token = os.getenv("SMARTTHINGS_TOKEN")
    if not smartthings_token:
        pytest.skip("SMARTTHINGS_TOKEN 환경변수가 설정되지 않았습니다.")

    device_config = {
        "living_room_speaker_outlet": "d5ae3413-10a4-4a03-b5e3-eaa0bee64db4",
        "bedroom_light": "55ca4824-3237-411b-88fd-efb549927553",
    }

    print("\n=== Manager I Agent 최종 테스트 ===")

    # Agent 생성
    agent = ManagerIAgent(
        model_name="gpt-4o-mini",
        temperature=0.0,
        smartthings_token=smartthings_token,
        device_config=device_config,
    )

    # 테스트 명령
    test_messages = [
        "거실 스피커 꺼줘",
        "안방 불 꺼줘",
    ]

    for i, message in enumerate(test_messages, 1):
        print(f"\n{i}. 사용자: {message}")
        print("-" * 60)

        result = agent.invoke(message=message, thread_id="test_final")

        if "messages" in result:
            for msg in result["messages"]:
                if hasattr(msg, "content") and msg.content:
                    if hasattr(msg, "type"):
                        if msg.type == "ai":
                            print(f"   🤖 {msg.content}")
                        elif msg.type == "tool":
                            print(f"   🔧 {msg.content}")

    print("\n✅ 최종 테스트 완료!")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # OpenAI API Key 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY 환경 변수가 설정되지 않았습니다!")
        sys.exit(1)

    # pytest 실행
    pytest.main([__file__, "-v", "-s", "-m", "integration and safe"])