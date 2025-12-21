"""
vLLM thinking 모델의 structured output 지원 여부 테스트

이 스크립트는 vLLM 서버가 structured output (tool calling)을 지원하는지 테스트합니다.
"""

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from pydantic import BaseModel, Field
from typing import Literal
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class AgentRouting(BaseModel):
    """간단한 라우팅 결정 모델"""
    target_agent: Literal["i", "m", "s", "t"] = Field(
        description="The target agent: 'i' for IoT, 'm' for memory, 's' for search, 't' for time"
    )
    reason: str = Field(
        description="Brief explanation of why this agent was chosen"
    )


def test_openai_with_structured_output():
    """OpenAI API로 with_structured_output 테스트 (정상 작동 확인용)"""
    print("\n" + "="*60)
    print("TEST 1: OpenAI API - with_structured_output")
    print("="*60)

    try:
        llm = init_chat_model(
            model="gpt-4o-mini",
            model_provider="openai",
            temperature=0.7,
        )

        routing_agent = llm.with_structured_output(AgentRouting)

        result = routing_agent.invoke([
            {"role": "system", "content": "You are a routing assistant. Route user requests to the appropriate agent."},
            {"role": "user", "content": "거실 불 켜줘"}
        ])

        print(f"✅ OpenAI with_structured_output 성공!")
        print(f"   Target: {result.target_agent}")
        print(f"   Reason: {result.reason}")
        return True

    except Exception as e:
        print(f"❌ OpenAI with_structured_output 실패: {e}")
        return False


def test_openai_tool_strategy():
    """OpenAI API로 ToolStrategy 테스트"""
    print("\n" + "="*60)
    print("TEST 2: OpenAI API - ToolStrategy")
    print("="*60)

    try:
        llm = init_chat_model(
            model="gpt-4o-mini",
            model_provider="openai",
            temperature=0.7,
        )

        agent = create_agent(
            model=llm,
            tools=[],
            response_format=ToolStrategy(AgentRouting)
        )

        result = agent.invoke({
            "messages": [{"role": "user", "content": "거실 불 켜줘"}]
        })

        structured_response = result.get("structured_response")
        if structured_response:
            print(f"✅ OpenAI ToolStrategy 성공!")
            print(f"   Target: {structured_response.target_agent}")
            print(f"   Reason: {structured_response.reason}")
            return True
        else:
            print(f"❌ OpenAI ToolStrategy 실패!")
            print(f"   structured_response를 찾을 수 없음")
            return False

    except Exception as e:
        print(f"❌ OpenAI ToolStrategy 실패: {e}")
        return False


def test_openai_provider_strategy():
    """OpenAI API로 ProviderStrategy 테스트"""
    print("\n" + "="*60)
    print("TEST 3: OpenAI API - ProviderStrategy")
    print("="*60)

    try:
        llm = init_chat_model(
            model="gpt-4o-mini",
            model_provider="openai",
            temperature=0.7,
        )

        agent = create_agent(
            model=llm,
            tools=[],
            response_format=ProviderStrategy(AgentRouting)
        )

        result = agent.invoke({
            "messages": [{"role": "user", "content": "거실 불 켜줘"}]
        })

        structured_response = result.get("structured_response")
        if structured_response:
            print(f"✅ OpenAI ProviderStrategy 성공!")
            print(f"   Target: {structured_response.target_agent}")
            print(f"   Reason: {structured_response.reason}")
            return True
        else:
            print(f"❌ OpenAI ProviderStrategy 실패!")
            print(f"   structured_response를 찾을 수 없음")
            return False

    except Exception as e:
        print(f"❌ OpenAI ProviderStrategy 실패: {e}")
        return False


def test_openai_simple_completion():
    """OpenAI API로 일반 텍스트 생성 테스트"""
    print("\n" + "="*60)
    print("TEST 4: OpenAI API - Simple Text Completion")
    print("="*60)

    try:
        llm = init_chat_model(
            model="gpt-4o-mini",
            model_provider="openai",
            temperature=0.7,
        )

        result = llm.invoke([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "안녕하세요. 간단히 인사해주세요."}
        ])

        print(f"✅ OpenAI 일반 텍스트 생성 성공!")
        print(f"   응답: {result.content[:100]}...")
        return True

    except Exception as e:
        print(f"❌ OpenAI 일반 텍스트 생성 실패: {e}")
        return False


def test_vllm_with_structured_output():
    """vLLM으로 with_structured_output 테스트"""
    print("\n" + "="*60)
    print("TEST 5: vLLM (gpt-oss-120b) - with_structured_output")
    print("="*60)

    # vLLM 설정 하드코딩 (.env의 주석 처리된 값 사용)
    vllm_base_url = "100.91.240.35:8002/v1"
    vllm_api_key = "EMPTY"
    vllm_model = "gpt-oss-120b"

    # URL에 http:// 없으면 추가
    if not vllm_base_url.startswith(("http://", "https://")):
        vllm_base_url = f"http://{vllm_base_url}"

    print(f"vLLM 설정:")
    print(f"  - URL: {vllm_base_url}")
    print(f"  - Model: {vllm_model}")
    print(f"  - API Key: {vllm_api_key}")

    try:
        llm = init_chat_model(
            model=vllm_model,
            model_provider="openai",  # OpenAI compatible
            base_url=vllm_base_url,
            api_key=vllm_api_key,
            temperature=0.7,
        )

        print("\n[시도 1] with_structured_output() 사용...")
        routing_agent = llm.with_structured_output(AgentRouting)

        result = routing_agent.invoke([
            {"role": "system", "content": "You are a routing assistant. Route user requests to the appropriate agent."},
            {"role": "user", "content": "거실 불 켜줘"}
        ])

        print(f"✅ vLLM structured output 성공!")
        print(f"   Target: {result.target_agent}")
        print(f"   Reason: {result.reason}")
        return True

    except Exception as e:
        print(f"❌ vLLM structured output 실패!")
        print(f"   에러: {e}")
        print(f"\n원인 분석:")

        error_msg = str(e)
        if "tool choice" in error_msg.lower():
            print("  → Tool choice 관련 에러 발생")
            print("  → vLLM 서버가 tool calling을 지원하지 않거나")
            print("  → --enable-auto-tool-choice 플래그가 설정되지 않음")
        elif "thinking" in error_msg.lower():
            print("  → Thinking 모델 관련 에러")
            print("  → Thinking 모델은 structured output을 지원하지 않을 수 있음")

        return False


def test_vllm_tool_strategy():
    """vLLM으로 ToolStrategy 방식의 structured output 테스트"""
    print("\n" + "="*60)
    print("TEST 6: vLLM (gpt-oss-120b) - ToolStrategy")
    print("="*60)

    # vLLM 설정 하드코딩
    vllm_base_url = "100.91.240.35:8002/v1"
    vllm_api_key = "EMPTY"
    vllm_model = "gpt-oss-120b"

    # URL에 http:// 없으면 추가
    if not vllm_base_url.startswith(("http://", "https://")):
        vllm_base_url = f"http://{vllm_base_url}"

    print(f"vLLM 설정:")
    print(f"  - URL: {vllm_base_url}")
    print(f"  - Model: {vllm_model}")
    print(f"  - Strategy: ToolStrategy")

    try:
        llm = init_chat_model(
            model=vllm_model,
            model_provider="openai",
            base_url=vllm_base_url,
            api_key=vllm_api_key,
            temperature=0.7,
        )

        print("\n[시도] ToolStrategy 사용...")

        # ToolStrategy로 agent 생성
        agent = create_agent(
            model=llm,
            tools=[],
            response_format=ToolStrategy(AgentRouting)
        )

        result = agent.invoke({
            "messages": [{"role": "user", "content": "거실 불 켜줘"}]
        })

        structured_response = result.get("structured_response")
        if structured_response:
            print(f"✅ vLLM ToolStrategy 성공!")
            print(f"   Target: {structured_response.target_agent}")
            print(f"   Reason: {structured_response.reason}")
            return True
        else:
            print(f"❌ vLLM ToolStrategy 실패!")
            print(f"   structured_response를 찾을 수 없음")
            return False

    except Exception as e:
        print(f"❌ vLLM ToolStrategy 실패!")
        print(f"   에러: {e}")
        print(f"\n원인 분석:")

        error_msg = str(e)
        if "tool choice" in error_msg.lower():
            print("  → Tool choice 관련 에러 발생")
            print("  → vLLM 서버가 tool calling을 지원하지 않거나")
            print("  → --enable-auto-tool-choice 플래그가 설정되지 않음")
        elif "thinking" in error_msg.lower():
            print("  → Thinking 모델 관련 에러")
            print("  → Thinking 모델은 tool calling을 지원하지 않을 수 있음")

        return False


def test_vllm_provider_strategy():
    """vLLM으로 ProviderStrategy 방식의 structured output 테스트"""
    print("\n" + "="*60)
    print("TEST 7: vLLM (gpt-oss-120b) - ProviderStrategy")
    print("="*60)

    # vLLM 설정 하드코딩
    vllm_base_url = "100.91.240.35:8002/v1"
    vllm_api_key = "EMPTY"
    vllm_model = "gpt-oss-120b"

    # URL에 http:// 없으면 추가
    if not vllm_base_url.startswith(("http://", "https://")):
        vllm_base_url = f"http://{vllm_base_url}"

    print(f"vLLM 설정:")
    print(f"  - URL: {vllm_base_url}")
    print(f"  - Model: {vllm_model}")
    print(f"  - Strategy: ProviderStrategy")

    try:
        llm = init_chat_model(
            model=vllm_model,
            model_provider="openai",
            base_url=vllm_base_url,
            api_key=vllm_api_key,
            temperature=0.7,
        )

        print("\n[시도] ProviderStrategy 사용...")

        # ProviderStrategy로 agent 생성
        agent = create_agent(
            model=llm,
            tools=[],
            response_format=ProviderStrategy(AgentRouting)
        )

        result = agent.invoke({
            "messages": [{"role": "user", "content": "거실 불 켜줘"}]
        })

        structured_response = result.get("structured_response")
        if structured_response:
            print(f"✅ vLLM ProviderStrategy 성공!")
            print(f"   Target: {structured_response.target_agent}")
            print(f"   Reason: {structured_response.reason}")
            return True
        else:
            print(f"❌ vLLM ProviderStrategy 실패!")
            print(f"   structured_response를 찾을 수 없음")
            return False

    except Exception as e:
        print(f"❌ vLLM ProviderStrategy 실패!")
        print(f"   에러: {e}")
        print(f"\n원인 분석:")

        error_msg = str(e)
        if "does not support" in error_msg.lower() or "not supported" in error_msg.lower():
            print("  → Provider가 native structured output을 지원하지 않음")
            print("  → vLLM은 OpenAI/Grok처럼 provider-native structured output을 지원하지 않을 수 있음")
            print("  → ToolStrategy를 사용하는 것이 권장됨")
        elif "tool choice" in error_msg.lower():
            print("  → Tool choice 관련 에러 발생")
            print("  → ProviderStrategy가 내부적으로 tool calling을 사용할 수 있음")

        return False


def test_vllm_simple_completion():
    """vLLM으로 일반 텍스트 생성 테스트 (서버 연결 확인용)"""
    print("\n" + "="*60)
    print("TEST 8: vLLM (gpt-oss-120b) - Simple Text Completion")
    print("="*60)

    # vLLM 설정 하드코딩
    vllm_base_url = "100.91.240.35:8002/v1"
    vllm_api_key = "EMPTY"
    vllm_model = "gpt-oss-120b"

    if not vllm_base_url.startswith(("http://", "https://")):
        vllm_base_url = f"http://{vllm_base_url}"

    try:
        llm = init_chat_model(
            model=vllm_model,
            model_provider="openai",
            base_url=vllm_base_url,
            api_key=vllm_api_key,
            temperature=0.7,
        )

        # structured output 없이 일반 호출
        result = llm.invoke([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "안녕하세요. 간단히 인사해주세요."}
        ])

        print(f"✅ vLLM 일반 텍스트 생성 성공!")
        print(f"   응답: {result.content[:100]}...")
        return True

    except Exception as e:
        print(f"❌ vLLM 일반 텍스트 생성 실패: {e}")
        return False


def main():
    """모든 테스트 실행"""
    print("\n" + "="*60)
    print("vLLM Thinking 모델 Structured Output 지원 테스트")
    print("="*60)

    results = {}

    # OpenAI Tests (baseline)
    results['openai_with_structured'] = test_openai_with_structured_output()
    results['openai_tool_strategy'] = test_openai_tool_strategy()
    results['openai_provider_strategy'] = test_openai_provider_strategy()
    results['openai_simple'] = test_openai_simple_completion()

    # vLLM Tests
    results['vllm_with_structured'] = test_vllm_with_structured_output()
    results['vllm_tool_strategy'] = test_vllm_tool_strategy()
    results['vllm_provider_strategy'] = test_vllm_provider_strategy()
    results['vllm_simple'] = test_vllm_simple_completion()

    # 결과 요약
    print("\n" + "="*60)
    print("테스트 결과 요약")
    print("="*60)
    print("\n[OpenAI API - Baseline]")
    print(f"1. with_structured_output:   {'✅ 성공' if results['openai_with_structured'] else '❌ 실패'}")
    print(f"2. ToolStrategy:              {'✅ 성공' if results['openai_tool_strategy'] else '❌ 실패'}")
    print(f"3. ProviderStrategy:          {'✅ 성공' if results['openai_provider_strategy'] else '❌ 실패'}")
    print(f"4. Simple Completion:         {'✅ 성공' if results['openai_simple'] else '❌ 실패'}")
    print("\n[vLLM (gpt-oss-120b)]")
    print(f"5. with_structured_output:   {'✅ 성공' if results['vllm_with_structured'] else '❌ 실패'}")
    print(f"6. ToolStrategy:              {'✅ 성공' if results['vllm_tool_strategy'] else '❌ 실패'}")
    print(f"7. ProviderStrategy:          {'✅ 성공' if results['vllm_provider_strategy'] else '❌ 실패'}")
    print(f"8. Simple Completion:         {'✅ 성공' if results['vllm_simple'] else '❌ 실패'}")

    print("\n" + "="*60)
    print("결론")
    print("="*60)

    # OpenAI 결과 분석
    openai_structured_success = (
        results['openai_with_structured'] or
        results['openai_tool_strategy'] or
        results['openai_provider_strategy']
    )

    # vLLM 결과 분석
    vllm_structured_success = (
        results['vllm_with_structured'] or
        results['vllm_tool_strategy'] or
        results['vllm_provider_strategy']
    )

    print("\n[OpenAI Baseline 상태]")
    if not results['openai_simple']:
        print("❌ OpenAI API 연결 실패 - API 키를 확인하세요")
    elif openai_structured_success:
        print("✅ OpenAI structured output 정상 작동 (baseline 확인됨)")
    else:
        print("⚠️  OpenAI에서도 structured output 실패 - 환경 또는 코드 문제 가능성")

    print("\n[vLLM 상태]")
    if not results['vllm_simple']:
        print("❌ vLLM 서버에 연결할 수 없습니다.")
        print("   → 서버 주소와 포트를 확인해주세요.")
    elif vllm_structured_success:
        print("✅ vLLM이 structured output을 지원합니다!")
        print("\n지원되는 방식:")
        if results['vllm_with_structured']:
            print("  ✅ with_structured_output")
        if results['vllm_tool_strategy']:
            print("  ✅ ToolStrategy")
        if results['vllm_provider_strategy']:
            print("  ✅ ProviderStrategy")
        print("\n권장 사항:")
        if results['vllm_tool_strategy']:
            print("  → ToolStrategy 방식 사용 권장 (LangChain agent에서 표준 방식)")
        elif results['vllm_with_structured']:
            print("  → with_structured_output 방식도 사용 가능")
    else:
        print("❌ vLLM 서버는 연결되었으나 structured output을 지원하지 않습니다.")
        print("\n테스트 결과:")
        print(f"  - with_structured_output: {'성공' if results['vllm_with_structured'] else '실패'}")
        print(f"  - ToolStrategy: {'성공' if results['vllm_tool_strategy'] else '실패'}")
        print(f"  - ProviderStrategy: {'성공' if results['vllm_provider_strategy'] else '실패'}")
        print("\n가능한 원인:")
        print("  1. vLLM 서버에 tool calling 관련 플래그가 설정되지 않음")
        print("  2. 모델이 tool calling 형식을 학습하지 않았거나 지원하지 않음")
        print("\n해결 방법:")
        print("  → gpt-oss-120b의 경우:")
        print("    --reasoning-parser openai_gptoss")
        print("    --tool-call-parser openai")
        print("    --enable-auto-tool-choice")
        print("  → 또는 router에서 structured output 대신 텍스트 파싱 사용")


if __name__ == "__main__":
    main()
