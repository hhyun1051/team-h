"""
vLLM thinking 모델의 structured output 지원 여부 테스트

이 스크립트는 vLLM 서버가 structured output (tool calling)을 지원하는지 테스트합니다.
"""

from langchain.chat_models import init_chat_model
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


def test_openai_structured_output():
    """OpenAI API로 structured output 테스트 (정상 작동 확인용)"""
    print("\n" + "="*60)
    print("TEST 1: OpenAI API - Structured Output")
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

        print(f"✅ OpenAI 성공!")
        print(f"   Target: {result.target_agent}")
        print(f"   Reason: {result.reason}")
        return True

    except Exception as e:
        print(f"❌ OpenAI 실패: {e}")
        return False


def test_vllm_structured_output():
    """vLLM으로 structured output 테스트"""
    print("\n" + "="*60)
    print("TEST 2: vLLM (kanana-2-30b-a3b-thinking) - Structured Output")
    print("="*60)

    # vLLM 설정 하드코딩 (.env의 주석 처리된 값 사용)
    vllm_base_url = "100.91.240.35:8002/v1"
    vllm_api_key = "EMPTY"
    vllm_model = "kanana-2-30b-a3b-thinking"

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


def test_vllm_simple_completion():
    """vLLM으로 일반 텍스트 생성 테스트 (서버 연결 확인용)"""
    print("\n" + "="*60)
    print("TEST 3: vLLM - Simple Text Completion")
    print("="*60)

    # vLLM 설정 하드코딩
    vllm_base_url = "100.91.240.35:8002/v1"
    vllm_api_key = "EMPTY"
    vllm_model = "kanana-2-30b-a3b-thinking"

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

    # Test 1: OpenAI (baseline)
    results['openai'] = test_openai_structured_output()

    # Test 2: vLLM structured output
    results['vllm_structured'] = test_vllm_structured_output()

    # Test 3: vLLM simple completion
    results['vllm_simple'] = test_vllm_simple_completion()

    # 결과 요약
    print("\n" + "="*60)
    print("테스트 결과 요약")
    print("="*60)
    print(f"1. OpenAI Structured Output: {'✅ 성공' if results['openai'] else '❌ 실패'}")
    print(f"2. vLLM Structured Output:   {'✅ 성공' if results['vllm_structured'] else '❌ 실패'}")
    print(f"3. vLLM Simple Completion:    {'✅ 성공' if results['vllm_simple'] else '❌ 실패'}")

    print("\n" + "="*60)
    print("결론")
    print("="*60)

    if results['vllm_simple'] and not results['vllm_structured']:
        print("✅ vLLM 서버는 정상적으로 연결되었습니다.")
        print("❌ 하지만 structured output (tool calling)은 지원하지 않습니다.")
        print("\n가능한 원인:")
        print("  1. Thinking 모델이 tool calling을 지원하지 않음")
        print("  2. vLLM 서버가 --enable-auto-tool-choice 플래그 없이 실행됨")
        print("  3. 모델 자체가 function calling 형식을 학습하지 않음")
        print("\n해결 방법:")
        print("  - vLLM 서버를 --enable-auto-tool-choice --tool-call-parser hermes 플래그로 재시작")
        print("  - 또는 router에서 structured output 대신 텍스트 파싱 사용")
    elif results['vllm_structured']:
        print("✅ vLLM이 structured output을 완벽하게 지원합니다!")
        print("   이 경우 다른 문제가 원인일 수 있습니다.")
    elif not results['vllm_simple']:
        print("❌ vLLM 서버에 연결할 수 없습니다.")
        print("   서버 주소와 포트를 확인해주세요.")


if __name__ == "__main__":
    main()
