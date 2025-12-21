"""
모델이 어떤 tool calling 형식을 사용하는지 테스트

vLLM 서버에 직접 요청해서 모델의 raw output을 확인합니다.
"""

import requests
import json

# vLLM 서버 설정
VLLM_URL = "http://100.91.240.35:8002/v1/completions"
MODEL_NAME = "kanana-2-30b-a3b-thinking"

# Tool calling을 유도하는 프롬프트
prompt = """You are a helpful assistant with access to the following function:

Function: get_weather
Description: Get current weather for a location
Parameters:
  - location (string, required): The city name

User: What's the weather in Seoul?
Assistant: I will call the function to get the weather."""

print("="*60)
print("Testing Tool Calling Format")
print("="*60)
print(f"Model: {MODEL_NAME}")
print(f"Server: {VLLM_URL}")
print()

# 요청 보내기
payload = {
    "model": MODEL_NAME,
    "prompt": prompt,
    "max_tokens": 200,
    "temperature": 0.1,
    "stop": ["\n\n", "User:"]
}

print("Sending request to vLLM server...")
try:
    response = requests.post(VLLM_URL, json=payload, timeout=30)
    response.raise_for_status()

    result = response.json()
    output = result['choices'][0]['text']

    print("\n" + "="*60)
    print("Model Output (Raw):")
    print("="*60)
    print(output)
    print()

    # 형식 분석
    print("="*60)
    print("Format Analysis:")
    print("="*60)

    if "<tool_call>" in output:
        print("✅ Detected: Hermes Format")
        print("   Example: <tool_call>{...}</tool_call>")
        print("   Use: --tool-call-parser hermes")
    elif '"tool_calls"' in output or '"function"' in output:
        print("✅ Detected: OpenAI Format")
        print("   Example: {\"tool_calls\": [...]}")
        print("   Use: --tool-call-parser auto (default)")
    elif "<|python_tag|>" in output:
        print("✅ Detected: Llama 3 Format")
        print("   Example: <|python_tag|>function_name(...)")
        print("   Use: --tool-call-parser llama3")
    elif ">>>" in output and "{" in output:
        print("✅ Detected: Functionary Format")
        print("   Example: >>>function_name\\n{...}")
        print("   Use: --tool-call-parser functionary")
    else:
        print("❌ No standard tool calling format detected")
        print("   The model might:")
        print("   1. Not be trained for function calling")
        print("   2. Use a custom format")
        print("   3. Need different prompting")
        print()
        print("   Raw output for manual inspection:")
        print(f"   '{output}'")

except requests.exceptions.RequestException as e:
    print(f"❌ Error connecting to vLLM server: {e}")
except Exception as e:
    print(f"❌ Error: {e}")