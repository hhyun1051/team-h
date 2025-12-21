"""
LLM Factory - LLM 생성 로직 중앙화

모든 LLM 인스턴스는 이 factory를 통해 생성됩니다.
- OpenAI API
- vLLM (OpenAI compatible API)
- Ollama

사용 예:
    from utils.llm_factory import create_llm

    llm = create_llm(model_name="gpt-4.1-mini", temperature=0.7)
"""

from typing import Optional
from langchain.chat_models import init_chat_model
from config.settings import llm_config


def create_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
):
    """
    중앙화된 LLM 생성 함수

    설정 파일(config/settings.py)의 LLM_PROVIDER 값에 따라 적절한 LLM을 생성합니다.

    Args:
        model_name: 모델 이름 (None이면 설정 파일의 기본값 사용)
        temperature: Temperature 설정 (None이면 설정 파일의 기본값 사용)

    Returns:
        LangChain ChatModel 인스턴스

    Examples:
        # 기본 설정 사용
        llm = create_llm()

        # 커스텀 모델/temperature
        llm = create_llm(model_name="gpt-4o", temperature=0.9)

        # vLLM 사용 (.env에서 LLM_PROVIDER=vllm 설정)
        llm = create_llm(model_name="meta-llama/Llama-3-8b-chat-hf")
    """
    # 설정에서 기본값 가져오기
    model = model_name or llm_config.llm_model_name
    temp = temperature if temperature is not None else llm_config.llm_temperature

    provider = llm_config.llm_provider

    print(f"[🤖] Creating LLM: provider={provider}, model={model}, temperature={temp}")

    if provider == "openai":
        return init_chat_model(
            model=model,
            model_provider="openai",
            temperature=temp,
        )

    elif provider == "vllm":
        # vLLM은 OpenAI compatible API를 제공
        # base_url을 직접 전달해야 함
        base_url = llm_config.vllm_base_url

        # URL에 http:// 또는 https:// 없으면 추가
        if not base_url.startswith(("http://", "https://")):
            base_url = f"http://{base_url}"

        print(f"[🔧] vLLM base_url: {base_url}")

        return init_chat_model(
            model=model,
            model_provider="openai",  # OpenAI compatible
            base_url=base_url,
            api_key=llm_config.vllm_api_key,
            temperature=temp,
        )

    elif provider == "ollama":
        return init_chat_model(
            model=model,
            model_provider="ollama",
            base_url=llm_config.ollama_base_url,
            temperature=temp,
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            f"Supported providers: 'openai', 'vllm', 'ollama'"
        )
