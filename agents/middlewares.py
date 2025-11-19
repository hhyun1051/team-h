# middlewares.py
"""
공통 Agent Middleware 모음

LangChain 에이전트에서 사용할 수 있는 재사용 가능한 middleware들을 정의합니다.
- Langfuse 로깅 middleware
- Tool call 추적 middleware
- 에러 처리 middleware 등
"""

from typing import Optional
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langfuse import get_client
import os


def create_langfuse_tool_logging_middleware():
    """
    Tool call을 Langfuse에 자동으로 로깅하는 middleware 생성

    이 middleware는 모든 tool call의 input/output을 Langfuse에 로깅합니다.
    - Tool call 시작 시: input과 metadata를 span으로 기록
    - Tool call 완료 시: output을 span에 추가
    - 에러 발생 시: 에러 정보를 span에 기록

    Returns:
        Tool logging middleware 함수

    Example:
        ```python
        from agents.middlewares import create_langfuse_tool_logging_middleware
        from langchain.agents import create_agent

        # Middleware 생성
        tool_logger = create_langfuse_tool_logging_middleware()

        # Agent에 적용
        agent = create_agent(
            model="gpt-4o",
            tools=[my_tools],
            middleware=[tool_logger]
        )
        ```
    """
    # Langfuse singleton client 가져오기 (환경 변수로 초기화됨)
    try:
        langfuse = get_client()
        print(f"[✅] Langfuse middleware initialized")
    except Exception as e:
        print(f"[⚠️] Langfuse middleware initialization failed: {e}")
        langfuse = None

    @wrap_tool_call
    def log_tool_call_to_langfuse(request, handler):
        """
        Tool call을 Langfuse에 로깅하는 wrapper

        Args:
            request: Tool call request
                - tool_call: dict with 'name', 'args', 'id'
                - state: Current agent state
                - runtime: Runtime context
            handler: Next handler in the chain

        Returns:
            ToolMessage: Tool execution result
        """
        # Langfuse가 비활성화되어 있으면 그냥 실행
        if not langfuse:
            return handler(request)

        # Tool call 정보 추출
        tool_name = request.tool_call.get("name", "unknown_tool")
        tool_args = request.tool_call.get("args", {})
        tool_call_id = request.tool_call.get("id")

        # 상태에서 추가 메타데이터 추출 (가능한 경우)
        metadata = {
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
        }

        # runtime context가 있으면 추가 정보 포함
        if hasattr(request, 'runtime') and request.runtime:
            runtime_context = getattr(request.runtime, 'context', {})
            if runtime_context:
                metadata["runtime_context"] = runtime_context

        try:
            # Langfuse v3: context manager를 사용하여 span 생성
            # CallbackHandler가 만든 trace context에 자동으로 중첩됨
            with langfuse.start_as_current_observation(
                as_type="span",
                name=f"tool:{tool_name}",
                input=tool_args,  # input을 시작 시 전달
                metadata=metadata,  # metadata도 시작 시 전달
            ) as span:
                # 실제 tool 실행
                result = handler(request)

                # Tool 실행 결과 로깅
                output_content = result.content if hasattr(result, 'content') else str(result)

                # Span에 output 기록
                span.update(output={"content": output_content})

                print(f"[📊] Langfuse logged tool call: {tool_name}")

                return result

        except Exception as e:
            # 에러 발생 시에도 Langfuse에 로깅
            # context manager가 자동으로 span을 종료하지만, 에러 정보 추가
            try:
                if 'span' in locals() and span:
                    span.update(
                        output={"error": str(e), "error_type": type(e).__name__},
                        level="ERROR"
                    )
            except:
                pass  # span 업데이트 실패해도 원래 에러를 전파

            print(f"[⚠️] Tool call error logged to Langfuse: {tool_name} - {e}")

            # 에러를 그대로 전파 (middleware는 에러를 숨기지 않음)
            raise

    return log_tool_call_to_langfuse


def create_tool_error_handler():
    """
    Tool 실행 에러를 우아하게 처리하는 middleware

    Tool 실행 중 발생한 예외를 catch하고,
    모델이 이해할 수 있는 친절한 에러 메시지로 변환합니다.

    Returns:
        Tool error handling middleware

    Example:
        ```python
        from agents.middlewares import create_tool_error_handler

        error_handler = create_tool_error_handler()

        agent = create_agent(
            model="gpt-4o",
            tools=[my_tools],
            middleware=[error_handler]
        )
        ```
    """
    @wrap_tool_call
    def handle_tool_errors(request, handler):
        """Handle tool execution errors with friendly messages."""
        try:
            return handler(request)
        except Exception as e:
            tool_name = request.tool_call.get("name", "unknown")
            error_msg = (
                f"⚠️ Tool '{tool_name}' encountered an error.\n"
                f"Error: {str(e)}\n"
                f"Please check your input and try again, or use a different approach."
            )

            return ToolMessage(
                content=error_msg,
                tool_call_id=request.tool_call["id"]
            )

    return handle_tool_errors


# 편의를 위한 pre-configured middleware
def get_default_middlewares():
    """
    기본 middleware 세트 반환

    Returns:
        list: [tool_logger, error_handler] middleware 리스트
    """
    return [
        create_langfuse_tool_logging_middleware(),
        create_tool_error_handler(),
    ]
