"""
Langfuse Tool Logging Middleware

LangChain 에이전트의 모든 tool call을 Langfuse에 자동으로 로깅하는 middleware입니다.
"""

from typing import Callable
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from langfuse import get_client


class LangfuseToolLoggingMiddleware(AgentMiddleware):
    """
    Tool call을 Langfuse에 자동으로 로깅하는 middleware

    이 middleware는 모든 tool call의 input/output을 Langfuse에 로깅합니다:
    - Tool call 시작 시: input과 metadata를 span으로 기록
    - Tool call 완료 시: output을 span에 추가
    - 에러 발생 시: 에러 정보를 span에 기록

    Args:
        langfuse_client: Langfuse client (None이면 get_client()로 자동 초기화)
        verbose: 로그 출력 여부 (기본값: True)
        log_errors: 에러도 Langfuse에 로깅할지 여부 (기본값: True)

    Example:
        ```python
        from agents.middleware import LangfuseToolLoggingMiddleware
        from langchain.agents import create_agent

        # 기본 설정으로 사용
        langfuse_logger = LangfuseToolLoggingMiddleware()

        # 커스터마이징
        langfuse_logger = LangfuseToolLoggingMiddleware(
            verbose=False,
            log_errors=True
        )

        # Agent에 적용
        agent = create_agent(
            model="gpt-4o",
            tools=[my_tools],
            middleware=[langfuse_logger]
        )
        ```
    """

    def __init__(
        self,
        langfuse_client=None,
        verbose: bool = True,
        log_errors: bool = True
    ):
        """
        Langfuse Tool Logging Middleware 초기화

        Args:
            langfuse_client: Langfuse client (None이면 자동 초기화)
            verbose: 로그 출력 여부
            log_errors: 에러도 로깅할지 여부
        """
        self.verbose = verbose
        self.log_errors = log_errors

        # Langfuse 클라이언트 초기화
        if langfuse_client is None:
            try:
                self.langfuse_client = get_client()
                if self.verbose:
                    print(f"[✅] LangfuseToolLoggingMiddleware initialized")
            except Exception as e:
                if self.verbose:
                    print(f"[⚠️] LangfuseToolLoggingMiddleware initialization failed: {e}")
                self.langfuse_client = None
        else:
            self.langfuse_client = langfuse_client
            if self.verbose:
                print(f"[✅] LangfuseToolLoggingMiddleware initialized with provided client")

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """
        Tool call을 Langfuse에 로깅하는 wrapper

        Args:
            request: Tool call request
                - tool_call: dict with 'name', 'args', 'id'
                - tool: BaseTool instance
                - state: Current agent state
                - runtime: Runtime context
            handler: Next handler in the chain

        Returns:
            ToolMessage or Command: Tool execution result
        """
        # Langfuse가 비활성화되어 있으면 그냥 실행
        if not self.langfuse_client:
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
            with self.langfuse_client.start_as_current_observation(
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

                if self.verbose:
                    print(f"[📊] Langfuse logged tool call: {tool_name}")

                return result

        except Exception as e:
            # 에러 발생 시에도 Langfuse에 로깅
            if self.log_errors:
                try:
                    if 'span' in locals() and span:
                        span.update(
                            output={"error": str(e), "error_type": type(e).__name__},
                            level="ERROR"
                        )
                except:
                    pass  # span 업데이트 실패해도 원래 에러를 전파

            if self.verbose:
                print(f"[⚠️] Tool call error logged to Langfuse: {tool_name} - {e}")

            # 에러를 그대로 전파 (middleware는 에러를 숨기지 않음)
            raise
