# middlewares.py
"""
공통 Agent Middleware 모음 (클래스 기반)

LangChain 에이전트에서 사용할 수 있는 재사용 가능한 middleware들을 정의합니다.
- Langfuse 로깅 middleware
- Tool call 에러 처리 middleware

설계:
- AgentMiddleware 상속을 통한 표준 패턴 준수
- 생성자를 통한 설정 주입으로 커스터마이징 용이
- wrap_tool_call 메서드 override로 tool 실행 가로채기
"""

from typing import Optional, Callable
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from langfuse import get_client
import os


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
        from agents.middlewares import LangfuseToolLoggingMiddleware
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


class ToolErrorHandlerMiddleware(AgentMiddleware):
    """
    Tool 실행 에러를 우아하게 처리하는 middleware

    Tool 실행 중 발생한 예외를 catch하고,
    모델이 이해할 수 있는 친절한 에러 메시지로 변환합니다.

    Args:
        error_message_template: 에러 메시지 템플릿 (tool_name, error를 포함)
        include_error_details: 상세 에러 내용 포함 여부 (기본값: True)

    Example:
        ```python
        from agents.middlewares import ToolErrorHandlerMiddleware

        # 기본 설정으로 사용
        error_handler = ToolErrorHandlerMiddleware()

        # 커스터마이징
        error_handler = ToolErrorHandlerMiddleware(
            error_message_template="⚠️ '{tool_name}' 도구에서 오류가 발생했습니다: {error}",
            include_error_details=False
        )

        agent = create_agent(
            model="gpt-4o",
            tools=[my_tools],
            middleware=[error_handler]
        )
        ```
    """

    def __init__(
        self,
        error_message_template: Optional[str] = None,
        include_error_details: bool = True
    ):
        """
        Tool Error Handler Middleware 초기화

        Args:
            error_message_template: 에러 메시지 템플릿
            include_error_details: 상세 에러 내용 포함 여부
        """
        self.error_message_template = error_message_template or (
            "⚠️ Tool '{tool_name}' encountered an error.\n"
            "Error: {error}\n"
            "Please check your input and try again, or use a different approach."
        )
        self.include_error_details = include_error_details

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """
        Tool 에러를 처리하는 wrapper

        Args:
            request: Tool call request
            handler: Next handler in the chain

        Returns:
            ToolMessage or Command: Tool execution result or error message
        """
        try:
            return handler(request)
        except Exception as e:
            tool_name = request.tool_call.get("name", "unknown")

            if self.include_error_details:
                error_msg = self.error_message_template.format(
                    tool_name=tool_name,
                    error=str(e)
                )
            else:
                error_msg = self.error_message_template.format(
                    tool_name=tool_name,
                    error="An error occurred"
                )

            return ToolMessage(
                content=error_msg,
                tool_call_id=request.tool_call["id"]
            )


class DefaultMiddlewares:
    """
    기본 middleware 세트를 제공하는 헬퍼 클래스

    편의를 위해 기본 설정의 middleware 조합을 제공합니다.
    각 middleware를 개별적으로 커스터마이징할 수도 있습니다.

    Args:
        langfuse_client: Langfuse client (None이면 자동 초기화)
        langfuse_verbose: Langfuse 로그 출력 여부
        langfuse_log_errors: Langfuse 에러 로깅 여부
        error_message_template: 에러 메시지 템플릿
        include_error_details: 에러 상세 정보 포함 여부

    Example:
        ```python
        from agents.middlewares import DefaultMiddlewares

        # 기본 설정으로 모든 middleware 사용
        middlewares = DefaultMiddlewares().get_all()

        # 커스터마이징
        default_mw = DefaultMiddlewares(
            langfuse_verbose=False,
            include_error_details=False
        )
        middlewares = default_mw.get_all()

        # 또는 개별 middleware만 사용
        middlewares = default_mw.get_langfuse_only()
        ```
    """

    def __init__(
        self,
        langfuse_client=None,
        langfuse_verbose: bool = True,
        langfuse_log_errors: bool = True,
        error_message_template: Optional[str] = None,
        include_error_details: bool = True
    ):
        """
        Default Middlewares 초기화

        Args:
            langfuse_client: Langfuse client
            langfuse_verbose: Langfuse 로그 출력 여부
            langfuse_log_errors: Langfuse 에러 로깅 여부
            error_message_template: 에러 메시지 템플릿
            include_error_details: 에러 상세 정보 포함 여부
        """
        self.langfuse_middleware = LangfuseToolLoggingMiddleware(
            langfuse_client=langfuse_client,
            verbose=langfuse_verbose,
            log_errors=langfuse_log_errors
        )
        self.error_handler_middleware = ToolErrorHandlerMiddleware(
            error_message_template=error_message_template,
            include_error_details=include_error_details
        )

    def get_all(self):
        """
        모든 기본 middleware를 리스트로 반환

        Returns:
            list: [langfuse_logger, error_handler] middleware 인스턴스 리스트
        """
        return [
            self.langfuse_middleware,
            self.error_handler_middleware,
        ]

    def get_langfuse_only(self):
        """
        Langfuse middleware만 반환

        Returns:
            list: [langfuse_logger] middleware 인스턴스 리스트
        """
        return [self.langfuse_middleware]

    def get_error_handler_only(self):
        """
        Error handler middleware만 반환

        Returns:
            list: [error_handler] middleware 인스턴스 리스트
        """
        return [self.error_handler_middleware]
