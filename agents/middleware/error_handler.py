"""
Tool Error Handler Middleware

Tool 실행 중 발생하는 에러를 우아하게 처리하는 middleware입니다.
"""

from typing import Optional, Callable
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command


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
        from agents.middleware import ToolErrorHandlerMiddleware

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
