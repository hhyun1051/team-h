"""
Team-H Runtime Context

모든 tool에서 접근 가능한 런타임 컨텍스트를 정의합니다.
LangChain의 ToolRuntime을 통해 안전하게 전달됩니다.
"""

from dataclasses import dataclass


@dataclass
class TeamHContext:
    """
    Team-H Runtime Context

    모든 Manager의 tool에서 runtime.context로 접근 가능한 컨텍스트입니다.
    타입 안전하게 user_id, thread_id, session_id를 제공합니다.

    Attributes:
        user_id: 사용자 식별자 (예: "user-123")
        thread_id: PostgreSQL checkpoint thread ID (대화 세션)
        session_id: Langfuse 추적용 session ID

    Example:
        ```python
        from langchain.tools import tool, ToolRuntime
        from agents.context import TeamHContext

        @tool
        def my_tool(arg: str, runtime: ToolRuntime[TeamHContext]) -> str:
            user_id = runtime.context.user_id  # 타입 안전!
            thread_id = runtime.context.thread_id
            # ...
        ```
    """
    user_id: str
    thread_id: str
    session_id: str
