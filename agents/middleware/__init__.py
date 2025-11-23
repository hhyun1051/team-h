"""
Agent Middleware Package

확장 가능한 middleware 컬렉션:
- Langfuse 로깅 middleware
- Tool 에러 처리 middleware
- 추후 추가될 middleware들을 위한 모듈형 구조

Usage:
    from agents.middleware import LangfuseToolLoggingMiddleware, ToolErrorHandlerMiddleware

    # 직접 조합하여 사용
    middlewares = [
        LangfuseToolLoggingMiddleware(verbose=True),
        ToolErrorHandlerMiddleware()
    ]
"""

from .langfuse_logging import LangfuseToolLoggingMiddleware
from .error_handler import ToolErrorHandlerMiddleware

__all__ = [
    "LangfuseToolLoggingMiddleware",
    "ToolErrorHandlerMiddleware",
]
