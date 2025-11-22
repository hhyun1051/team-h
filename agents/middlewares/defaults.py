"""
Default Middlewares Helper

기본 middleware 세트를 편리하게 사용하기 위한 헬퍼 클래스입니다.
"""

from typing import Optional
from .langfuse_logging import LangfuseToolLoggingMiddleware
from .error_handler import ToolErrorHandlerMiddleware


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
