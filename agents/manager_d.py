# manager_d.py

"""
Manager D Agent - 복잡한 멀티스텝 태스크 처리 에이전트

Manager D는 여러 도메인에 걸친 복합적인 요청을 조율하는 에이전트입니다:
- 작업 계획 수립 (TodoListMiddleware)
- 파일 시스템 관리 (FilesystemMiddleware)
- 다른 매니저들에게 작업 위임 (SubAgentMiddleware)
- 컨텍스트 자동 관리 (SummarizationMiddleware)

ManagerBase를 상속받되, deepagents의 미들웨어를 활용합니다.
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, TYPE_CHECKING

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Agents import
from agents import ManagerBase
from agents.context import TeamHContext

# LangChain middleware
from langchain.agents.middleware import TodoListMiddleware, SummarizationMiddleware

# DeepAgents middleware and backends
try:
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from deepagents.middleware.subagents import SubAgentMiddleware
    from deepagents.backends import FilesystemBackend
    from deepagents import CompiledSubAgent
    DEEPAGENTS_AVAILABLE = True
except ImportError:
    print("⚠️  deepagents가 설치되지 않았습니다. pip install deepagents를 실행하세요.")
    DEEPAGENTS_AVAILABLE = False
    # TYPE_CHECKING을 위한 더미 타입
    if TYPE_CHECKING:
        from deepagents import CompiledSubAgent


class ManagerD(ManagerBase):
    """Manager D 에이전트 클래스 - 복잡한 멀티스텝 태스크 조율 전문"""

    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        temperature: float = 0.7,
        workspace_dir: str = "./workspace",
        existing_managers: Optional[Dict[str, Any]] = None,
        additional_tools: Optional[List] = None,
        middleware: Optional[List] = None,
        enable_summarization: bool = True,
        max_context_tokens: int = 170000,
    ):
        """
        Manager D 에이전트 초기화

        Args:
            model_name: 사용할 LLM 모델 이름 (기본값: gpt-4.1-mini)
            temperature: 모델 temperature 설정
            workspace_dir: 작업 파일을 저장할 디렉토리
            existing_managers: 기존 Manager 인스턴스 dict (예: {'m': manager_m, 's': manager_s})
            additional_tools: 핸드오프 등 추가 툴 리스트
            middleware: 외부에서 전달받은 미들웨어 리스트 (Langfuse 로깅 등)
            enable_summarization: 자동 요약 활성화 여부
            max_context_tokens: 요약을 트리거하는 최대 컨텍스트 토큰 수
        """
        if not DEEPAGENTS_AVAILABLE:
            raise ImportError(
                "deepagents 패키지가 필요합니다. pip install deepagents를 실행하세요."
            )

        # Workspace 디렉토리 생성
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        print(f"[📁] Workspace directory: {self.workspace_dir.absolute()}")

        # 기존 매니저들 저장
        self.existing_managers = existing_managers or {}

        # Model 초기화 (SummarizationMiddleware에서 필요)
        from langchain.chat_models import init_chat_model
        model = init_chat_model(model_name, temperature=temperature)

        # DeepAgents 미들웨어 조합
        deep_middlewares = []

        # 1. TodoList Middleware (계획 수립)
        deep_middlewares.append(TodoListMiddleware())

        # 2. Filesystem Middleware (파일 관리)
        # FilesystemBackend를 직접 사용 (모든 파일 작업이 workspace_dir에서 이루어짐)
        backend = FilesystemBackend(root_dir=str(self.workspace_dir))
        deep_middlewares.append(FilesystemMiddleware(backend=backend))

        # 3. SubAgent Middleware는 HITL이 작동하지 않으므로 사용하지 않음
        # 대신 Handoff 툴을 사용하여 다른 매니저에게 위임
        # (Handoff 방식은 TeamHGraph 레벨에서 HITL이 정상 작동함)

        # 4. Summarization Middleware (자동 요약)
        if enable_summarization:
            deep_middlewares.append(
                SummarizationMiddleware(model=model, max_tokens=max_context_tokens)
            )

        # 외부 미들웨어와 합치기
        combined_middleware = []
        if middleware:
            combined_middleware.extend(middleware)
        combined_middleware.extend(deep_middlewares)

        # 베이스 클래스 초기화 (공통 로직)
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            additional_tools=additional_tools,
            middleware=combined_middleware,
        )

        # 추가 초기화 메시지
        print(f"    - Workspace: {self.workspace_dir}")
        print(f"    - SubAgents: {len(self.existing_managers)} managers available")
        print(f"    - Summarization: {'Enabled' if enable_summarization else 'Disabled'}")


    def _create_tools(self) -> List:
        """
        Manager D는 deepagents 미들웨어가 자동으로 도구를 추가합니다.

        DeepAgents 미들웨어가 자동으로 추가하는 도구:
        - TodoListMiddleware: write_todos
        - FilesystemMiddleware: ls, read_file, write_file, edit_file, glob, grep

        다른 매니저로의 위임은 Handoff 툴을 사용합니다 (additional_tools로 전달됨).
        """
        # 현재는 커스텀 도구 없음
        # 필요시 여기에 Manager D만의 특수 도구 추가 가능
        return []
