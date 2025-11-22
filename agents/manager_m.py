# manager_m.py

"""
Manager M Agent - 일반 기억 관리 에이전트

Manager M은 목표 외의 모든 일반적인 기억을 관리하는 에이전트입니다:
- 사용자 선호도
- 대화 컨텍스트
- 일상적인 상호작용
- 사용자 습관 및 패턴

ManagerBase를 상속받아 공통 로직을 재사용합니다.
HumanInTheLoopMiddleware를 통해 모든 기억 관련 작업에 대한 승인을 요구합니다.
"""

import sys
from pathlib import Path
from typing import Optional, List

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Agents import (__init__.py 활용)
from agents import ManagerBase
from agents.context import TeamHContext
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool, ToolRuntime
from database.qdrant.manager_m_memory import ManagerMMemory


class ManagerM(ManagerBase):
    """Manager M 에이전트 클래스 - 일반 기억 관리 전문"""

    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        temperature: float = 0.7,
        embedding_type: Optional[str] = None,
        embedder_url: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        embedding_dims: Optional[int] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        additional_tools: Optional[List] = None,
        middleware: Optional[List] = None,
    ):
        """
        Manager M 에이전트 초기화

        Args:
            model_name: 사용할 LLM 모델 이름 (기본값: gpt-4o-mini)
            temperature: 모델 temperature 설정
            embedding_type: 임베딩 타입 ("fastapi" 또는 "openai")
            embedder_url: FastAPI 임베딩 서버 URL
            openai_api_key: OpenAI API 키 (embedding_type="openai"일 때 사용)
            embedding_dims: 임베딩 차원 (선택사항)
            qdrant_url: Qdrant 서버 URL
            qdrant_api_key: Qdrant API 키
            collection_name: Qdrant 컬렉션 이름
            additional_tools: 핸드오프 등 추가 툴 리스트
            middleware: 외부에서 전달받은 미들웨어 리스트 (Langfuse 로깅 등)
        """
        # HITL 미들웨어 생성
        hitl_middleware = HumanInTheLoopMiddleware(
            interrupt_on={
                # 쓰기/수정/삭제 작업만 승인 필요
                "add_memory": True,
                "update_memory": True,
                "delete_memory": True,
                "delete_all_memories": True,
            },
            description_prefix="🧠 Memory operation pending approval",
        )

        # middleware 리스트 합치기 (외부 middleware + HITL)
        combined_middleware = []
        if middleware:
            combined_middleware.extend(middleware)
        combined_middleware.append(hitl_middleware)

        # 베이스 클래스 초기화 (공통 로직)
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            additional_tools=additional_tools,
            middleware=combined_middleware,
            # Memory 초기화를 위한 파라미터 전달
            embedding_type=embedding_type,
            embedder_url=embedder_url,
            openai_api_key=openai_api_key,
            embedding_dims=embedding_dims,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            collection_name=collection_name,
        )

        # 추가 초기화 메시지
        print(f"    - HITL: Enabled for write operations")

    def _pre_init_hook(self, **kwargs):
        """Memory 초기화 (툴 생성 전에 필요)"""
        self.memory = ManagerMMemory(
            embedding_type=kwargs.get("embedding_type"),
            embedder_url=kwargs.get("embedder_url"),
            openai_api_key=kwargs.get("openai_api_key"),
            embedding_dims=kwargs.get("embedding_dims"),
            qdrant_url=kwargs.get("qdrant_url"),
            qdrant_api_key=kwargs.get("qdrant_api_key"),
            collection_name=kwargs.get("collection_name"),
        )

    def _create_tools(self) -> List:
        """메모리 관리 관련 툴 생성 (ToolRuntime 사용)"""

        @tool
        def add_memory(
            content: str,
            memory_type: str = "general",
            runtime: ToolRuntime[TeamHContext] = None
        ) -> str:
            """
            Add a new memory to the system.

            Args:
                content: The content of the memory to add
                memory_type: Type of memory (general, preference, habit, interaction, etc.)
                runtime: Automatically injected runtime context (contains user_id)

            Returns:
                Confirmation message with the memory ID
            """
            try:
                # Runtime context에서 user_id 안전하게 가져옴
                user_id = runtime.context.user_id if runtime else "default_user"

                result = self.memory.add_memory(
                    content=content,
                    memory_type=memory_type,
                    user_id=user_id,
                )
                return f"✅ Memory added successfully: {result}"
            except Exception as e:
                return f"❌ Error adding memory: {str(e)}"

        @tool
        def search_memories(
            query: str,
            limit: int = 5,
            runtime: ToolRuntime[TeamHContext] = None
        ) -> str:
            """
            Search for memories related to the query.

            Args:
                query: Search query string
                limit: Maximum number of results to return
                runtime: Automatically injected runtime context (contains user_id)

            Returns:
                List of relevant memories
            """
            try:
                user_id = runtime.context.user_id if runtime else "default_user"

                results = self.memory.search_memories(
                    query=query,
                    user_id=user_id,
                    limit=limit,
                )

                if not results:
                    return f"No memories found for query: '{query}'"

                formatted_results = []
                for i, result in enumerate(results, 1):
                    memory_id = result.get("id", "unknown")
                    content = result.get("content", "No content")
                    memory_type = result.get("type", "unknown")
                    score = result.get("score", 0.0)

                    formatted_results.append(
                        f"### Memory {i}\n"
                        f"**ID:** {memory_id}\n"
                        f"**Type:** {memory_type}\n"
                        f"**Content:** {content}\n"
                        f"**Score:** {score:.3f}\n"
                    )

                return "\n".join(formatted_results)

            except Exception as e:
                return f"❌ Error searching memories: {str(e)}"

        @tool
        def get_all_memories(
            limit: int = 10,
            runtime: ToolRuntime[TeamHContext] = None
        ) -> str:
            """
            Get all memories for a user.

            Args:
                limit: Maximum number of memories to return
                runtime: Automatically injected runtime context (contains user_id)

            Returns:
                List of all user's memories
            """
            try:
                user_id = runtime.context.user_id if runtime else "default_user"

                results = self.memory.get_all_memories(
                    user_id=user_id,
                    limit=limit,
                )

                if not results:
                    return f"No memories found for user: {user_id}"

                formatted_results = []
                for i, result in enumerate(results, 1):
                    memory_id = result.get("id", "unknown")
                    content = result.get("content", "No content")
                    memory_type = result.get("type", "unknown")

                    formatted_results.append(
                        f"### Memory {i}\n"
                        f"**ID:** {memory_id}\n"
                        f"**Type:** {memory_type}\n"
                        f"**Content:** {content}\n"
                    )

                return "\n".join(formatted_results)

            except Exception as e:
                return f"❌ Error getting memories: {str(e)}"

        @tool
        def update_memory(
            memory_id: str,
            content: str,
            runtime: ToolRuntime[TeamHContext] = None
        ) -> str:
            """
            Update an existing memory.

            Args:
                memory_id: ID of the memory to update
                content: New content for the memory
                runtime: Automatically injected runtime context (contains user_id)

            Returns:
                Confirmation message
            """
            try:
                user_id = runtime.context.user_id if runtime else "default_user"

                result = self.memory.update_memory(
                    memory_id=memory_id,
                    content=content,
                    user_id=user_id,
                )
                return f"✅ Memory updated successfully: {result}"
            except Exception as e:
                return f"❌ Error updating memory: {str(e)}"

        @tool
        def delete_memory(
            memory_id: str,
            runtime: ToolRuntime[TeamHContext] = None
        ) -> str:
            """
            Delete a specific memory.

            Args:
                memory_id: ID of the memory to delete
                runtime: Automatically injected runtime context (contains user_id)

            Returns:
                Confirmation message
            """
            try:
                user_id = runtime.context.user_id if runtime else "default_user"

                result = self.memory.delete_memory(
                    memory_id=memory_id,
                    user_id=user_id,
                )
                return f"✅ Memory deleted successfully: {result}"
            except Exception as e:
                return f"❌ Error deleting memory: {str(e)}"

        @tool
        def delete_all_memories(runtime: ToolRuntime[TeamHContext] = None) -> str:
            """
            Delete all memories for a user.

            ⚠️ WARNING: This is a destructive operation!

            Args:
                runtime: Automatically injected runtime context (contains user_id)

            Returns:
                Confirmation message
            """
            try:
                user_id = runtime.context.user_id if runtime else "default_user"

                result = self.memory.delete_all_memories(user_id=user_id)
                return f"✅ All memories deleted for user {user_id}: {result}"
            except Exception as e:
                return f"❌ Error deleting all memories: {str(e)}"

        return [
            add_memory,
            search_memories,
            get_all_memories,
            update_memory,
            delete_memory,
            delete_all_memories,
        ]


def create_manager_m_agent(**kwargs) -> ManagerM:
    """
    Manager M 에이전트 생성 헬퍼 함수

    Args:
        **kwargs: ManagerM 초기화 파라미터

    Returns:
        ManagerM 인스턴스
    """
    return ManagerM(**kwargs)


# 싱글톤 인스턴스 (선택적 사용)
_manager_m_agent_instance = None


def get_manager_m_agent(**kwargs) -> ManagerM:
    """
    Manager M 에이전트 싱글톤 인스턴스 반환

    Args:
        **kwargs: ManagerM 초기화 파라미터 (처음 생성 시에만 사용됨)

    Returns:
        ManagerM 인스턴스
    """
    global _manager_m_agent_instance
    if _manager_m_agent_instance is None:
        _manager_m_agent_instance = ManagerM(**kwargs)
    return _manager_m_agent_instance
