"""
Team-H Agent System - LangGraph 기반 구현

LangGraph의 StateGraph를 사용하여 명확하고 시각화 가능한 에이전트 협업 시스템 구축:
- 명시적 노드: 라우터와 각 매니저를 노드로 표현
- 조건부 엣지: 라우팅과 핸드오프를 엣지로 구현
- 통합 상태 관리: GraphState로 모든 상태 관리
- 무한 루프 방지: 핸드오프 횟수 제한
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import os
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
import psycopg
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

# Langfuse 통합
from langfuse import observe, get_client
from langfuse.langchain import CallbackHandler

# Agents import
from agents import ManagerI, ManagerM, ManagerS, ManagerT
from agents.context import TeamHContext
from agents.middleware import LangfuseToolLoggingMiddleware, ToolErrorHandlerMiddleware

# Utils import
from utils.llm_factory import create_llm

# Local imports
from .state import TeamHState, AgentRouting
from .nodes import NodesMixin


class TeamHGraph(NodesMixin):
    """LangGraph 기반 Team-H 에이전트 시스템"""

    def __init__(
        self,
        # Manager activation flags
        enable_manager_i: bool = True,
        enable_manager_m: bool = True,
        enable_manager_s: bool = True,
        enable_manager_t: bool = True,

        # Manager I params (Home Assistant)
        homeassistant_url: str = "http://localhost:8124",
        homeassistant_token: Optional[str] = None,
        entity_map: Optional[Dict[str, str]] = None,

        # Manager M params
        embedding_type: Optional[str] = None,
        embedder_url: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        embedding_dims: Optional[int] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        m_collection_name: Optional[str] = None,

        # Manager S params
        tavily_api_key: Optional[str] = None,
        max_search_results: int = 5,

        # Manager T params
        google_credentials_path: Optional[str] = None,
        google_token_path: Optional[str] = None,
        calendar_id: str = "primary",

        # Common params
        model_name: str = "gpt-4.1-mini",
        temperature: float = 0.7,
        max_handoffs: int = 5,

        # PostgreSQL checkpoint params
        postgres_connection_string: Optional[str] = None,
        use_postgres_checkpoint: bool = True,
    ):
        """
        Team-H Graph 초기화

        Args:
            enable_manager_i: Manager I 활성화 여부
            enable_manager_m: Manager M 활성화 여부
            enable_manager_s: Manager S 활성화 여부
            enable_manager_t: Manager T 활성화 여부
            homeassistant_url: Home Assistant URL
            homeassistant_token: Home Assistant Long-Lived Access Token
            entity_map: Entity ID 매핑 (옵션)
            embedding_type: 임베딩 타입 ("fastapi" 또는 "openai")
            embedder_url: FastAPI 임베딩 서버 URL
            openai_api_key: OpenAI API 키
            embedding_dims: 임베딩 차원
            qdrant_url: Qdrant 서버 URL
            qdrant_api_key: Qdrant API 키
            m_collection_name: Qdrant 컬렉션 이름
            tavily_api_key: Tavily API 키
            max_search_results: 검색 결과 최대 개수
            google_credentials_path: Google OAuth credentials.json 경로
            google_token_path: Google OAuth token 저장 경로
            calendar_id: Google Calendar ID
            model_name: LLM 모델 이름
            temperature: 모델 temperature
            max_handoffs: 최대 핸드오프 횟수 (무한 루프 방지)
            postgres_connection_string: PostgreSQL connection string (옵션)
            use_postgres_checkpoint: PostgreSQL checkpoint 사용 여부 (기본값: True)
        """
        print(f"[🤖] Initializing Team-H Graph System...")

        # AgentRouting 클래스 저장 (nodes.py에서 사용)
        self.AgentRouting = AgentRouting

        # Langfuse 초기화 (환경 변수 기반)
        self._init_langfuse()

        # PostgreSQL Checkpoint 초기화
        self.use_postgres_checkpoint = use_postgres_checkpoint
        self.postgres_connection_string = postgres_connection_string
        self._init_postgres_checkpoint()

        self.model_name = model_name
        self.temperature = temperature
        self.max_handoffs = max_handoffs

        # Manager 활성화 플래그 저장
        self.enable_manager_i = enable_manager_i and homeassistant_token
        self.enable_manager_m = enable_manager_m
        self.enable_manager_s = enable_manager_s and tavily_api_key
        self.enable_manager_t = enable_manager_t

        # 설정 저장 (Home Assistant)
        self.homeassistant_url = homeassistant_url
        self.homeassistant_token = homeassistant_token
        self.entity_map = entity_map
        self.embedding_type = embedding_type
        self.embedder_url = embedder_url
        self.openai_api_key = openai_api_key
        self.embedding_dims = embedding_dims
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.m_collection_name = m_collection_name
        self.tavily_api_key = tavily_api_key
        self.max_search_results = max_search_results
        self.google_credentials_path = google_credentials_path
        self.google_token_path = google_token_path
        self.calendar_id = calendar_id

        # 최소 하나의 매니저는 활성화되어야 함
        assert (
            self.enable_manager_i
            or self.enable_manager_m
            or self.enable_manager_s
            or self.enable_manager_t
        ), "At least one manager must be enabled"

        # Handoff tools 생성 (Manager 생성 전)
        self._create_handoff_tools()

        # 각 매니저를 handoff tools와 함께 초기화
        self.manager_i = None
        self.manager_m = None
        self.manager_s = None
        self.manager_t = None

        self._init_managers()

        # 라우터 LLM 초기화
        self._init_router_llm()

        # 그래프 빌드
        self.graph = self._build_graph()

        print(f"[✅] Team-H Graph System initialized successfully")
        print(f"    - Max handoffs: {self.max_handoffs}")

    # ========================================================================
    # 🎯 핵심: 그래프 구조 정의
    # ========================================================================

    def _build_graph(self) -> StateGraph:
        """
        Team-H 에이전트 그래프 빌드

        Nodes:
          - router: 요청 분석 및 라우팅 (첫 턴만)
          - manager_i: IoT 디바이스 제어 (Home Assistant)
          - manager_m: 메모리 관리 (Qdrant 벡터 DB)
          - manager_s: 웹 검색 (Tavily API)
          - manager_t: 일정/시간 관리 (Google Calendar)

        Flow:
          1. 사용자 메시지 → router
          2. router → 적절한 manager 선택
          3. manager 실행 → 다른 manager로 handoff 가능
          4. 최대 {self.max_handoffs}번까지 handoff
        """
        workflow = StateGraph(TeamHState)

        # 노드 추가
        workflow.add_node("router", self._router_node)

        if self.manager_i:
            workflow.add_node("manager_i", self._manager_i_node)

        if self.manager_m:
            workflow.add_node("manager_m", self._manager_m_node)

        if self.manager_s:
            workflow.add_node("manager_s", self._manager_s_node)

        if self.manager_t:
            workflow.add_node("manager_t", self._manager_t_node)

        # 시작점: 라우터
        workflow.set_entry_point("router")
        
        # Command 패턴을 사용하므로 conditional edges 불필요

        # 컴파일 - PostgresSaver 또는 MemorySaver 사용
        return workflow.compile(checkpointer=self.checkpointer)

    # ========================================================================
    # 외부 인터페이스
    # ========================================================================

    @observe(name="team-h-graph-invoke", capture_input=True, capture_output=True)
    def invoke(
        self,
        message: str,
        user_id: str = "default_user",
        thread_id: str = "default",
        session_id: Optional[str] = None,
        callbacks: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        그래프 실행

        Args:
            message: 사용자 메시지
            user_id: 사용자 ID (예: "user-123", 로그인 시스템에서 제공)
            thread_id: PostgreSQL checkpoint thread ID (대화 세션 식별)
            session_id: Langfuse session ID (옵션, 없으면 thread_id 사용)
            callbacks: Langfuse CallbackHandler 등의 콜백 리스트

        Returns:
            최종 상태

        Note:
            통합 ID 전략:
            - thread_id: PostgreSQL checkpoint (대화 저장/재개)
            - session_id: Langfuse 추적 (없으면 thread_id 사용)
            - user_id: 사용자 식별 (PostgreSQL + Langfuse)
        """
        # session_id가 없으면 thread_id를 session_id로 사용 (통합 전략)
        if session_id is None:
            session_id = thread_id

        # Langfuse CallbackHandler 자동 생성
        if callbacks is None and self.langfuse_client:
            try:
                # v3: session_id, user_id는 metadata로 전달
                langfuse_handler = CallbackHandler()
                callbacks = [langfuse_handler]
            except Exception as e:
                print(f"[⚠️] Failed to create Langfuse handler: {e}")
                callbacks = []

        # Context 생성 (TeamHContext)
        context = TeamHContext(
            user_id=user_id,
            thread_id=thread_id,
            session_id=session_id
        )

        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": callbacks or [],
            "metadata": {
                "langfuse_session_id": session_id,  # Langfuse v3: metadata로 session_id 전달
                "langfuse_user_id": user_id,        # Langfuse v3: metadata로 user_id 전달
                "langfuse_tags": ["team-h", "graph"],
            }
        }

        initial_state = {
            "messages": [HumanMessage(content=message)],
            "handoff_count": 0,
        }

        result = self.graph.invoke(initial_state, config, context=context)
        return result

    def invoke_command(
        self,
        command: Any,
        config: Dict[str, Any],
        user_id: str = "default_user",
        thread_id: str = "default",
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Command를 사용하여 그래프 재개 (HITL 지원)

        Args:
            command: LangGraph Command 객체 (resume 등)
            config: 그래프 설정 (thread_id 포함)
            user_id: 사용자 ID
            thread_id: 스레드 ID
            session_id: Langfuse 세션 ID (옵션)

        Returns:
            그래프 실행 결과
        """
        # Context 생성 (tools에서 runtime.context로 접근 가능)
        context = TeamHContext(
            user_id=user_id,
            thread_id=thread_id,
            session_id=session_id or thread_id
        )

        result = self.graph.invoke(command, config, context=context)
        return result

    @observe(name="team-h-graph-stream", capture_input=True, capture_output=True)
    def stream(
        self,
        message: str,
        user_id: str = "default_user",
        thread_id: str = "default",
        session_id: Optional[str] = None,
        callbacks: Optional[List] = None,
    ):
        """
        그래프 스트리밍 실행

        Args:
            message: 사용자 메시지
            user_id: 사용자 ID (예: "user-123")
            thread_id: PostgreSQL checkpoint thread ID (대화 세션 식별)
            session_id: Langfuse session ID (옵션, 없으면 thread_id 사용)
            callbacks: Langfuse CallbackHandler 등의 콜백 리스트

        Yields:
            각 노드 실행 결과
        """
        # session_id가 없으면 thread_id를 session_id로 사용
        if session_id is None:
            session_id = thread_id

        # Langfuse CallbackHandler 자동 생성
        if callbacks is None and self.langfuse_client:
            try:
                # v3: session_id, user_id는 metadata로 전달
                langfuse_handler = CallbackHandler()
                callbacks = [langfuse_handler]
            except Exception as e:
                print(f"[⚠️] Failed to create Langfuse handler: {e}")
                callbacks = []

        # Context 생성 (TeamHContext)
        context = TeamHContext(
            user_id=user_id,
            thread_id=thread_id,
            session_id=session_id
        )

        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": callbacks or [],
            "metadata": {
                "langfuse_session_id": session_id,  # Langfuse v3: metadata로 session_id 전달
                "langfuse_user_id": user_id,        # Langfuse v3: metadata로 user_id 전달
                "langfuse_tags": ["team-h", "graph"],
            }
        }

        initial_state = {
            "messages": [HumanMessage(content=message)],
            "handoff_count": 0,
        }

        for chunk in self.graph.stream(initial_state, config, context=context):
            yield chunk

    def get_graph_visualization(self) -> str:
        """
        그래프를 Mermaid 다이어그램으로 반환

        Returns:
            Mermaid 다이어그램 문자열
        """
        try:
            from langgraph.graph import draw_mermaid
            return draw_mermaid(self.graph)
        except Exception as e:
            return f"Visualization not available: {e}"

    # ========================================================================
    # 초기화 헬퍼 메서드 (내부용 - IDE에서 접어두고 볼 것)
    # ========================================================================

    def _init_langfuse(self):
        """Langfuse 초기화 (환경 변수 기반)"""
        try:
            # .env 파일이 있으면 로드
            from pathlib import Path
            from dotenv import load_dotenv

            env_path = Path(__file__).parent.parent.parent / ".env"
            if env_path.exists():
                load_dotenv(env_path)

            # Langfuse v3: singleton client 사용 (환경 변수 자동 사용)
            # LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_BASE_URL
            self.langfuse_client = get_client()
            print(f"[✅] Langfuse initialized: {os.getenv('LANGFUSE_BASE_URL')}")

            # Tool call 로깅을 위한 공통 middleware 생성
            # v3에서는 middleware가 내부적으로 get_client()를 호출함
            self.tool_logging_middleware = LangfuseToolLoggingMiddleware(verbose=True)
            print(f"[✅] Langfuse tool logging middleware created")
        except Exception as e:
            print(f"[⚠️] Langfuse initialization failed: {e}")
            self.langfuse_client = None
            self.tool_logging_middleware = None

    def _init_postgres_checkpoint(self):
        """PostgreSQL checkpoint 초기화"""
        if not self.use_postgres_checkpoint:
            print(f"[ℹ️] Using in-memory checkpoint (MemorySaver)")
            self.checkpointer = MemorySaver()
            return

        try:
            # .env 파일이 있으면 로드 (Langfuse 초기화에서 이미 했을 수 있지만 안전하게)
            from pathlib import Path
            from dotenv import load_dotenv

            env_path = Path(__file__).parent.parent.parent / ".env"
            if env_path.exists():
                load_dotenv(env_path)

            # 환경 변수에서 connection string 가져오기
            conn_string = self.postgres_connection_string or os.getenv(
                "POSTGRES_CONNECTION_STRING"
            )

            if not conn_string:
                print(f"[⚠️] PostgreSQL connection string not found, falling back to MemorySaver")
                print(f"[ℹ️] Set POSTGRES_CONNECTION_STRING in .env or pass postgres_connection_string parameter")
                self.checkpointer = MemorySaver()
                return

            # Async Connection pool 생성
            self.db_pool = AsyncConnectionPool(
                conninfo=conn_string,
                max_size=20,
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": 0,
                    "row_factory": dict_row,
                }
            )

            # AsyncPostgresSaver 초기화
            self.checkpointer = AsyncPostgresSaver(self.db_pool)

            # 테이블 자동 생성은 비동기로 수행되어야 하므로 startup에서 처리
            # Note: setup()은 동기 메서드이므로 여기서는 호출하지 않음

            print(f"[✅] PostgreSQL checkpoint initialized")
            print(f"[ℹ️] Chat history will be persisted to PostgreSQL")

        except Exception as e:
            print(f"[⚠️] PostgreSQL checkpoint initialization failed: {e}")
            print(f"[ℹ️] Falling back to in-memory checkpoint (MemorySaver)")
            self.checkpointer = MemorySaver()

    def _init_router_llm(self):
        """라우터 LLM 초기화 (중앙화된 factory 사용)"""
        self.router_llm = create_llm()

        # 프롬프트 파일 경로
        prompts_dir = Path(__file__).parent.parent / "prompts"
        router_template_path = prompts_dir / "router.yaml"
        router_descriptions_path = prompts_dir / "router_manager_descriptions.yaml"

        # 라우터 템플릿 읽기 (YAML)
        try:
            import yaml
            with open(router_template_path, "r", encoding="utf-8") as f:
                router_data = yaml.safe_load(f)
            router_template = router_data['content']
        except Exception as e:
            print(f"[⚠️] Failed to load router template: {e}")
            router_template = "You are a routing assistant. Route to appropriate manager."

        # 매니저 설명 읽기 (YAML)
        manager_descriptions_map = {}
        try:
            import yaml
            with open(router_descriptions_path, "r", encoding="utf-8") as f:
                manager_descriptions_map = yaml.safe_load(f)

            # YAML에서 읽은 값들의 끝 공백 제거
            manager_descriptions_map = {
                k: v.strip() if isinstance(v, str) else v
                for k, v in manager_descriptions_map.items()
            }
        except Exception as e:
            print(f"[⚠️] Failed to load manager descriptions: {e}")
            # 폴백: 하드코딩된 설명 사용
            manager_descriptions_map = {
                "i": "'i' (IoT Control): Control smart devices",
                "m": "'m' (Memory): Store/recall user information",
                "s": "'s' (Web Search): Find real-time information",
                "t": "'t' (Calendar/Time): Manage schedules",
            }

        # 활성화된 매니저에 대한 설명만 선택
        manager_descriptions = []
        if self.manager_i and "i" in manager_descriptions_map:
            manager_descriptions.append(manager_descriptions_map["i"])
        if self.manager_m and "m" in manager_descriptions_map:
            manager_descriptions.append(manager_descriptions_map["m"])
        if self.manager_s and "s" in manager_descriptions_map:
            manager_descriptions.append(manager_descriptions_map["s"])
        if self.manager_t and "t" in manager_descriptions_map:
            manager_descriptions.append(manager_descriptions_map["t"])

        # 템플릿에 매니저 설명 주입
        self.router_prompt = router_template.format(
            manager_descriptions="\n\n".join(manager_descriptions)
        )

    def _create_handoff_tools(self):
        """각 Manager로 handoff하는 tool 생성 (Manager 생성 전, 플래그 기반)"""
        self.handoff_tools = {}

        # Manager I로 handoff
        if self.enable_manager_i:
            @tool
            def handoff_to_manager_i(reason: str) -> str:
                """
                Hand off the conversation to Manager I (IoT Control Agent).

                Use this when you need IoT device control capabilities:
                - Controlling lights (living room, bedroom, bathroom)
                - Controlling smart speakers
                - Shutting down mini PC

                Args:
                    reason: Brief explanation of why handoff is needed

                Returns:
                    Confirmation message
                """
                return f"[HANDOFF_TO_I] {reason}"

            self.handoff_tools["handoff_to_manager_i"] = handoff_to_manager_i

        # Manager M로 handoff
        if self.enable_manager_m:
            @tool
            def handoff_to_manager_m(reason: str) -> str:
                """
                Hand off the conversation to Manager M (Memory Management Agent).

                Use this when you need memory/context capabilities:
                - Storing user information, preferences, or habits
                - Recalling past conversations or user data
                - Managing long-term context

                Args:
                    reason: Brief explanation of why handoff is needed

                Returns:
                    Confirmation message
                """
                return f"[HANDOFF_TO_M] {reason}"

            self.handoff_tools["handoff_to_manager_m"] = handoff_to_manager_m

        # Manager S로 handoff
        if self.enable_manager_s:
            @tool
            def handoff_to_manager_s(reason: str) -> str:
                """
                Hand off the conversation to Manager S (Web Search Agent).

                Use this when you need web search capabilities:
                - Finding real-time information
                - Searching for news or current events
                - Looking up facts or data online

                Args:
                    reason: Brief explanation of why handoff is needed

                Returns:
                    Confirmation message
                """
                return f"[HANDOFF_TO_S] {reason}"

            self.handoff_tools["handoff_to_manager_s"] = handoff_to_manager_s

        # Manager T로 handoff
        if self.enable_manager_t:
            @tool
            def handoff_to_manager_t(reason: str) -> str:
                """
                Hand off the conversation to Manager T (Time/Calendar Management Agent).

                Use this when you need calendar/scheduling capabilities:
                - Creating, viewing, or modifying calendar events
                - Setting reminders and notifications
                - Checking schedules and upcoming events
                - Managing time-based tasks

                Args:
                    reason: Brief explanation of why handoff is needed

                Returns:
                    Confirmation message
                """
                return f"[HANDOFF_TO_T] {reason}"

            self.handoff_tools["handoff_to_manager_t"] = handoff_to_manager_t

    def _init_managers(self):
        """각 Manager를 handoff tools와 함께 초기화"""

        # 공통 middleware 리스트 (모든 Manager에 적용)
        common_middlewares = []
        if self.tool_logging_middleware:
            common_middlewares.append(self.tool_logging_middleware)

        # Manager I 초기화
        if self.enable_manager_i:
            try:
                # Manager I에게 M, S, T로의 handoff tool 추가
                handoff_tools_for_i = []
                if self.enable_manager_m:
                    handoff_tools_for_i.append(self.handoff_tools["handoff_to_manager_m"])
                if self.enable_manager_s:
                    handoff_tools_for_i.append(self.handoff_tools["handoff_to_manager_s"])
                if self.enable_manager_t:
                    handoff_tools_for_i.append(self.handoff_tools["handoff_to_manager_t"])

                self.manager_i = ManagerI(
                    model_name=self.model_name,
                    temperature=self.temperature,
                    homeassistant_url=self.homeassistant_url,
                    homeassistant_token=self.homeassistant_token,
                    entity_map=self.entity_map,
                    additional_tools=handoff_tools_for_i if handoff_tools_for_i else None,
                    middleware=common_middlewares if common_middlewares else None,
                )
                print(f"[✅] Manager I initialized")
            except Exception as e:
                print(f"[⚠️] Manager I initialization failed: {e}")
                self.manager_i = None

        # Manager M 초기화
        if self.enable_manager_m:
            try:
                # Manager M에게 I, S, T로의 handoff tool 추가
                handoff_tools_for_m = []
                if self.enable_manager_i:
                    handoff_tools_for_m.append(self.handoff_tools["handoff_to_manager_i"])
                if self.enable_manager_s:
                    handoff_tools_for_m.append(self.handoff_tools["handoff_to_manager_s"])
                if self.enable_manager_t:
                    handoff_tools_for_m.append(self.handoff_tools["handoff_to_manager_t"])

                self.manager_m = ManagerM(
                    model_name=self.model_name,
                    temperature=self.temperature,
                    embedding_type=self.embedding_type,
                    embedder_url=self.embedder_url,
                    openai_api_key=self.openai_api_key,
                    embedding_dims=self.embedding_dims,
                    qdrant_url=self.qdrant_url,
                    qdrant_api_key=self.qdrant_api_key,
                    collection_name=self.m_collection_name,
                    additional_tools=handoff_tools_for_m if handoff_tools_for_m else None,
                    middleware=common_middlewares if common_middlewares else None,
                )
                print(f"[✅] Manager M initialized")
            except Exception as e:
                print(f"[⚠️] Manager M initialization failed: {e}")
                self.manager_m = None

        # Manager S 초기화
        if self.enable_manager_s:
            try:
                # Manager S에게 I, M, T로의 handoff tool 추가
                handoff_tools_for_s = []
                if self.enable_manager_i:
                    handoff_tools_for_s.append(self.handoff_tools["handoff_to_manager_i"])
                if self.enable_manager_m:
                    handoff_tools_for_s.append(self.handoff_tools["handoff_to_manager_m"])
                if self.enable_manager_t:
                    handoff_tools_for_s.append(self.handoff_tools["handoff_to_manager_t"])

                self.manager_s = ManagerS(
                    model_name=self.model_name,
                    temperature=self.temperature,
                    tavily_api_key=self.tavily_api_key,
                    max_results=self.max_search_results,
                    additional_tools=handoff_tools_for_s if handoff_tools_for_s else None,
                    middleware=common_middlewares if common_middlewares else None,
                )
                print(f"[✅] Manager S initialized")
            except Exception as e:
                print(f"[⚠️] Manager S initialization failed: {e}")
                self.manager_s = None

        # Manager T 초기화
        if self.enable_manager_t:
            try:
                # Manager T에게 I, M, S로의 handoff tool 추가
                handoff_tools_for_t = []
                if self.enable_manager_i:
                    handoff_tools_for_t.append(self.handoff_tools["handoff_to_manager_i"])
                if self.enable_manager_m:
                    handoff_tools_for_t.append(self.handoff_tools["handoff_to_manager_m"])
                if self.enable_manager_s:
                    handoff_tools_for_t.append(self.handoff_tools["handoff_to_manager_s"])

                self.manager_t = ManagerT(
                    model_name=self.model_name,
                    temperature=self.temperature,
                    google_credentials_path=self.google_credentials_path,
                    google_token_path=self.google_token_path,
                    calendar_id=self.calendar_id,
                    additional_tools=handoff_tools_for_t if handoff_tools_for_t else None,
                    middleware=common_middlewares if common_middlewares else None,
                )
                print(f"[✅] Manager T initialized")
            except Exception as e:
                print(f"[⚠️] Manager T initialization failed: {e}")
                self.manager_t = None
