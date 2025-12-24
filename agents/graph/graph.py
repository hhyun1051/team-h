"""
Team-H Agent System - LangGraph 기반 구현

LangGraph의 StateGraph를 사용하여 명확하고 시각화 가능한 에이전트 협업 시스템 구축:
- 명시적 노드: 라우터와 각 매니저를 노드로 표현
- 조건부 엣지: 라우팅과 핸드오프를 엣지로 구현
- 통합 상태 관리: GraphState로 모든 상태 관리
- 무한 루프 방지: 핸드오프 횟수 제한
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import os
from langgraph.graph import StateGraph
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

# Langfuse 통합
# Note: CallbackHandler는 api/main.py에서 사용
# Middleware는 AgentBase에서 자동 추가

# Agents import
from agents import ManagerI, ManagerM, ManagerS, ManagerT
from agents.context import TeamHContext

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

        # 환경 변수 로딩 (한 번만)
        self._load_env()

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
            workflow.add_node("manager_i", self._create_manager_node("i"))

        if self.manager_m:
            workflow.add_node("manager_m", self._create_manager_node("m"))

        if self.manager_s:
            workflow.add_node("manager_s", self._create_manager_node("s"))

        if self.manager_t:
            workflow.add_node("manager_t", self._create_manager_node("t"))

        # 시작점: 라우터
        workflow.set_entry_point("router")

        # Command 패턴을 사용하므로 conditional edges 불필요

        return workflow.compile(checkpointer=self.checkpointer)

    # ========================================================================
    # 외부 인터페이스
    # ========================================================================
    # Note: invoke(), stream(), invoke_command() 메서드는 제거되었습니다.
    # FastAPI (api/main.py)에서 self.graph.astream_events()를 직접 사용합니다.
    # Langfuse 로깅은 다음 두 계층에서 처리됩니다:
    # 1. Graph 레벨: config["callbacks"]에 CallbackHandler 추가 (api/main.py)
    # 2. Tool 레벨: LangfuseToolLoggingMiddleware (AgentBase)

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

    def _load_env(self):
        """환경 변수 로딩 (한 번만 실행)"""
        from dotenv import load_dotenv

        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)


    def _init_postgres_checkpoint(self):
        """PostgreSQL checkpoint 초기화"""
        if not self.use_postgres_checkpoint:
            raise ValueError(
                "PostgreSQL checkpoint is required for FastAPI backend. "
                "Set use_postgres_checkpoint=True and provide POSTGRES_CONNECTION_STRING in .env"
            )

        try:
            # 환경 변수에서 connection string 가져오기
            conn_string = self.postgres_connection_string or os.getenv(
                "POSTGRES_CONNECTION_STRING"
            )

            if not conn_string:
                raise ValueError(
                    "PostgreSQL connection string not found. "
                    "Set POSTGRES_CONNECTION_STRING in .env or pass postgres_connection_string parameter. "
                    "FastAPI backend requires persistent storage for chat history."
                )

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
            raise RuntimeError(
                f"Failed to initialize PostgreSQL checkpoint: {e}. "
                "FastAPI backend requires persistent storage for chat history."
            ) from e

    def _init_router_llm(self):
        """라우터 LLM 초기화 (중앙화된 factory 사용)"""
        import yaml

        self.router_llm = create_llm()

        # 프롬프트 파일 경로
        prompts_dir = Path(__file__).parent.parent / "prompts"
        router_template_path = prompts_dir / "router.yaml"
        router_descriptions_path = prompts_dir / "router_manager_descriptions.yaml"

        # 라우터 템플릿 읽기 (YAML)
        try:
            with open(router_template_path, "r", encoding="utf-8") as f:
                router_data = yaml.safe_load(f)
            router_template = router_data['content']
        except Exception as e:
            print(f"[⚠️] Failed to load router template: {e}")
            router_template = "You are a routing assistant. Route to appropriate manager."

        # 매니저 설명 읽기 (YAML)
        manager_descriptions_map = {}
        try:
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

    def _get_handoff_tools_for_manager(self, manager_key: str) -> List:
        """
        특정 매니저가 사용할 handoff tools 반환

        Args:
            manager_key: Manager 키 ("i", "m", "s", "t")

        Returns:
            해당 매니저가 사용할 handoff tools 리스트
        """
        tools = []
        for other_key in ["i", "m", "s", "t"]:
            if other_key != manager_key:  # 자기 자신 제외
                enabled_flag = getattr(self, f"enable_manager_{other_key}")
                if enabled_flag:
                    tool_name = f"handoff_to_manager_{other_key}"
                    if tool_name in self.handoff_tools:
                        tools.append(self.handoff_tools[tool_name])
        return tools

    def _init_single_manager(self, manager_key: str, manager_class, **init_kwargs):
        """
        단일 매니저 초기화 헬퍼

        Args:
            manager_key: Manager 키 ("i", "m", "s", "t")
            manager_class: Manager 클래스 (ManagerI, ManagerM, ManagerS, ManagerT)
            **init_kwargs: Manager별 특수 초기화 파라미터

        Returns:
            초기화된 Manager 인스턴스 또는 None (실패 시)
        """
        try:
            # handoff tools 가져오기
            handoff_tools = self._get_handoff_tools_for_manager(manager_key)

            # Manager 초기화
            # Note: AgentBase가 내부적으로 Langfuse 미들웨어를 자동으로 추가함
            manager = manager_class(
                model_name=self.model_name,
                temperature=self.temperature,
                additional_tools=handoff_tools if handoff_tools else None,
                **init_kwargs
            )
            print(f"[✅] Manager {manager_key.upper()} initialized")
            return manager
        except Exception as e:
            print(f"[⚠️] Manager {manager_key.upper()} initialization failed: {e}")
            return None

    def _init_managers(self):
        """각 Manager를 handoff tools와 함께 초기화"""

        # Manager I 초기화
        if self.enable_manager_i:
            self.manager_i = self._init_single_manager(
                "i",
                ManagerI,
                homeassistant_url=self.homeassistant_url,
                homeassistant_token=self.homeassistant_token,
                entity_map=self.entity_map,
            )

        # Manager M 초기화
        if self.enable_manager_m:
            self.manager_m = self._init_single_manager(
                "m",
                ManagerM,
                embedding_type=self.embedding_type,
                embedder_url=self.embedder_url,
                openai_api_key=self.openai_api_key,
                embedding_dims=self.embedding_dims,
                qdrant_url=self.qdrant_url,
                qdrant_api_key=self.qdrant_api_key,
                collection_name=self.m_collection_name,
            )

        # Manager S 초기화
        if self.enable_manager_s:
            self.manager_s = self._init_single_manager(
                "s",
                ManagerS,
                tavily_api_key=self.tavily_api_key,
                max_results=self.max_search_results,
            )

        # Manager T 초기화
        if self.enable_manager_t:
            self.manager_t = self._init_single_manager(
                "t",
                ManagerT,
                google_credentials_path=self.google_credentials_path,
                google_token_path=self.google_token_path,
                calendar_id=self.calendar_id,
            )
