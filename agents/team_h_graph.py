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
from typing import Annotated, Literal, Optional, Dict, Any, TypedDict, List
from pydantic import BaseModel, Field

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

# Langfuse 통합
from langfuse import observe

from agents.manager_i import ManagerI
from agents.manager_m import ManagerM
from agents.manager_s import ManagerS
from agents.manager_t import ManagerT


# ============================================================================
# 1. State 정의
# ============================================================================

class TeamHState(TypedDict):
    """
    Team-H 그래프 상태
    """
    messages: Annotated[list, add_messages]  # 대화 메시지
    next_agent: Literal["router", "i", "m", "s", "t", "end"]  # 다음 실행할 노드
    user_id: str  # 사용자 ID (Manager M용)
    routing_reason: str  # 라우팅 이유 (디버그용)
    handoff_count: int  # 핸드오프 횟수 (무한 루프 방지)
    current_agent: Optional[str]  # 현재 실행 중인 에이전트
    last_active_manager: Optional[str]  # 마지막 활성 Manager ("i", "m", "s", "t")


# ============================================================================
# 2. 라우팅 결정 스키마
# ============================================================================

class AgentRouting(BaseModel):
    """라우터의 라우팅 결정"""
    target_agent: Literal["i", "m", "s", "t"] = Field(
        description="The target agent: 'i' for IoT, 'm' for memory, 's' for search, 't' for time/calendar"
    )
    reason: str = Field(
        description="Brief explanation of why this agent was chosen"
    )


# ============================================================================
# 3. Team-H Graph 클래스
# ============================================================================

class TeamHGraph:
    """LangGraph 기반 Team-H 에이전트 시스템"""

    def __init__(
        self,
        # Manager activation flags
        enable_manager_i: bool = True,
        enable_manager_m: bool = True,
        enable_manager_s: bool = True,
        enable_manager_t: bool = True,

        # Manager I params
        smartthings_token: Optional[str] = None,
        device_config: Optional[Dict[str, str]] = None,

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
    ):
        """
        Team-H Graph 초기화

        Args:
            enable_manager_i: Manager I 활성화 여부
            enable_manager_m: Manager M 활성화 여부
            enable_manager_s: Manager S 활성화 여부
            enable_manager_t: Manager T 활성화 여부
            smartthings_token: SmartThings API 토큰
            device_config: IoT 장치 설정
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
        """
        print(f"[🤖] Initializing Team-H Graph System...")

        self.model_name = model_name
        self.temperature = temperature
        self.max_handoffs = max_handoffs

        # Manager 활성화 플래그 저장
        self.enable_manager_i = enable_manager_i and smartthings_token
        self.enable_manager_m = enable_manager_m
        self.enable_manager_s = enable_manager_s and tavily_api_key
        self.enable_manager_t = enable_manager_t

        # 설정 저장
        self.smartthings_token = smartthings_token
        self.device_config = device_config
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

    def _init_router_llm(self):
        """라우터 LLM 초기화"""
        self.router_llm = init_chat_model(
            model=self.model_name,
            model_provider="openai",
            temperature=self.temperature,
        )

        # 프롬프트 파일 경로
        prompts_dir = Path(__file__).parent / "prompts"
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
                    smartthings_token=self.smartthings_token,
                    device_config=self.device_config,
                    additional_tools=handoff_tools_for_i if handoff_tools_for_i else None,
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
                )
                print(f"[✅] Manager T initialized")
            except Exception as e:
                print(f"[⚠️] Manager T initialization failed: {e}")
                self.manager_t = None

    def _build_graph(self) -> StateGraph:
        """그래프 빌드 (Command 패턴 사용)"""
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
        # 각 노드에서 Command의 goto 파라미터로 다음 노드를 직접 지정

        # 컴파일
        return workflow.compile(checkpointer=MemorySaver())

    # ========================================================================
    # 노드 함수들
    # ========================================================================

    def _router_node(self, state: TeamHState, config: Optional[Dict[str, Any]] = None) -> Command:
        """라우터 노드 - 초기 라우팅 결정 (첫 턴) 또는 last_active_manager 사용"""
        last_active = state.get("last_active_manager")

        # last_active_manager가 있으면 Router LLM 호출 생략하고 계속 사용
        if last_active:
            print(f"[🔀] Router: Continuing with last active Manager {last_active.upper()}")
            return Command(
                goto=f"manager_{last_active}",
                update={
                    "routing_reason": "Continuing with last active manager",
                    "current_agent": last_active,
                }
            )

        # 첫 턴: Router LLM 호출
        last_message = state["messages"][-1].content

        print(f"[🔀] Router analyzing request (first turn)...")

        # config에서 callbacks 추출
        callbacks = config.get("callbacks", []) if config else []
        router_config = {"callbacks": callbacks} if callbacks else {}

        # structured output으로 라우팅 결정
        routing_agent = self.router_llm.with_structured_output(AgentRouting)
        routing = routing_agent.invoke(
            [
                {"role": "system", "content": self.router_prompt},
                {"role": "user", "content": last_message}
            ],
            config=router_config
        )

        print(f"[🔀] Routing to Manager {routing.target_agent.upper()}: {routing.reason}")

        # Command로 다음 노드 지정
        return Command(
            goto=f"manager_{routing.target_agent}",
            update={
                "routing_reason": routing.reason,
                "current_agent": routing.target_agent,
            }
        )

    def _manager_i_node(self, state: TeamHState, config: Optional[Dict[str, Any]] = None) -> Command:
        """Manager I 노드 - 전체 대화 맥락 포함"""
        print(f"[🏠] Manager I executing...")

        # config에서 callbacks 추출
        callbacks = config.get("callbacks", []) if config else []
        manager_config = {"callbacks": callbacks} if callbacks else {}

        # 전체 messages를 Manager I의 agent에 직접 전달
        # Manager agent는 checkpointer를 사용하지 않으므로 thread_id 불필요
        result = self.manager_i.agent.invoke(
            {"messages": state["messages"]},
            config=manager_config
        )

        # 마지막 AI 메시지 추출
        ai_response = self._extract_last_ai_message(result)

        # Handoff tool 호출 감지
        handoff_count = state.get("handoff_count", 0)
        handoff_target = self._detect_handoff(result)

        # 무한 루프 방지
        if handoff_count >= self.max_handoffs:
            print(f"[⚠️] Max handoffs reached ({self.max_handoffs}), ending conversation")
            next_agent = "end"
        elif handoff_target:
            print(f"[🤝] Handoff tool detected: Manager I → Manager {handoff_target.upper()}")
            next_agent = handoff_target
        else:
            # Handoff tool이 호출되지 않았으면 종료
            next_agent = "end"

        # 다음 노드 결정
        if next_agent == "end":
            goto = END
        else:
            goto = f"manager_{next_agent}"

        # last_active_manager 업데이트
        # Handoff가 발생하면 handoff_target으로, 아니면 현재 Manager (i)로 설정
        last_active = next_agent if next_agent != "end" else "i"

        # Command로 반환
        return Command(
            goto=goto,
            update={
                "messages": [AIMessage(content=ai_response)],
                "handoff_count": handoff_count + (1 if next_agent != "end" else 0),
                "current_agent": "i",
                "last_active_manager": last_active,
            }
        )

    def _manager_m_node(self, state: TeamHState, config: Optional[Dict[str, Any]] = None) -> Command:
        """Manager M 노드 - 전체 대화 맥락 포함"""
        print(f"[🧠] Manager M executing...")

        user_id = state.get("user_id", "default_user")

        # 전체 messages를 복사하고, 마지막 user 메시지에 user_id 주입
        messages = list(state["messages"])  # 복사
        if messages and len(messages) > 0:
            # 마지막 Human 메시지를 찾아서 user_id 주입
            for i in range(len(messages) - 1, -1, -1):
                msg = messages[i]
                if isinstance(msg, HumanMessage) or (hasattr(msg, "type") and msg.type == "human"):
                    # user_id를 메시지에 주입 (ManagerM의 _prepare_message와 동일)
                    messages[i] = HumanMessage(content=f"[User ID: {user_id}]\n{msg.content}")
                    break

        # config에서 callbacks 추출
        callbacks = config.get("callbacks", []) if config else []
        manager_config = {
            "recursion_limit": 20,  # 재귀 제한을 20으로 설정 (기본값 25)
            "callbacks": callbacks,
        }

        # 전체 messages를 Manager M의 agent에 직접 전달
        # Manager agent는 checkpointer를 사용하지 않으므로 thread_id 불필요
        result = self.manager_m.agent.invoke(
            {"messages": messages},
            config=manager_config
        )

        # 마지막 AI 메시지 추출
        ai_response = self._extract_last_ai_message(result)

        # Handoff tool 호출 감지
        handoff_count = state.get("handoff_count", 0)
        handoff_target = self._detect_handoff(result)

        # 무한 루프 방지
        if handoff_count >= self.max_handoffs:
            print(f"[⚠️] Max handoffs reached ({self.max_handoffs}), ending conversation")
            next_agent = "end"
        elif handoff_target:
            print(f"[🤝] Handoff tool detected: Manager M → Manager {handoff_target.upper()}")
            next_agent = handoff_target
        else:
            # Handoff tool이 호출되지 않았으면 종료
            next_agent = "end"

        # 다음 노드 결정
        if next_agent == "end":
            goto = END
        else:
            goto = f"manager_{next_agent}"

        # last_active_manager 업데이트
        # Handoff가 발생하면 handoff_target으로, 아니면 현재 Manager (m)로 설정
        last_active = next_agent if next_agent != "end" else "m"

        # Command로 반환
        return Command(
            goto=goto,
            update={
                "messages": [AIMessage(content=ai_response)],
                "handoff_count": handoff_count + (1 if next_agent != "end" else 0),
                "current_agent": "m",
                "last_active_manager": last_active,
            }
        )

    def _manager_s_node(self, state: TeamHState, config: Optional[Dict[str, Any]] = None) -> Command:
        """Manager S 노드 - 전체 대화 맥락 포함"""
        print(f"[🔍] Manager S executing...")

        # config에서 callbacks 추출
        callbacks = config.get("callbacks", []) if config else []
        manager_config = {"callbacks": callbacks} if callbacks else {}

        # 전체 messages를 Manager S의 agent에 직접 전달
        # Manager agent는 checkpointer를 사용하지 않으므로 thread_id 불필요
        result = self.manager_s.agent.invoke(
            {"messages": state["messages"]},
            config=manager_config
        )

        # 마지막 AI 메시지 추출
        ai_response = self._extract_last_ai_message(result)

        # Handoff tool 호출 감지
        handoff_count = state.get("handoff_count", 0)
        handoff_target = self._detect_handoff(result)

        # 무한 루프 방지
        if handoff_count >= self.max_handoffs:
            print(f"[⚠️] Max handoffs reached ({self.max_handoffs}), ending conversation")
            next_agent = "end"
        elif handoff_target:
            print(f"[🤝] Handoff tool detected: Manager S → Manager {handoff_target.upper()}")
            next_agent = handoff_target
        else:
            # Handoff tool이 호출되지 않았으면 종료
            next_agent = "end"

        # 다음 노드 결정
        if next_agent == "end":
            goto = END
        else:
            goto = f"manager_{next_agent}"

        # last_active_manager 업데이트
        # Handoff가 발생하면 handoff_target으로, 아니면 현재 Manager (s)로 설정
        last_active = next_agent if next_agent != "end" else "s"

        # Command로 반환
        return Command(
            goto=goto,
            update={
                "messages": [AIMessage(content=ai_response)],
                "handoff_count": handoff_count + (1 if next_agent != "end" else 0),
                "current_agent": "s",
                "last_active_manager": last_active,
            }
        )

    def _manager_t_node(self, state: TeamHState, config: Optional[Dict[str, Any]] = None) -> Command:
        """Manager T 노드 - 전체 대화 맥락 포함"""
        print(f"[📅] Manager T executing...")

        # config에서 callbacks 추출
        callbacks = config.get("callbacks", []) if config else []
        manager_config = {"callbacks": callbacks} if callbacks else {}

        # 전체 messages를 Manager T의 agent에 직접 전달
        # Manager agent는 checkpointer를 사용하지 않으므로 thread_id 불필요
        result = self.manager_t.agent.invoke(
            {"messages": state["messages"]},
            config=manager_config
        )

        # 마지막 AI 메시지 추출
        ai_response = self._extract_last_ai_message(result)

        # Handoff tool 호출 감지
        handoff_count = state.get("handoff_count", 0)
        handoff_target = self._detect_handoff(result)

        # 무한 루프 방지
        if handoff_count >= self.max_handoffs:
            print(f"[⚠️] Max handoffs reached ({self.max_handoffs}), ending conversation")
            next_agent = "end"
        elif handoff_target:
            print(f"[🤝] Handoff tool detected: Manager T → Manager {handoff_target.upper()}")
            next_agent = handoff_target
        else:
            # Handoff tool이 호출되지 않았으면 종료
            next_agent = "end"

        # 다음 노드 결정
        if next_agent == "end":
            goto = END
        else:
            goto = f"manager_{next_agent}"

        # last_active_manager 업데이트
        # Handoff가 발생하면 handoff_target으로, 아니면 현재 Manager (t)로 설정
        last_active = next_agent if next_agent != "end" else "t"

        # Command로 반환
        return Command(
            goto=goto,
            update={
                "messages": [AIMessage(content=ai_response)],
                "handoff_count": handoff_count + (1 if next_agent != "end" else 0),
                "current_agent": "t",
                "last_active_manager": last_active,
            }
        )

    # ========================================================================
    # 헬퍼 함수들
    # ========================================================================

    def _detect_handoff(self, result: Dict[str, Any]) -> Optional[str]:
        """
        결과에서 handoff tool 호출 감지

        Args:
            result: Manager agent의 실행 결과

        Returns:
            handoff 대상 agent ID ("i", "m", "s", "t") 또는 None
        """
        messages = result.get("messages", [])

        # 역순으로 확인 (최근 메시지부터)
        for msg in reversed(messages):
            # ToolMessage 확인
            if hasattr(msg, "type") and msg.type == "tool":
                content = str(msg.content)
                if "[HANDOFF_TO_I]" in content:
                    return "i"
                elif "[HANDOFF_TO_M]" in content:
                    return "m"
                elif "[HANDOFF_TO_S]" in content:
                    return "s"
                elif "[HANDOFF_TO_T]" in content:
                    return "t"

        return None

    def _extract_last_ai_message(self, result: Dict[str, Any]) -> str:
        """결과에서 마지막 AI 메시지 추출"""
        messages = result.get("messages", [])

        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                return msg.content
            elif hasattr(msg, "type") and msg.type == "ai":
                return msg.content

        return "No response from agent"

    # ========================================================================
    # 외부 인터페이스
    # ========================================================================

    @observe(name="team-h-graph-invoke", capture_input=True, capture_output=True)
    def invoke(
        self,
        message: str,
        user_id: str = "default_user",
        thread_id: str = "default",
        callbacks: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        그래프 실행

        Args:
            message: 사용자 메시지
            user_id: 사용자 ID
            thread_id: 스레드 ID
            callbacks: Langfuse CallbackHandler 등의 콜백 리스트

        Returns:
            최종 상태
        """
        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": callbacks or [],
        }

        initial_state = {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
            "handoff_count": 0,
        }

        result = self.graph.invoke(initial_state, config)
        return result

    def invoke_command(
        self,
        command: Command,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Command를 사용하여 그래프 재개 (HITL 지원)

        Args:
            command: LangGraph Command 객체 (resume 등)
            config: 그래프 설정 (thread_id 포함)

        Returns:
            그래프 실행 결과
        """
        result = self.graph.invoke(command, config)
        return result

    @observe(name="team-h-graph-stream", capture_input=True, capture_output=True)
    def stream(
        self,
        message: str,
        user_id: str = "default_user",
        thread_id: str = "default",
        callbacks: Optional[List] = None,
    ):
        """
        그래프 스트리밍 실행

        Args:
            message: 사용자 메시지
            user_id: 사용자 ID
            thread_id: 스레드 ID
            callbacks: Langfuse CallbackHandler 등의 콜백 리스트

        Yields:
            각 노드 실행 결과
        """
        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": callbacks or [],
        }

        initial_state = {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
            "handoff_count": 0,
        }

        for chunk in self.graph.stream(initial_state, config):
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


# ============================================================================
# 헬퍼 함수
# ============================================================================

def create_team_h_graph(**kwargs) -> TeamHGraph:
    """Team-H Graph 생성 헬퍼 함수"""
    return TeamHGraph(**kwargs)


# 싱글톤 (선택적)
_team_h_graph_instance = None


def get_team_h_graph(**kwargs) -> TeamHGraph:
    """Team-H Graph 싱글톤 인스턴스"""
    global _team_h_graph_instance
    if _team_h_graph_instance is None:
        _team_h_graph_instance = TeamHGraph(**kwargs)
    return _team_h_graph_instance