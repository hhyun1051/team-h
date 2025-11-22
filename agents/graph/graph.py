import os
import logging
from typing import Dict, Any, Optional, List, Literal
from functools import partial

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command
from psycopg_pool import ConnectionPool

# Langfuse
from langfuse.langchain import CallbackHandler

# Config
from config.settings import settings

# Agents import
from agents import ManagerI, ManagerM, ManagerS, ManagerT
from agents.middlewares import LangfuseToolLoggingMiddleware, ToolErrorHandlerMiddleware

# Internal imports
from agents.graph.state import TeamHState, AgentRouting
from agents.graph.nodes import (
    router_node,
    manager_i_node,
    manager_m_node,
    manager_s_node,
    manager_t_node
)

# 로거 설정
logger = logging.getLogger("TeamHGraph")

class TeamHGraph:
    """
    LangGraph 기반 Team-H 에이전트 시스템
    """
    def __init__(
        self,
        # Manager activation flags
        enable_manager_i: bool = True,
        enable_manager_m: bool = True,
        enable_manager_s: bool = True,
        enable_manager_t: bool = True,
        
        # LLM configs
        model_name: str = "gpt-4o",
        temperature: float = 0.7,
        max_handoffs: int = 5,
    ):
        self.enable_manager_i = enable_manager_i
        self.enable_manager_m = enable_manager_m
        self.enable_manager_s = enable_manager_s
        self.enable_manager_t = enable_manager_t
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_handoffs = max_handoffs
        
        # Components
        self.langfuse_handler = None
        self.checkpointer = None
        self.router_chain = None
        
        # Managers
        self.manager_i = None
        self.manager_m = None
        self.manager_s = None
        self.manager_t = None
        
        # Handoff Tools
        self.handoff_tools = []
        
        # Graph
        self.graph = None
        
        # Initialization
        self._init_langfuse()
        self._init_postgres_checkpoint()
        self._init_router_llm()
        self._create_handoff_tools() # Create tools first
        self._init_managers()        # Then init managers with tools
        self._build_graph()

    def _init_langfuse(self):
        """Langfuse 초기화"""
        # settings.langfuse 사용
        if settings.langfuse.langfuse_public_key and settings.langfuse.langfuse_secret_key:
            # CallbackHandler가 환경변수를 우선적으로 사용하는 경우가 많으므로
            # 명시적으로 환경변수 설정 (또는 인자 전달 방식 확인 필요)
            os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse.langfuse_public_key
            os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse.langfuse_secret_key
            os.environ["LANGFUSE_HOST"] = settings.langfuse.langfuse_base_url
            
            self.langfuse_handler = CallbackHandler()
            logger.info("Langfuse initialized")
        else:
            logger.warning("Langfuse credentials not found in settings")

    def _init_postgres_checkpoint(self):
        """PostgreSQL checkpoint 초기화"""
        # settings.database 사용
        # use_postgres_checkpoint 플래그는 여전히 유효하지만, 
        # connection string은 settings에서 가져옴
        if settings.database.database_url:
            try:
                pool = ConnectionPool(conninfo=settings.database.database_url)
                self.checkpointer = PostgresSaver(pool)
                # 테이블 생성 (필요시)
                self.checkpointer.setup()
                logger.info("PostgreSQL checkpointer initialized")
            except Exception as e:
                logger.error(f"Failed to init Postgres checkpointer: {e}")
                self.checkpointer = MemorySaver()
        else:
            self.checkpointer = MemorySaver()
            logger.info("Using MemorySaver checkpointer (No DB URL)")

    def _init_router_llm(self):
        """라우터 LLM 초기화"""
        # settings.api.openai_api_key 사용 (LangChain이 자동 감지하지만 명시적으로 전달 가능)
        llm = ChatOpenAI(
            model=self.model_name,
            temperature=0,
            api_key=settings.api.openai_api_key
        )
        
        system_prompt = """You are the Router Agent for Team-H.
Your job is to route the user's request to the most appropriate specialist agent.

Available Agents:
- 'i' (IoT Manager): Controls smart home devices (lights, plugs, speakers).
- 'm' (Memory Manager): Manages user memory, preferences, and long-term context.
- 's' (Search Manager): Searches the web for real-time information.
- 't' (Time Manager): Manages calendar, schedule, and time-based tasks.

Routing Rules:
1. Analyze the user's latest message and the conversation history.
2. Select ONE agent that is best suited to handle the request.
3. Provide a brief reason for your choice.

Output Format:
Return a JSON object matching the AgentRouting schema.
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{messages}"),
        ])
        
        self.router_chain = prompt | llm.with_structured_output(AgentRouting)

    def _create_handoff_tools(self):
        """각 Manager로 handoff하는 tool 생성"""
        # 여기서는 간단하게 툴 정의만 함. 실제 로직은 Graph의 노드 전환으로 처리됨.
        # 하지만 Manager들이 이 툴을 "호출"해야 하므로, 툴 객체가 필요함.
        # LangChain tool 데코레이터 사용
        from langchain_core.tools import tool

        @tool
        def handoff_to_manager_i(reason: str):
            """Hand off to Manager I (IoT). Use for device control."""
            return f"Handoff to Manager I initiated. Reason: {reason}"

        @tool
        def handoff_to_manager_m(reason: str):
            """Hand off to Manager M (Memory). Use for remembering/recalling info."""
            return f"Handoff to Manager M initiated. Reason: {reason}"

        @tool
        def handoff_to_manager_s(reason: str):
            """Hand off to Manager S (Search). Use for web search."""
            return f"Handoff to Manager S initiated. Reason: {reason}"

        @tool
        def handoff_to_manager_t(reason: str):
            """Hand off to Manager T (Time). Use for calendar/scheduling."""
            return f"Handoff to Manager T initiated. Reason: {reason}"

        self.handoff_tools = []
        if self.enable_manager_i: self.handoff_tools.append(handoff_to_manager_i)
        if self.enable_manager_m: self.handoff_tools.append(handoff_to_manager_m)
        if self.enable_manager_s: self.handoff_tools.append(handoff_to_manager_s)
        if self.enable_manager_t: self.handoff_tools.append(handoff_to_manager_t)

    def _init_managers(self):
        """각 Manager 초기화"""
        # 공통 미들웨어
        middlewares = []
        if self.langfuse_handler:
            middlewares.append(LangfuseToolLoggingMiddleware(langfuse_handler=self.langfuse_handler))
        middlewares.append(ToolErrorHandlerMiddleware())

        if self.enable_manager_i:
            self.manager_i = ManagerI(
                model_name=self.model_name,
                temperature=self.temperature,
                tools=self.handoff_tools, # 다른 매니저로 handoff 가능
                middlewares=middlewares
            )
        
        if self.enable_manager_m:
            self.manager_m = ManagerM(
                model_name=self.model_name,
                temperature=self.temperature,
                tools=self.handoff_tools,
                middlewares=middlewares
            )

        if self.enable_manager_s:
            self.manager_s = ManagerS(
                model_name=self.model_name,
                temperature=self.temperature,
                tools=self.handoff_tools,
                middlewares=middlewares
            )

        if self.enable_manager_t:
            self.manager_t = ManagerT(
                model_name=self.model_name,
                temperature=self.temperature,
                tools=self.handoff_tools,
                middlewares=middlewares
            )

    def _build_graph(self):
        """그래프 빌드"""
        workflow = StateGraph(TeamHState)

        # 1. Add Nodes
        # Router Node (Partial binding)
        workflow.add_node("router", partial(router_node, router_chain=self.router_chain))

        # Manager Nodes
        if self.enable_manager_i and self.manager_i:
            workflow.add_node("i", partial(manager_i_node, manager=self.manager_i))
        if self.enable_manager_m and self.manager_m:
            workflow.add_node("m", partial(manager_m_node, manager=self.manager_m))
        if self.enable_manager_s and self.manager_s:
            workflow.add_node("s", partial(manager_s_node, manager=self.manager_s))
        if self.enable_manager_t and self.manager_t:
            workflow.add_node("t", partial(manager_t_node, manager=self.manager_t))

        # 2. Add Edges
        # Start -> Router
        workflow.add_edge(START, "router")

        # Router -> Managers (Conditional)
        def route_after_router(state: TeamHState):
            return state.next_agent

        mapping = {"end": END}
        if self.enable_manager_i: mapping["i"] = "i"
        if self.enable_manager_m: mapping["m"] = "m"
        if self.enable_manager_s: mapping["s"] = "s"
        if self.enable_manager_t: mapping["t"] = "t"

        workflow.add_conditional_edges(
            "router",
            route_after_router,
            mapping
        )

        # Managers -> End (or Router? In this design, managers return Command(goto=...))
        # Nodes that return Command do not need explicit edges if they handle all transitions.
        # But if they return normal dict updates, we need edges.
        # Our nodes.py returns Command, so we don't strictly need edges from managers 
        # UNLESS they fall through.
        # But LangGraph requires nodes to be connected if they don't use Command for everything.
        # Let's assume Command covers it.

        # 3. Compile
        self.graph = workflow.compile(checkpointer=self.checkpointer)

    def invoke(
        self,
        message: str,
        user_id: str = "default_user",
        thread_id: str = "default",
        session_id: Optional[str] = None,
        callbacks: Optional[List] = None,
    ) -> Dict[str, Any]:
        """그래프 실행"""
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
        if callbacks:
            config["callbacks"] = callbacks
        elif self.langfuse_handler:
            config["callbacks"] = [self.langfuse_handler]

        # 입력 상태 구성
        input_state = {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
            "handoff_count": 0
        }

        return self.graph.invoke(input_state, config)

    def stream(
        self,
        message: str,
        user_id: str = "default_user",
        thread_id: str = "default",
        session_id: Optional[str] = None,
        callbacks: Optional[List] = None,
    ):
        """그래프 스트리밍 실행"""
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
        if callbacks:
            config["callbacks"] = callbacks
        elif self.langfuse_handler:
            config["callbacks"] = [self.langfuse_handler]

        input_state = {
            "messages": [HumanMessage(content=message)],
            "user_id": user_id,
            "handoff_count": 0
        }

        return self.graph.stream(input_state, config, stream_mode="values")

    def invoke_command(self, command: Command, config: Dict[str, Any]):
        """Command를 사용하여 그래프 재개 (HITL 지원)"""
        return self.graph.invoke(command, config)

    def get_graph_visualization(self):
        """Mermaid 다이어그램 반환"""
        return self.graph.get_graph().draw_mermaid()
