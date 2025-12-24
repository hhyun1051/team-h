"""
FastAPI Backend for Team-H Graph

핵심 개선:
1. TeamHGraph 인스턴스는 앱 시작 시 한 번만 생성 (lifespan)
2. PostgreSQL checkpointer를 통해 thread_id 기반으로 상태 자동 복원
3. 각 요청은 thread_id만 전달하여 기존 대화 재개
4. astream_events()를 사용한 실시간 토큰 스트리밍
5. Human-in-the-Loop 인터럽트 처리
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
import os
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Project root setup
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Load .env
load_dotenv(project_root / ".env")

# Import agents
from agents.graph import TeamHGraph
from langgraph.types import Command

# Import Langfuse
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler

# Import models
try:
    from .models import ChatRequest, ResumeRequest, InterruptResponse, StateResponse
except ImportError:
    # 직접 실행 시 (python main.py)
    from models import ChatRequest, ResumeRequest, InterruptResponse, StateResponse


# ============================================================================
# Global Agent Instance (initialized once at startup)
# ============================================================================

_agent: Optional[TeamHGraph] = None


def get_agent() -> TeamHGraph:
    """전역 agent 인스턴스 반환"""
    if _agent is None:
        raise RuntimeError("Agent not initialized. This should not happen.")
    return _agent


# ============================================================================
# Application Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 라이프사이클 관리

    앱 시작 시 TeamHGraph를 한 번만 생성하고,
    모든 요청에서 재사용합니다.
    """
    global _agent

    # Startup: Langfuse singleton 초기화 (middleware가 사용)
    print("[🔧] Initializing Langfuse singleton...")
    Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_BASE_URL", "http://localhost:3000"),
    )
    print("[✅] Langfuse singleton initialized")

    # Startup: Agent 한 번만 생성
    print("[🚀] Initializing TeamHGraph (once)...")

    _agent = TeamHGraph(
        # Manager 활성화 (환경 변수 기반)
        enable_manager_i=bool(os.getenv("HOMEASSISTANT_TOKEN")),
        enable_manager_m=True,
        enable_manager_s=bool(os.getenv("TAVILY_API_KEY")),
        enable_manager_t=bool(os.getenv("GOOGLE_CALENDAR_CREDENTIALS_PATH")),

        # Manager I 설정
        homeassistant_url=os.getenv("HOMEASSISTANT_URL", "http://localhost:8124"),
        homeassistant_token=os.getenv("HOMEASSISTANT_TOKEN"),

        # Manager M 설정
        embedding_type=os.getenv("EMBEDDING_TYPE", "openai"),
        embedder_url=os.getenv("EMBEDDER_URL"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        embedding_dims=int(os.getenv("OPENAI_EMBEDDING_DIMS", "3072")) if os.getenv("EMBEDDING_TYPE") == "openai" else int(os.getenv("FASTAPI_EMBEDDING_DIMS", "1024")),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=os.getenv("QDRANT_PASSWORD"),
        m_collection_name=os.getenv("MANAGER_M_COLLECTION", "manager_m_memories"),

        # Manager S 설정
        tavily_api_key=os.getenv("TAVILY_API_KEY"),

        # Manager T 설정
        google_credentials_path=os.getenv("GOOGLE_CALENDAR_CREDENTIALS_PATH"),
        google_token_path=os.getenv("GOOGLE_CALENDAR_TOKEN_PATH"),

        # LLM 설정 (.env의 LLM_MODEL_NAME, LLM_TEMPERATURE 사용)
        model_name=os.getenv("LLM_MODEL_NAME", "gpt-4o-mini"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),

        # PostgreSQL checkpoint (자동 상태 저장/복원)
        postgres_connection_string=os.getenv("POSTGRES_CONNECTION_STRING"),
        use_postgres_checkpoint=True,
    )

    # AsyncPostgresSaver의 테이블 초기화 (비동기)
    if hasattr(_agent.checkpointer, 'setup'):
        print("[🔧] Setting up PostgreSQL checkpoint tables...")
        await _agent.checkpointer.setup()
        print("[✅] PostgreSQL checkpoint tables ready")

    print("[✅] TeamHGraph initialized successfully")

    yield

    # Shutdown
    print("[👋] FastAPI server shutting down...")


app = FastAPI(
    title="Team-H Graph API",
    description="LangGraph-based multi-agent system with streaming and HITL",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS 설정 (Streamlit에서 접근 가능하도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 origin으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# SSE Streaming Helper
# ============================================================================

async def generate_sse_stream(
    agent: TeamHGraph,
    config: Dict[str, Any],
    input_data: Any,
    context: Any = None,  # TeamHContext 전달
):
    """
    SSE (Server-Sent Events) 스트림 생성

    LangGraph의 astream_events()를 사용하여 모든 이벤트를 스트리밍:
    - on_chat_model_start: LLM 호출 시작
    - on_chat_model_stream: LLM 토큰 스트리밍 (실시간)
    - on_chat_model_end: LLM 호출 완료
    - on_tool_start/end: 툴 실행 상태
    - on_chain_start/end: 노드 실행 상태
    - interrupt: HITL 인터럽트

    참고: docs/langchain_models.md, docs/langgraph_streaming.md

    Args:
        agent: TeamHGraph 인스턴스 (전역 인스턴스 재사용)
        config: LangGraph config (thread_id 포함)
        input_data: 초기 입력 또는 Command
    """
    try:
        # 스트리밍 시작 전에 현재 상태 조회 (어떤 manager가 활성화되어 있는지 확인)
        initial_snapshot = await agent.graph.aget_state(config)
        current_manager = initial_snapshot.values.get("current_agent") or initial_snapshot.values.get("last_active_manager")

        # 초기 manager 정보 전송
        if current_manager:
            init_data = {
                "event": "agent_start",
                "current_agent": current_manager,
            }
            yield f"data: {json.dumps(init_data, ensure_ascii=False)}\n\n"

        # astream_events로 모든 이벤트 스트리밍
        # context 전달: tools의 runtime.context로 접근 가능
        async for event in agent.graph.astream_events(
            input_data,
            config,
            version="v2",  # v2는 더 상세한 이벤트 제공
            context=context,  # TeamHContext 전달
        ):
            event_type = event.get("event")
            event_name = event.get("name", "")
            event_data = event.get("data", {})
            metadata = event.get("metadata", {})

            # SSE 형식으로 전송할 데이터
            sse_data = {
                "type": event_type,
                "name": event_name,
            }

            # ===== LLM 호출 시작 =====
            if event_type == "on_chat_model_start":
                sse_data["event"] = "llm_start"
                sse_data["node"] = metadata.get("langgraph_node", "unknown")
                yield f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n"

            # ===== LLM 토큰 스트리밍 (실시간) =====
            elif event_type == "on_chat_model_stream":
                chunk = event_data.get("chunk", {})
                if not (hasattr(chunk, "content") and chunk.content):
                    continue  # 내용 없는 청크는 무시

                # 라우터 노드의 LLM 스트리밍은 필터링 (router_decision 이벤트로 대체)
                langgraph_node = metadata.get("langgraph_node", "")

                # 디버그: 라우터 관련 이벤트만 로그
                if "router" in langgraph_node.lower() or "router" in event_name.lower():
                    print(f"[DEBUG] Router stream - langgraph_node: '{langgraph_node}', event_name: '{event_name}', content: {chunk.content[:50] if chunk.content else 'None'}")

                if langgraph_node == "router":
                    print(f"[DEBUG] Router token FILTERED (not sent to client)")
                    continue  # 라우터 노드의 토큰은 무시

                # Manager 노드의 토큰만 전송
                print(f"[DEBUG] Sending token to client - node: '{langgraph_node}', content: {chunk.content[:30]}")
                sse_data["event"] = "token"
                sse_data["content"] = chunk.content
                # current_manager 정보 포함 (Streamlit에서 agent 아이콘 표시용)
                sse_data["current_agent"] = current_manager
                yield f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n"

            # ===== LLM 호출 완료 =====
            elif event_type == "on_chat_model_end":
                langgraph_node = metadata.get("langgraph_node", "")

                # 라우터 노드의 LLM 완료는 무시 (router_decision 이벤트로 대체)
                if langgraph_node == "router":
                    print(f"[DEBUG] Router llm_end FILTERED (not sent to client)")
                    continue

                output = event_data.get("output", {})
                sse_data["event"] = "llm_end"
                if hasattr(output, "content"):
                    sse_data["full_message"] = output.content
                else:
                    sse_data["full_message"] = str(output)
                sse_data["node"] = langgraph_node
                yield f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n"

            # ===== 툴 실행 시작 =====
            elif event_type == "on_tool_start":
                sse_data["event"] = "tool_start"
                sse_data["tool_name"] = event_name
                sse_data["tool_input"] = event_data.get("input", {})
                sse_data["node"] = metadata.get("langgraph_node", "unknown")
                yield f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n"

            # ===== 툴 실행 완료 =====
            elif event_type == "on_tool_end":
                sse_data["event"] = "tool_end"
                sse_data["tool_name"] = event_name
                sse_data["tool_output"] = str(event_data.get("output"))
                sse_data["node"] = metadata.get("langgraph_node", "unknown")
                yield f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n"

            # ===== 체인/노드 시작 =====
            elif event_type == "on_chain_start":
                # Manager 노드 진입 감지 및 current_manager 업데이트
                langgraph_node = metadata.get("langgraph_node", "")
                if langgraph_node.startswith("manager_"):
                    # manager_m -> m 변환
                    current_manager = langgraph_node.replace("manager_", "")
                    # Manager 변경 알림
                    manager_change_data = {
                        "event": "agent_change",
                        "current_agent": current_manager,
                    }
                    yield f"data: {json.dumps(manager_change_data, ensure_ascii=False)}\n\n"

                # 그래프 노드 시작 (router, manager_i, manager_m 등)
                sse_data["event"] = "node_start"
                sse_data["node_name"] = event_name
                yield f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n"

            # ===== 체인/노드 완료 =====
            elif event_type == "on_chain_end":
                langgraph_node = metadata.get("langgraph_node", "")
                output = event_data.get("output")

                # 디버그: 모든 chain_end 이벤트 로깅
                print(f"[DEBUG] on_chain_end - langgraph_node: '{langgraph_node}', event_name: '{event_name}', output_type: {type(output).__name__}, has_target_agent: {hasattr(output, 'target_agent') if output else False}")

                # 라우터 노드 완료 시 특별 처리 (AgentRouting 객체만)
                # RunnableSequence만 선택: RunnableLambda(내부), RunnableSequence(중간), router(최종 Command) 중 RunnableSequence에서만 emit
                if (langgraph_node == "router" and event_name == "RunnableSequence" and output and
                    hasattr(output, "target_agent") and hasattr(output, "reason")):
                    print(f"[DEBUG] Sending router_decision: target={output.target_agent}, reason={output.reason[:50]}")
                    router_data = {
                        "event": "router_decision",
                        "target_agent": output.target_agent,
                        "reason": output.reason,
                    }
                    yield f"data: {json.dumps(router_data, ensure_ascii=False)}\n\n"
                    # 라우터 결정은 node_end 이벤트를 전송하지 않음 (중복 방지)
                    continue

                # 일반 노드 완료 이벤트
                sse_data["event"] = "node_end"
                sse_data["node_name"] = event_name
                if output:
                    # 노드 결과 요약만 전송 (너무 크면 생략)
                    output_str = str(output)
                    if len(output_str) > 500:
                        output_str = output_str[:500] + "..."
                    sse_data["output"] = output_str
                yield f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n"

        # 스트리밍 완료 후 최종 상태 확인 (인터럽트 체크)
        snapshot = await agent.graph.aget_state(config)

        # 최종 상태에서 current_agent 정보 추출
        final_current_agent = snapshot.values.get("current_agent") or snapshot.values.get("last_active_manager")

        if snapshot.next:
            # 인터럽트 발생
            interrupts = []
            for task in snapshot.tasks:
                interrupts.extend(task.interrupts)

            if interrupts:
                interrupt_data = {
                    "event": "interrupt",
                    "type": "interrupt",
                    "interrupt": interrupts[0].value,
                    "thread_id": config["configurable"]["thread_id"],
                }
                yield f"data: {json.dumps(interrupt_data, ensure_ascii=False, default=str)}\n\n"
        else:
            # 정상 완료
            final_data = {
                "event": "done",
                "type": "done",
                "messages_count": len(snapshot.values.get("messages", [])),
                "current_agent": snapshot.values.get("current_agent"),
                "handoff_count": snapshot.values.get("handoff_count", 0),
            }
            yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"

    except Exception as e:
        import traceback
        error_data = {
            "event": "error",
            "type": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "ok",
        "service": "Team-H Graph API",
        "version": "2.0.0",
        "agent_initialized": _agent is not None,
    }


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    채팅 스트리밍 엔드포인트

    핵심:
    - Agent 재사용 (전역 인스턴스)
    - thread_id로 상태 자동 복원 (PostgreSQL checkpointer)
    - astream_events()로 실시간 토큰 스트리밍
    - Langfuse CallbackHandler로 전체 흐름 로깅

    Args:
        request: ChatRequest (message, thread_id, user_id, session_id)

    Returns:
        StreamingResponse: SSE 스트림
    """
    try:
        agent = get_agent()  # 전역 인스턴스 재사용

        session_id = request.session_id or request.thread_id

        # CallbackHandler 생성
        langfuse_handler = CallbackHandler()

        # Config (thread_id + callbacks + metadata)
        config = {
            "configurable": {
                "thread_id": request.thread_id,
            },
            "callbacks": [langfuse_handler],  # Langfuse 전체 흐름 로깅
            "metadata": {
                "langfuse_user_id": request.user_id,
                "langfuse_session_id": session_id,
                "langfuse_tags": ["team-h", "api", "streaming"],
            }
        }

        # Context 생성 (TeamHContext - tools의 runtime.context로 전달)
        from agents.context import TeamHContext
        context = TeamHContext(
            user_id=request.user_id,
            thread_id=request.thread_id,
            session_id=session_id,
        )

        # 초기 입력
        from langchain_core.messages import HumanMessage
        initial_state = {
            "messages": [HumanMessage(content=request.message)],
            "handoff_count": 0,
        }

        # SSE 스트림 반환
        return StreamingResponse(
            generate_sse_stream(agent, config, initial_state, context),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Nginx 버퍼링 비활성화
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/resume")
async def chat_resume(request: ResumeRequest):
    """
    HITL 재개 엔드포인트

    인터럽트된 대화를 사용자 결정(승인/거부/편집)을 기반으로 재개합니다.

    핵심:
    - 같은 agent 재사용 (전역 인스턴스)
    - thread_id로 상태 복원
    - Command(resume=...)로 인터럽트 재개
    - Langfuse CallbackHandler로 전체 흐름 로깅

    Args:
        request: ResumeRequest (thread_id, decisions, user_id, session_id)

    Returns:
        StreamingResponse: SSE 스트림
    """
    try:
        agent = get_agent()  # 전역 인스턴스 재사용

        session_id = request.session_id or request.thread_id

        # CallbackHandler 생성
        langfuse_handler = CallbackHandler()

        # Config (thread_id + callbacks + metadata)
        config = {
            "configurable": {
                "thread_id": request.thread_id,
            },
            "callbacks": [langfuse_handler],  # Langfuse 전체 흐름 로깅
            "metadata": {
                "langfuse_user_id": request.user_id,
                "langfuse_session_id": session_id,
                "langfuse_tags": ["team-h", "api", "resume"],
            }
        }

        # Context 생성 (TeamHContext - tools의 runtime.context로 전달)
        from agents.context import TeamHContext
        context = TeamHContext(
            user_id=request.user_id,
            thread_id=request.thread_id,
            session_id=session_id,
        )

        # Command 생성
        command = Command(resume={"decisions": request.decisions})

        # SSE 스트림 반환
        return StreamingResponse(
            generate_sse_stream(agent, config, command, context),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state/{thread_id}")
async def get_state(thread_id: str):
    """
    특정 thread의 현재 상태 조회

    Args:
        thread_id: 대화 스레드 ID

    Returns:
        StateResponse: 현재 그래프 상태, 다음 노드, 인터럽트 여부
    """
    try:
        agent = get_agent()
        config = {"configurable": {"thread_id": thread_id}}

        # 상태 조회
        snapshot = await agent.graph.aget_state(config)

        # 인터럽트 확인
        interrupts = []
        if snapshot.next:
            for task in snapshot.tasks:
                interrupts.extend(task.interrupts)

        return {
            "status": "success",
            "thread_id": thread_id,
            "state": snapshot.values,
            "next_nodes": snapshot.next,
            "has_interrupt": len(interrupts) > 0,
            "interrupts": [interrupt.value for interrupt in interrupts],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # 개발 서버 실행
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
