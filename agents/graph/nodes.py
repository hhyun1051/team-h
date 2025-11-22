import logging
from typing import Dict, Any, Optional, List, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from agents.graph.state import TeamHState, AgentRouting

# 로거 설정
logger = logging.getLogger("TeamHGraph")

def router_node(
    state: TeamHState,
    config: Optional[Dict[str, Any]] = None,
    *,
    router_chain: Any = None
) -> Dict[str, Any]:
    """
    라우터 노드 - 초기 라우팅 결정 (첫 턴) 또는 last_active_manager 사용
    
    Args:
        state: 현재 상태
        config: 실행 설정
        router_chain: (Dependency Injection) 라우터 LLM 체인
    """
    logger.info(f"--- Router Node ---")
    messages = state.messages
    last_active = state.last_active_manager
    
    # 1. 마지막 활성 매니저가 있으면 거기로 바로 보냄 (Sticky Routing)
    # 단, 명시적으로 router로 돌아온 경우(handoff_count > 0 등)는 제외할 수도 있으나,
    # 여기서는 "대화의 연속성"을 위해 우선 last_active를 확인.
    # (만약 last_active가 일을 끝내고 router로 보냈다면 state.next_agent가 router일 것임)
    
    # 하지만 router_node에 진입했다는 것은, 누군가 next_agent="router"로 보냈거나,
    # 그래프 시작점(START)에서 왔다는 뜻.
    
    # START에서 온 경우 (messages의 마지막이 HumanMessage)
    if messages and isinstance(messages[-1], HumanMessage):
        # 이전에 대화하던 매니저가 있다면 우선권 부여 (Sticky)
        if last_active:
            logger.info(f"Sticky routing to last active manager: {last_active}")
            return {"next_agent": last_active, "routing_reason": "Sticky routing"}

    # 2. 그 외의 경우 (또는 Sticky가 없는 경우) 라우팅 결정
    if router_chain is None:
        # Fallback if chain not provided
        logger.warning("Router chain not provided, defaulting to Manager M")
        return {"next_agent": "m", "routing_reason": "Default fallback"}

    try:
        # 마지막 사용자 메시지 추출
        last_user_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        if not last_user_msg:
            return {"next_agent": "end", "routing_reason": "No user message found"}

        decision: AgentRouting = router_chain.invoke({"messages": messages})
        
        logger.info(f"Router decision: {decision.target_agent} (Reason: {decision.reason})")
        
        return {
            "next_agent": decision.target_agent,
            "routing_reason": decision.reason
        }
        
    except Exception as e:
        logger.error(f"Router error: {e}")
        # 에러 시 안전하게 메모리 매니저나 종료로 이동
        return {"next_agent": "m", "routing_reason": f"Error fallback: {str(e)}"}


def execute_manager_node(
    state: TeamHState,
    config: Optional[Dict[str, Any]],
    manager_instance: Any,
    manager_key: str,
    messages: Optional[List] = None,
    recursion_limit: Optional[int] = None
) -> Command:
    """
    Generic manager node execution logic
    """
    logger.info(f"--- Manager {manager_key.upper()} Node ---")
    
    # 1. Manager 실행
    # Manager의 invoke는 (state, config)를 받거나 messages를 받음.
    # 기존 코드에서는 manager_instance.invoke(state, config) 형태였음.
    
    # Manager 인스턴스가 LangChain Runnable이라고 가정
    # 필요한 경우 input을 조정
    
    # 기존 로직: result = await manager.ainvoke(...) or manager.invoke(...)
    # 여기서는 동기 invoke 사용 가정 (비동기 필요시 async def로 변경)
    
    # Manager에게 전달할 입력 구성
    # TeamHState를 그대로 전달하거나, messages만 전달
    # Manager 구현에 따라 다름. 여기서는 state를 전달한다고 가정.
    
    try:
        # config에 recursion_limit 추가
        run_config = config.copy() if config else {}
        if recursion_limit:
            run_config["recursion_limit"] = recursion_limit
            
        result = manager_instance.invoke(state, run_config)
        
        # 2. 결과 처리 및 Handoff 감지
        # result는 {"messages": [...]} 형태라고 가정
        
        original_msg_count = len(state.messages)
        handoff_target = detect_handoff(result, original_msg_count)
        
        update_dict = {
            "messages": result.get("messages", []),
            "current_agent": manager_key,
            "last_active_manager": manager_key # 현재 매니저를 마지막 활성 매니저로 기록
        }
        
        if handoff_target:
            logger.info(f"Handoff detected from {manager_key} to {handoff_target}")
            update_dict["next_agent"] = handoff_target
            update_dict["handoff_count"] = state.handoff_count + 1
            
            # Goto other agent
            return Command(
                update=update_dict,
                goto=handoff_target
            )
        else:
            # No handoff, stay or end?
            # 보통 Manager가 응답을 완료하면 END로 가거나, 
            # 사용자의 입력을 기다리기 위해 END(interrupt)로 감.
            # LangGraph에서 node가 끝나면 다음 엣지를 따라감.
            # 여기서는 Command를 사용하여 명시적으로 제어 가능.
            
            # 만약 툴 호출이 포함되어 있다면? -> Manager 내부에서 처리됨 (prebuilt agent라면)
            # 하지만 여기서는 Manager가 "에이전트" 자체이므로, 
            # 실행이 끝나면 "응답 완료" 상태임.
            
            return Command(
                update=update_dict,
                goto="__end__" # 대화 턴 종료 (사용자 입력 대기)
            )
            
    except Exception as e:
        logger.error(f"Error in Manager {manager_key}: {e}")
        return Command(
            update={
                "messages": [AIMessage(content=f"Error in Manager {manager_key}: {str(e)}")]
            },
            goto="__end__"
        )

def detect_handoff(result: Dict[str, Any], original_msg_count: int) -> Optional[str]:
    """
    결과에서 handoff tool 호출 감지 (새로 생성된 메시지만 검사)
    
    Args:
        result: Manager agent의 실행 결과
        original_msg_count: 실행 전 메시지 개수
        
    Returns:
        handoff 대상 agent ID ("i", "m", "s", "t") 또는 None
    """
    messages = result.get("messages", [])
    if not messages:
        return None
        
    # 새로 추가된 메시지만 검사
    # (참고: LangGraph가 state를 업데이트하기 전의 result일 수 있으므로
    #  result["messages"]가 전체인지, 추가분인지 확인 필요.
    #  보통 invoke 결과는 전체 메시지일 수 있음.
    #  여기서는 안전하게 뒤에서부터 검사)
    
    # 만약 result["messages"]가 전체 리스트라면:
    new_messages = messages[original_msg_count:] if len(messages) > original_msg_count else messages
    
    for msg in reversed(new_messages):
        if isinstance(msg, ToolMessage):
            # ToolMessage는 툴 실행 결과임. 
            # Handoff 툴이 실행되었다면, 그 결과(ToolMessage)가 있을 것임.
            # 하지만 Handoff는 "툴 호출(AIMessage의 tool_calls)"을 보고 판단하는게 아니라,
            # "툴 실행 결과"를 보고 판단해야 하나? 
            # 아니면 툴 실행 자체가 "goto"를 유발해야 하나?
            
            # 기존 로직: 툴 이름으로 감지
            if msg.name and msg.name.startswith("handoff_to_manager_"):
                target = msg.name.split("_")[-1] # i, m, s, t
                return target
                
    return None

def extract_last_ai_message(result: Dict[str, Any]) -> Optional[str]:
    """결과에서 마지막 AI 메시지 추출"""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)
    return None

# --- Wrapper Nodes for Specific Managers ---
# These will be bound with partial in the graph builder

def manager_i_node(state: TeamHState, config: Optional[Dict[str, Any]] = None, *, manager: Any) -> Command:
    return execute_manager_node(state, config, manager, "i")

def manager_m_node(state: TeamHState, config: Optional[Dict[str, Any]] = None, *, manager: Any) -> Command:
    # Manager M might need user_id injection
    if config and "configurable" in config:
        config["configurable"]["user_id"] = state.user_id
    return execute_manager_node(state, config, manager, "m", recursion_limit=10)

def manager_s_node(state: TeamHState, config: Optional[Dict[str, Any]] = None, *, manager: Any) -> Command:
    return execute_manager_node(state, config, manager, "s")

def manager_t_node(state: TeamHState, config: Optional[Dict[str, Any]] = None, *, manager: Any) -> Command:
    return execute_manager_node(state, config, manager, "t")
