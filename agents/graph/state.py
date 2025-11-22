from typing import Annotated, Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages

# ============================================================================
# 1. State 정의
# ============================================================================
class TeamHState(BaseModel):
    """Team-H 그래프 상태"""
    messages: Annotated[list, add_messages]
    next_agent: Literal["router", "i", "m", "s", "t", "end"] = "router"
    user_id: str = "default_user"
    routing_reason: str = ""
    handoff_count: int = 0
    current_agent: Optional[str] = None
    last_active_manager: Optional[str] = None

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
