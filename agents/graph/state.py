"""
State definitions for Team-H Graph
"""

from typing import Annotated, Literal, Optional, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


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


class AgentRouting(BaseModel):
    """라우터의 라우팅 결정"""
    target_agent: Literal["i", "m", "s", "t"] = Field(
        description="The target agent: 'i' for IoT, 'm' for memory, 's' for search, 't' for time/calendar"
    )
    reason: str = Field(
        description="Brief explanation of why this agent was chosen"
    )
