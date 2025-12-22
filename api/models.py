"""
Pydantic models for FastAPI endpoints (Simplified)

핵심 원칙:
1. Agent는 앱 시작 시 한 번만 생성 (환경 변수 기반)
2. 각 요청은 thread_id만 전달 (상태는 PostgreSQL에서 자동 복원)
3. 불필요한 설정 파라미터 제거
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    채팅 요청 모델 (간소화)

    Agent 설정은 환경 변수에서 로드하므로 요청에 포함 안 함
    """
    message: str = Field(..., description="사용자 메시지")
    thread_id: str = Field(..., description="대화 스레드 ID (상태 식별용)")
    user_id: str = Field(default="default_user", description="사용자 ID")
    session_id: Optional[str] = Field(default=None, description="Langfuse 세션 ID (없으면 thread_id 사용)")


class ResumeRequest(BaseModel):
    """
    HITL 재개 요청 모델

    인터럽트된 대화를 재개할 때 사용
    """
    thread_id: str = Field(..., description="대화 스레드 ID (인터럽트된 대화 식별)")
    decisions: List[Dict[str, Any]] = Field(..., description="승인/거부 결정 리스트")
    user_id: str = Field(default="default_user", description="사용자 ID")
    session_id: Optional[str] = Field(default=None, description="세션 ID")


class InterruptResponse(BaseModel):
    """인터럽트 상태 응답"""
    has_interrupt: bool = Field(..., description="인터럽트 여부")
    interrupt_data: Optional[Dict[str, Any]] = Field(default=None, description="인터럽트 데이터")
    thread_id: str = Field(..., description="스레드 ID")
    next_nodes: List[str] = Field(default_factory=list, description="다음 실행될 노드 목록")


class StateResponse(BaseModel):
    """그래프 상태 응답"""
    status: str = Field(..., description="상태 코드")
    thread_id: str = Field(..., description="스레드 ID")
    state: Dict[str, Any] = Field(..., description="현재 그래프 상태")
    next_nodes: List[str] = Field(default_factory=list, description="다음 실행될 노드")
    has_interrupt: bool = Field(default=False, description="인터럽트 여부")
    interrupts: List[Dict[str, Any]] = Field(default_factory=list, description="인터럽트 리스트")
