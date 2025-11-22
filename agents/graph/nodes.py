"""
Node execution logic for Team-H Graph
"""

from typing import Optional, Dict, Any, List
from langgraph.graph import END
from langgraph.types import Command
from langchain_core.messages import AIMessage, HumanMessage

from .state import TeamHState


class NodesMixin:
    """Mixin class containing all node execution logic for TeamHGraph"""

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
        routing_agent = self.router_llm.with_structured_output(self.AgentRouting)
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
        """Manager I 노드"""
        return self._execute_manager_node(state, config, self.manager_i, "i")

    def _manager_m_node(self, state: TeamHState, config: Optional[Dict[str, Any]] = None) -> Command:
        """Manager M 노드 - user_id 주입 및 재귀 제한 설정"""
        user_id = state.get("user_id", "default_user")

        # 전체 messages를 복사하고, 마지막 user 메시지에 user_id 주입
        messages = list(state["messages"])
        if messages and len(messages) > 0:
            for i in range(len(messages) - 1, -1, -1):
                msg = messages[i]
                if isinstance(msg, HumanMessage) or (hasattr(msg, "type") and msg.type == "human"):
                    messages[i] = HumanMessage(content=f"[User ID: {user_id}]\n{msg.content}")
                    break

        return self._execute_manager_node(
            state, config, self.manager_m, "m",
            messages=messages, recursion_limit=20
        )

    def _manager_s_node(self, state: TeamHState, config: Optional[Dict[str, Any]] = None) -> Command:
        """Manager S 노드"""
        return self._execute_manager_node(state, config, self.manager_s, "s")

    def _manager_t_node(self, state: TeamHState, config: Optional[Dict[str, Any]] = None) -> Command:
        """Manager T 노드"""
        return self._execute_manager_node(state, config, self.manager_t, "t")

    def _execute_manager_node(
        self,
        state: TeamHState,
        config: Optional[Dict[str, Any]],
        manager_instance: Any,
        manager_key: str,
        messages: Optional[List] = None,
        recursion_limit: Optional[int] = None
    ) -> Command:
        """Generic manager node execution logic"""
        icons = {"i": "🏠", "m": "🧠", "s": "🔍", "t": "📅"}
        icon = icons.get(manager_key, "🤖")
        print(f"[{icon}] Manager {manager_key.upper()} executing...")

        # config에서 callbacks 추출
        callbacks = config.get("callbacks", []) if config else []
        manager_config = {"callbacks": callbacks} if callbacks else {}
        if recursion_limit:
            manager_config["recursion_limit"] = recursion_limit

        # Messages setup
        if messages is None:
            messages = state["messages"]

        # 전체 messages를 Manager의 agent에 직접 전달
        result = manager_instance.agent.invoke(
            {"messages": messages},
            config=manager_config
        )

        # Agent 실행 결과에서 새로 생성된 메시지들 추출
        # (기존 state 이후에 생성된 모든 메시지: AIMessage with tool_calls, ToolMessage, 최종 AIMessage)
        original_msg_count = len(state["messages"])
        new_messages = result["messages"][original_msg_count:]

        # Handoff tool 호출 감지 (새로 생성된 메시지만 검사)
        handoff_count = state.get("handoff_count", 0)
        handoff_target = self._detect_handoff(result, original_msg_count)

        # 무한 루프 방지
        if handoff_count >= self.max_handoffs:
            print(f"[⚠️] Max handoffs reached ({self.max_handoffs}), ending conversation")
            next_agent = "end"
        elif handoff_target:
            print(f"[🤝] Handoff tool detected: Manager {manager_key.upper()} → Manager {handoff_target.upper()}")
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
        # Handoff가 발생하면 handoff_target으로, 종료 시에는 현재 Manager 유지
        last_active = next_agent if next_agent != "end" else manager_key

        # Command로 반환 - 새로 생성된 모든 메시지 추가 (ToolMessage 포함)
        return Command(
            goto=goto,
            update={
                "messages": new_messages,  # ✅ AIMessage, ToolMessage 모두 포함
                "handoff_count": handoff_count + (1 if next_agent != "end" else 0),
                "current_agent": manager_key,
                "last_active_manager": last_active,
            }
        )

    def _detect_handoff(self, result: Dict[str, Any], original_msg_count: int) -> Optional[str]:
        """
        결과에서 handoff tool 호출 감지 (새로 생성된 메시지만 검사)

        Args:
            result: Manager agent의 실행 결과
            original_msg_count: 실행 전 메시지 개수

        Returns:
            handoff 대상 agent ID ("i", "m", "s", "t") 또는 None
        """
        messages = result.get("messages", [])

        # 새로 생성된 메시지만 검사 (이전 handoff 재감지 방지)
        new_messages = messages[original_msg_count:]

        # 역순으로 확인 (최근 메시지부터)
        for msg in reversed(new_messages):
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
