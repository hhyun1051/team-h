# base_manager.py
"""
Base Manager Agent - 공통 기능을 제공하는 추상 베이스 클래스

모든 매니저 에이전트(ManagerS, ManagerM, ManagerI)가 상속하는 베이스 클래스입니다:
- 공통 초기화 로직
- 에이전트 생성 패턴
- invoke/stream/get_state 메서드
- 프롬프트 파일 관리
- hook 메서드를 통한 확장 지원

"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from pathlib import Path
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model


class ManagerBase(ABC):
    """모든 매니저 에이전트의 베이스 클래스"""

    # 프롬프트 디렉토리 경로
    PROMPTS_DIR = Path(__file__).parent / "prompts"

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        additional_tools: Optional[List] = None,
        middleware: Optional[List] = None,
        **kwargs,
    ):
        """
        베이스 매니저 초기화

        Args:
            model_name: 사용할 LLM 모델 이름 (기본값: gpt-4o-mini)
            temperature: 모델 temperature 설정
            additional_tools: 핸드오프 등 추가 툴 리스트
            middleware: 에이전트 미들웨어 리스트 (예: HumanInTheLoopMiddleware)
            **kwargs: 자식 클래스의 특수 파라미터
        """
        manager_type = self.__class__.__name__
        print(f"[🤖] Initializing {manager_type} Agent...")

        self.model_name = model_name
        self.temperature = temperature

        # 자식 클래스의 특수 초기화를 위한 hook
        # 이 메서드는 _create_tools() 호출 전에 실행되어야 함
        # (툴 생성 시 자식 클래스의 특수 속성이 필요할 수 있음)
        self._pre_init_hook(**kwargs)

        # 각 매니저가 자신의 툴을 생성
        self.tools = self._create_tools()
        if additional_tools:
            self.tools.extend(additional_tools)
            print(f"[➕] Added {len(additional_tools)} additional tools (handoff tools)")

        # LLM 모델 생성
        model = init_chat_model(
            model=self.model_name,
            model_provider="openai",
            temperature=self.temperature,
        )

        # 시스템 프롬프트 생성
        base_prompt = self._get_base_prompt()

        # 핸드오프 툴이 있으면 협업 프롬프트 추가
        if additional_tools:
            handoff_prompt = self._get_handoff_prompt()
            system_prompt = base_prompt + handoff_prompt
        else:
            system_prompt = base_prompt

        # 공통 마지막 메시지 추가
        system_prompt += self._get_closing_prompt()

        # 에이전트 생성 (LangChain v1)
        # Note: checkpointer는 사용하지 않음
        # TeamHGraph가 상위 레벨에서 상태를 관리하고,
        # 각 Manager는 state["messages"]를 통해 대화 맥락을 받음
        agent_kwargs = {
            "model": model,
            "tools": self.tools,
            "system_prompt": system_prompt,
        }

        # 미들웨어가 있으면 추가
        if middleware:
            agent_kwargs["middleware"] = middleware

        self.agent = create_agent(**agent_kwargs)

        # 초기화 완료 메시지
        self._print_initialization_summary()

    def _pre_init_hook(self, **kwargs):
        """
        자식 클래스의 특수 초기화를 위한 hook

        _create_tools() 호출 전에 실행됩니다.
        자식 클래스에서 오버라이드하여 특수 속성을 초기화하세요.

        Args:
            **kwargs: 자식 클래스의 특수 파라미터

        Example:
            class ManagerM(ManagerBase):
                def _pre_init_hook(self, **kwargs):
                    self.memory = ManagerMMemory(
                        embedder_url=kwargs.get("embedder_url"),
                        ...
                    )
        """
        pass

    def _prepare_message(self, message: str, **kwargs) -> str:
        """
        메시지 전처리를 위한 hook

        invoke/stream 호출 시 메시지를 전처리합니다.
        자식 클래스에서 오버라이드하여 메시지를 수정하세요.

        Args:
            message: 원본 메시지
            **kwargs: invoke/stream에서 전달된 추가 파라미터

        Returns:
            전처리된 메시지

        Example:
            class ManagerM(ManagerBase):
                def _prepare_message(self, message, **kwargs):
                    user_id = kwargs.get("user_id", "default_user")
                    return f"[User ID: {user_id}]\\n{message}"
        """
        return message

    @abstractmethod
    def _create_tools(self) -> List:
        """
        각 매니저가 자신의 툴을 정의

        Returns:
            툴 함수 리스트
        """
        pass

    def _get_prompt_filename(self) -> str:
        """
        프롬프트 파일명 반환 (기본값: 클래스 이름을 snake_case로 변환)

        Returns:
            프롬프트 파일명 (예: "manager_s.yaml")

        Note:
            자식 클래스에서 오버라이드하여 다른 파일명 사용 가능
        """
        # ManagerS -> manager_s
        class_name = self.__class__.__name__
        # CamelCase를 snake_case로 변환
        import re
        snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()
        return f"{snake_case}.yaml"

    def _load_prompt_from_file(self, filename: str) -> str:
        """
        프롬프트 파일 로드 (YAML 형식)

        Args:
            filename: 프롬프트 파일명 (prompts/ 디렉토리 내)

        Returns:
            프롬프트 문자열

        Raises:
            FileNotFoundError: 프롬프트 파일이 없을 경우
        """
        prompt_path = self.PROMPTS_DIR / filename

        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {prompt_path}\n"
                f"Please create the prompt file at: {prompt_path}"
            )

        # YAML 파일 읽기
        try:
            import yaml
            with open(prompt_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            # 'content' 키에서 프롬프트 추출
            if isinstance(data, dict) and 'content' in data:
                return data['content'].strip()
            else:
                raise ValueError(f"YAML file must contain 'content' key: {prompt_path}")
        except Exception as e:
            raise ValueError(f"Failed to load YAML prompt from {prompt_path}: {e}")

    def _get_base_prompt(self) -> str:
        """
        베이스 시스템 프롬프트 반환

        기본적으로 prompts/ 디렉토리에서 파일을 로드합니다.
        자식 클래스에서 오버라이드하여 직접 문자열 반환도 가능합니다.

        Returns:
            시스템 프롬프트 문자열
        """
        filename = self._get_prompt_filename()
        return self._load_prompt_from_file(filename)

    def _get_handoff_prompt(self) -> str:
        """
        핸드오프 툴이 있을 때 추가되는 협업 프롬프트

        기본적으로 prompts/handoff_common.yaml에서 로드합니다.

        Returns:
            핸드오프 프롬프트 문자열
        """
        return self._load_prompt_from_file("handoff_common.yaml")

    def _get_closing_prompt(self) -> str:
        """
        모든 프롬프트 마지막에 추가되는 공통 메시지

        Returns:
            마지막 프롬프트 문자열
        """
        return "\n\nBe helpful, accurate, and respond in Korean when appropriate."

    def _print_initialization_summary(self):
        """초기화 완료 메시지 출력"""
        manager_type = self.__class__.__name__
        print(f"[✅] {manager_type} Agent initialized successfully")
        print(f"    - Model: {self.model_name}")
        print(f"    - Temperature: {self.temperature}")
        print(f"    - Tools: {len(self.tools)} tools")

    def invoke(self, message: str, thread_id: str = "default_thread", **kwargs) -> Dict[str, Any]:
        """
        에이전트 실행

        Args:
            message: 사용자 메시지
            thread_id: 대화 스레드 ID
            **kwargs: 추가 파라미터 (각 매니저별로 다를 수 있음)

        Returns:
            에이전트 응답
        """
        config = {"configurable": {"thread_id": thread_id}}

        # 메시지 전처리 hook 호출 (자식 클래스에서 오버라이드 가능)
        prepared_message = self._prepare_message(message, **kwargs)

        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": prepared_message}]},
            config=config,
        )

        return result

    def stream(self, message: str, thread_id: str = "default_thread", **kwargs):
        """
        에이전트 스트리밍 실행

        Args:
            message: 사용자 메시지
            thread_id: 대화 스레드 ID
            **kwargs: 추가 파라미터

        Yields:
            에이전트 응답 청크
        """
        config = {"configurable": {"thread_id": thread_id}}

        # 메시지 전처리 hook 호출 (자식 클래스에서 오버라이드 가능)
        prepared_message = self._prepare_message(message, **kwargs)

        for chunk in self.agent.stream(
            {"messages": [{"role": "user", "content": prepared_message}]},
            config=config,
            stream_mode="values",
        ):
            yield chunk

    def get_state(self, config: dict):
        """
        에이전트의 현재 상태 반환

        Args:
            config: 설정 딕셔너리 (thread_id 포함)

        Returns:
            에이전트 상태
        """
        return self.agent.get_state(config)

    def invoke_command(self, command, config: dict):
        """
        Command 객체를 사용한 에이전트 실행 (HITL 승인 처리용)

        Args:
            command: langgraph Command 객체
            config: 설정 딕셔너리 (thread_id 포함)

        Returns:
            에이전트 응답
        """
        return self.agent.invoke(command, config)