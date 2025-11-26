# manager_i.py
"""
Manager I Agent - IoT 제어 에이전트 (Home Assistant 버전)

Manager I는 집안의 IoT 장치를 제어하는 에이전트입니다:
- 미니PC 종료
- 거실/안방/화장실 불 제어
- 거실 스피커 제어 (IoT 콘센트)

변경사항 (2025-11-26):
- SmartThings OAuth → Home Assistant API로 전환
- 토큰 갱신 복잡도 제거
- SmartThings 허브는 Home Assistant Integration으로 연결

ManagerBase를 상속받아 공통 로직을 재사용합니다.
HumanInTheLoopMiddleware를 통해 위험한 작업에 대한 승인을 요구합니다.
"""

import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Literal
import asyncio

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Agents import (__init__.py 활용)
from agents import ManagerBase
from agents.context import TeamHContext
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool, ToolRuntime

# Home Assistant API Client
from config.homeassistant_api import HomeAssistantAPIClient


class ManagerI(ManagerBase):
    """Manager I 에이전트 클래스 - IoT 제어 전문 (Home Assistant)"""

    # 클래스 레벨 상수: Entity ID 매핑
    # SmartThings Integration 후 Home Assistant에서 확인한 실제 entity_id 사용
    ENTITY_MAP = {
        # 조명 (실제로는 모두 switch로 등록됨)
        "living_room_light": "switch.geosil",  # 거실
        "bedroom_light": "switch.naebang",  # 내방 (안방)
        "bathroom_light": "switch.kyubeu",  # 큐브 (화장실 공기청정기)
        # 스위치 (스피커 콘센트)
        "living_room_speaker_outlet": "switch.seupikeo",  # 스피커
    }

    # 방 이름 별칭 매핑
    ROOM_ALIASES = {
        # Living room
        "거실": "living_room",
        "프로젝터": "living_room",
        "living_room": "living_room",
        # Bedroom
        "안방": "bedroom",
        "세로모니터": "bedroom",
        "서브모니터": "bedroom",
        "bedroom": "bedroom",
        # Bathroom
        "화장실": "bathroom",
        "공기청정기": "bathroom",
        "큐브": "bathroom",
        "bathroom": "bathroom",
    }

    # 방 이름 한글 변환
    ROOM_NAME_KR = {
        "living_room": "거실",
        "bedroom": "안방",
        "bathroom": "화장실",
    }

    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        temperature: float = 0.7,
        homeassistant_url: str = "http://localhost:8124",
        homeassistant_token: Optional[str] = None,
        entity_map: Optional[Dict[str, str]] = None,
        additional_tools: Optional[List] = None,
        middleware: Optional[List] = None,
    ):
        """
        Manager I 에이전트 초기화

        Args:
            model_name: 사용할 LLM 모델 이름 (기본값: gpt-4.1-mini)
            temperature: 모델 temperature 설정
            homeassistant_url: Home Assistant URL
            homeassistant_token: Home Assistant Long-Lived Access Token
            entity_map: Entity ID 매핑 (기본값: ENTITY_MAP)
            additional_tools: 핸드오프 등 추가 툴 리스트
            middleware: 외부에서 전달받은 미들웨어 리스트 (Langfuse 로깅 등)
        """
        # Home Assistant API Client 초기화
        if not homeassistant_token:
            raise ValueError(
                "Home Assistant Long-Lived Access Token is required.\n"
                "Generate token in Home Assistant:\n"
                "  Profile → Security → Long-Lived Access Tokens → Create Token"
            )

        self.ha_client = HomeAssistantAPIClient(
            url=homeassistant_url,
            token=homeassistant_token
        )

        # Entity ID 매핑 설정
        self.entity_map = entity_map or self.ENTITY_MAP.copy()

        # Entity 설정 검증 (비동기로 수행할 수 없으므로 경고만 출력)
        self._validate_entity_config()

        # HITL 미들웨어 생성
        hitl_middleware = HumanInTheLoopMiddleware(
            interrupt_on={
                # 위험한 작업 - 승인 필요
                "shutdown_mini_pc": True,
                "turn_on_light": False,
                "turn_off_light": False,
                "turn_off_speaker": False,
                "get_device_status": False,
            },
            description_prefix="🏠 IoT operation pending approval",
        )

        # middleware 리스트 합치기 (외부 middleware + HITL)
        combined_middleware = []
        if middleware:
            combined_middleware.extend(middleware)
        combined_middleware.append(hitl_middleware)

        # 베이스 클래스 초기화 (공통 로직)
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            additional_tools=additional_tools,
            middleware=combined_middleware,
        )

        # 추가 초기화 메시지
        print(f"    - Home Assistant: {homeassistant_url}")
        print(f"    - HITL: Enabled for dangerous operations")
        print(f"    - Entities configured: {len(self.entity_map)}")

    def _validate_entity_config(self):
        """초기화 시 Entity 설정 검증"""
        required_entities = [
            "living_room_light",
            "bedroom_light",
            "bathroom_light",
            "living_room_speaker_outlet"
        ]

        missing_entities = [e for e in required_entities if e not in self.entity_map]
        if missing_entities:
            print(f"[⚠️] 경고: 일부 Entity가 설정되지 않았습니다: {missing_entities}")
            print(f"[⚠️] 이 장치들에 대한 제어 명령은 실패할 수 있습니다.")
            print(f"[⚠️] Home Assistant에서 SmartThings Integration 설정 후 entity_id를 확인하세요.")

    def _control_light(self, room: str, action: Literal["on", "off"]) -> str:
        """
        통합된 조명 제어 로직 (turn_on/turn_off 중복 제거)

        Args:
            room: 방 이름 (한글/영어 모두 지원)
            action: "on" 또는 "off"

        Returns:
            작업 결과 메시지
        """
        try:
            # 방 이름 정규화
            room_normalized = self.ROOM_ALIASES.get(room.lower(), room.lower())

            # Entity 키 확인
            entity_key = f"{room_normalized}_light"
            if entity_key not in self.entity_map:
                return f"❌ Unknown room: '{room}'. 사용 가능: 거실/안방/화장실 또는 living_room/bedroom/bathroom"

            # Entity ID 확인
            entity_id = self.entity_map[entity_key]

            # Home Assistant API로 장치 제어
            # 모든 장치가 switch로 등록되어 있으므로 switch API 사용
            if action == "on":
                asyncio.run(self.ha_client.turn_on_switch(entity_id))
                action_kr = "켰습니다"
            else:
                asyncio.run(self.ha_client.turn_off_switch(entity_id))
                action_kr = "껐습니다"

            room_kr = self.ROOM_NAME_KR.get(room_normalized, room)
            return f"✅ {room_kr} 불을 {action_kr}."

        except Exception as e:
            return f"❌ Error controlling light in {room}: {str(e)}"

    def _create_tools(self) -> List:
        """IoT 제어 관련 툴 생성"""

        @tool
        def shutdown_mini_pc(runtime: ToolRuntime[TeamHContext] = None) -> str:
            """
            Shutdown the mini PC (Linux system).

            This is a DANGEROUS operation that will turn off the mini PC.
            Use this only when explicitly requested by the user.

            Args:
                runtime: Automatically injected runtime context

            Returns:
                Confirmation message about shutdown
            """
            try:
                # Linux shutdown 명령어 실행
                result = subprocess.run(
                    ["sudo", "shutdown", "-h", "now"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                if result.returncode == 0:
                    return "✅ Mini PC shutdown initiated. The system will shut down now."
                else:
                    return f"⚠️ Shutdown command executed but returned code {result.returncode}: {result.stderr}"

            except subprocess.TimeoutExpired:
                return "✅ Shutdown command sent (timed out as expected - system is shutting down)"
            except Exception as e:
                return f"❌ Error shutting down mini PC: {str(e)}"

        @tool
        def turn_on_light(room: str, runtime: ToolRuntime[TeamHContext] = None) -> str:
            """
            Turn on the light in a specified room.

            Args:
                room: Room name. Supports both English and Korean:
                    - living_room, 거실, 프로젝터 → living room light
                    - bedroom, 안방, 세로모니터, 서브모니터 → bedroom light
                    - bathroom, 화장실, 공기청정기, 큐브 → bathroom light
                runtime: Automatically injected runtime context

            Returns:
                Status message about the light operation
            """
            return self._control_light(room, "on")

        @tool
        def turn_off_light(room: str, runtime: ToolRuntime[TeamHContext] = None) -> str:
            """
            Turn off the light in a specified room.

            Args:
                room: Room name. Supports both English and Korean:
                    - living_room, 거실, 프로젝터 → living room light
                    - bedroom, 안방, 세로모니터, 서브모니터 → bedroom light
                    - bathroom, 화장실, 공기청정기, 큐브 → bathroom light
                runtime: Automatically injected runtime context

            Returns:
                Status message about the light operation
            """
            return self._control_light(room, "off")

        @tool
        def turn_on_speaker(runtime: ToolRuntime[TeamHContext] = None) -> str:
            """
            Turn on the living room speaker via smart outlet.

            The speaker is connected to a smart plug that can be controlled remotely.

            Args:
                runtime: Automatically injected runtime context

            Returns:
                Status message about the speaker operation
            """
            try:
                entity_id = self.entity_map.get("living_room_speaker_outlet")
                if not entity_id:
                    return "❌ Speaker outlet entity not configured"

                # Home Assistant API로 스피커 콘센트 켜기
                asyncio.run(self.ha_client.turn_on_switch(entity_id))
                return "✅ 거실 스피커를 켰습니다."

            except Exception as e:
                return f"❌ Error turning on speaker: {str(e)}"

        @tool
        def turn_off_speaker(runtime: ToolRuntime[TeamHContext] = None) -> str:
            """
            Turn off the living room speaker via smart outlet.

            The speaker is connected to a smart plug that can be controlled remotely.

            Args:
                runtime: Automatically injected runtime context

            Returns:
                Status message about the speaker operation
            """
            try:
                entity_id = self.entity_map.get("living_room_speaker_outlet")
                if not entity_id:
                    return "❌ Speaker outlet entity not configured"

                # Home Assistant API로 스피커 콘센트 끄기
                asyncio.run(self.ha_client.turn_off_switch(entity_id))
                return "✅ 거실 스피커를 껐습니다."

            except Exception as e:
                return f"❌ Error turning off speaker: {str(e)}"

        @tool
        def get_device_status(room: str, device_type: str = "light", runtime: ToolRuntime[TeamHContext] = None) -> str:
            """
            Get the current status of a device in a specified room.

            Args:
                room: Room name. Supports both English and Korean:
                    - living_room, 거실, 프로젝터 → living room light
                    - bedroom, 안방, 세로모니터, 서브모니터 → bedroom light
                    - bathroom, 화장실, 공기청정기, 큐브 → bathroom light
                device_type: Type of device (light or speaker)
                runtime: Automatically injected runtime context

            Returns:
                Current status of the device
            """
            try:
                if device_type == "speaker":
                    entity_key = "living_room_speaker_outlet"
                    room_normalized = "living_room"
                else:
                    # 방 이름 정규화 (클래스 상수 사용)
                    room_normalized = self.ROOM_ALIASES.get(room.lower(), room.lower())
                    entity_key = f"{room_normalized}_light"

                if entity_key not in self.entity_map:
                    return f"❌ Unknown room or device type"

                entity_id = self.entity_map[entity_key]

                # Home Assistant API로 상태 확인
                is_on = asyncio.run(self.ha_client.is_on(entity_id))

                room_kr = self.ROOM_NAME_KR.get(room_normalized, room)
                device_kr = "스피커" if device_type == "speaker" else "불"
                state_kr = "켜져 있습니다" if is_on else "꺼져 있습니다"

                return f"📊 {room_kr} {device_kr}은(는) 현재 {state_kr}."

            except Exception as e:
                return f"❌ Error getting device status: {str(e)}"

        return [
            shutdown_mini_pc,
            turn_on_light,
            turn_off_light,
            turn_on_speaker,
            turn_off_speaker,
            get_device_status,
        ]


def create_manager_i_agent(**kwargs) -> ManagerI:
    """
    Manager I 에이전트 생성 헬퍼 함수

    Args:
        **kwargs: ManagerI 초기화 파라미터

    Returns:
        ManagerI 인스턴스
    """
    return ManagerI(**kwargs)


# 싱글톤 인스턴스 (선택적 사용)
_manager_i_agent_instance = None


def get_manager_i_agent(**kwargs) -> ManagerI:
    """
    Manager I 에이전트 싱글톤 인스턴스 반환

    Args:
        **kwargs: ManagerI 초기화 파라미터 (처음 생성 시에만 사용됨)

    Returns:
        ManagerI 인스턴스
    """
    global _manager_i_agent_instance
    if _manager_i_agent_instance is None:
        _manager_i_agent_instance = ManagerI(**kwargs)
    return _manager_i_agent_instance
