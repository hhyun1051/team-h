# manager_i.py
"""
Manager I Agent - IoT 제어 에이전트

Manager I는 집안의 IoT 장치를 제어하는 에이전트입니다:
- 미니PC 종료
- 거실/안방/화장실 불 제어
- 거실 스피커 제어 (IoT 콘센트)

ManagerBase를 상속받아 공통 로직을 재사용합니다.
HumanInTheLoopMiddleware를 통해 위험한 작업에 대한 승인을 요구합니다.
"""

import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Literal
import asyncio
import aiohttp
import time

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from agents.base_manager import ManagerBase
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool

# SmartThings API (pysmartthings 사용)
try:
    import pysmartthings
except ImportError:
    print("⚠️  pysmartthings가 설치되지 않았습니다. pip install pysmartthings를 실행하세요.")
    pysmartthings = None


class ManagerI(ManagerBase):
    """Manager I 에이전트 클래스 - IoT 제어 전문"""

    # 클래스 레벨 상수: 방 이름 별칭 매핑
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

    # 방 이름 -> 장치 키 매핑
    ROOM_DEVICE_MAP = {
        "living_room": "living_room_light",
        "bedroom": "bedroom_light",
        "bathroom": "bathroom_light",
    }

    # 방 이름 한글 변환
    ROOM_NAME_KR = {
        "living_room": "거실",
        "bedroom": "안방",
        "bathroom": "화장실",
    }

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        smartthings_token: Optional[str] = None,
        device_config: Optional[Dict[str, str]] = None,
        additional_tools: Optional[List] = None,
    ):
        """
        Manager I 에이전트 초기화

        Args:
            model_name: 사용할 LLM 모델 이름 (기본값: gpt-4o-mini)
            temperature: 모델 temperature 설정
            smartthings_token: SmartThings API 토큰
            device_config: 장치 설정 (room_name -> device_id 매핑)
            additional_tools: 핸드오프 등 추가 툴 리스트
        """
        # 특수 파라미터 검증 및 저장
        if not smartthings_token:
            raise ValueError("SmartThings API token is required")

        self.smartthings_token = smartthings_token
        self.device_config = device_config or {}

        # 장치 설정 검증
        self._validate_device_config()

        # HITL 미들웨어 생성
        hitl_middleware = HumanInTheLoopMiddleware(
            interrupt_on={
                # 위험한 작업 - 승인 필요
                "shutdown_mini_pc": True,
                # 일반 제어 작업 - 승인 불필요 (빠른 실행)
                "turn_on_light": False,
                "turn_off_light": False,
                "turn_off_speaker": False,
                "get_device_status": False,
            },
            description_prefix="🏠 IoT operation pending approval",
        )

        # 베이스 클래스 초기화 (공통 로직)
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            additional_tools=additional_tools,
            middleware=[hitl_middleware],
        )

        # 추가 초기화 메시지
        print(f"    - HITL: Enabled for dangerous operations")
        print(f"    - Devices configured: {len(self.device_config)}")

    def _validate_device_config(self):
        """초기화 시 장치 설정 검증"""
        required_devices = [
            "living_room_light",
            "bedroom_light",
            "bathroom_light",
            "living_room_speaker_outlet"
        ]

        missing_devices = [d for d in required_devices if d not in self.device_config]
        if missing_devices:
            print(f"[⚠️] 경고: 일부 장치가 설정되지 않았습니다: {missing_devices}")
            print(f"[⚠️] 이 장치들에 대한 제어 명령은 실패할 수 있습니다.")

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

            # 장치 키 확인
            device_key = self.ROOM_DEVICE_MAP.get(room_normalized)
            if not device_key:
                return f"❌ Unknown room: '{room}'. 사용 가능: 거실/안방/화장실 또는 living_room/bedroom/bathroom"

            # 장치 ID 확인
            device_id = self.device_config.get(device_key)
            if not device_id:
                return f"❌ Device not configured for room: {room}"

            # SmartThings API로 장치 제어
            if action == "on":
                asyncio.run(self._turn_on_device(device_id))
                action_kr = "켰습니다"
            else:
                asyncio.run(self._turn_off_device(device_id))
                action_kr = "껐습니다"

            room_kr = self.ROOM_NAME_KR.get(room_normalized, room)
            return f"✅ {room_kr} 불을 {action_kr}."

        except Exception as e:
            return f"❌ Error controlling light in {room}: {str(e)}"

    def _create_tools(self) -> List:
        """IoT 제어 관련 툴 생성"""

        @tool
        def shutdown_mini_pc() -> str:
            """
            Shutdown the mini PC (Linux system).

            This is a DANGEROUS operation that will turn off the mini PC.
            Use this only when explicitly requested by the user.

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
        def turn_on_light(room: str) -> str:
            """
            Turn on the light in a specified room.

            Args:
                room: Room name. Supports both English and Korean:
                    - living_room, 거실, 프로젝터 → living room light
                    - bedroom, 안방, 세로모니터, 서브모니터 → bedroom light
                    - bathroom, 화장실, 공기청정기, 큐브 → bathroom light

            Returns:
                Status message about the light operation
            """
            return self._control_light(room, "on")

        @tool
        def turn_off_light(room: str) -> str:
            """
            Turn off the light in a specified room.

            Args:
                room: Room name. Supports both English and Korean:
                    - living_room, 거실, 프로젝터 → living room light
                    - bedroom, 안방, 세로모니터, 서브모니터 → bedroom light
                    - bathroom, 화장실, 공기청정기, 큐브 → bathroom light

            Returns:
                Status message about the light operation
            """
            return self._control_light(room, "off")

        @tool
        def turn_on_speaker() -> str:
            """
            Turn on the living room speaker via smart outlet.

            The speaker is connected to a smart plug that can be controlled remotely.

            Returns:
                Status message about the speaker operation
            """
            try:
                device_id = self.device_config.get("living_room_speaker_outlet")
                if not device_id:
                    return "❌ Speaker outlet device not configured"

                # SmartThings API로 스피커 콘센트 켜기
                asyncio.run(self._turn_on_device(device_id))
                time.sleep(0.1)
                return "✅ 거실 스피커를 켰습니다."

            except Exception as e:
                return f"❌ Error turning on speaker: {str(e)}"

        @tool
        def turn_off_speaker() -> str:
            """
            Turn off the living room speaker via smart outlet.

            The speaker is connected to a smart plug that can be controlled remotely.

            Returns:
                Status message about the speaker operation
            """
            try:
                device_id = self.device_config.get("living_room_speaker_outlet")
                if not device_id:
                    return "❌ Speaker outlet device not configured"

                # SmartThings API로 스피커 콘센트 끄기
                asyncio.run(self._turn_off_device(device_id))
                time.sleep(0.1)
                return "✅ 거실 스피커를 껐습니다."

            except Exception as e:
                return f"❌ Error turning off speaker: {str(e)}"

        @tool
        def get_device_status(room: str, device_type: str = "light") -> str:
            """
            Get the current status of a device in a specified room.

            Args:
                room: Room name. Supports both English and Korean:
                    - living_room, 거실, 프로젝터 → living room light
                    - bedroom, 안방, 세로모니터, 서브모니터 → bedroom light
                    - bathroom, 화장실, 공기청정기, 큐브 → bathroom light
                device_type: Type of device (light or speaker)

            Returns:
                Current status of the device
            """
            try:
                if device_type == "speaker":
                    device_key = "living_room_speaker_outlet"
                    room_normalized = "living_room"
                else:
                    # 방 이름 정규화 (클래스 상수 사용)
                    room_normalized = self.ROOM_ALIASES.get(room.lower(), room.lower())
                    device_key = self.ROOM_DEVICE_MAP.get(room_normalized)

                if not device_key:
                    return f"❌ Unknown room or device type"

                device_id = self.device_config.get(device_key)
                if not device_id:
                    return f"❌ Device not configured"

                # SmartThings API로 상태 확인
                status = asyncio.run(self._get_device_status(device_id))

                room_kr = self.ROOM_NAME_KR.get(room_normalized, room)
                device_kr = "스피커" if device_type == "speaker" else "불"
                state_kr = "켜져 있습니다" if status == "on" else "꺼져 있습니다"

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

    async def _turn_on_device(self, device_id: str) -> bool:
        """SmartThings API를 사용하여 장치 켜기"""
        async with aiohttp.ClientSession() as session:
            api = pysmartthings.SmartThings(_token=self.smartthings_token, session=session)
            await api.execute_device_command(
                device_id=device_id,
                capability=pysmartthings.Capability.SWITCH,
                command=pysmartthings.Command.ON,
                component="main"
            )
            return True

    async def _turn_off_device(self, device_id: str) -> bool:
        """SmartThings API를 사용하여 장치 끄기"""
        async with aiohttp.ClientSession() as session:
            api = pysmartthings.SmartThings(_token=self.smartthings_token, session=session)
            await api.execute_device_command(
                device_id=device_id,
                capability=pysmartthings.Capability.SWITCH,
                command=pysmartthings.Command.OFF,
                component="main"
            )
            return True

    async def _get_device_status(self, device_id: str) -> str:
        """SmartThings API를 사용하여 장치 상태 확인"""
        async with aiohttp.ClientSession() as session:
            api = pysmartthings.SmartThings(_token=self.smartthings_token, session=session)
            status = await api.get_device_status(device_id)

            # switch capability의 상태 확인
            switch_status = status.switch
            return switch_status  # "on" or "off"


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
