# config/homeassistant_api.py
"""
Home Assistant REST API Client

SmartThings OAuth 복잡도를 제거하고 Home Assistant를 통한 장치 제어로 전환.

주요 기능:
- Home Assistant REST API 래퍼
- Long-lived Access Token 사용 (자동 갱신 불필요)
- 동기/비동기 API 지원
- Entity ID 기반 장치 제어

참고:
- Home Assistant API: https://developers.home-assistant.io/docs/api/rest/
"""

import aiohttp
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class HomeAssistantEntity:
    """Home Assistant Entity 정보"""
    entity_id: str
    state: str
    attributes: Dict[str, Any]
    last_changed: str
    last_updated: str


class HomeAssistantAPIClient:
    """
    Home Assistant REST API Client

    Examples:
        >>> client = HomeAssistantAPIClient(
        ...     url="http://localhost:8124",
        ...     token="your_long_lived_token"
        ... )
        >>> await client.turn_on_light("light.living_room")
        >>> await client.turn_off_light("light.bedroom")
        >>> state = await client.get_state("light.bathroom")
    """

    def __init__(
        self,
        url: str = "http://localhost:8124",
        token: Optional[str] = None
    ):
        """
        Home Assistant API Client 초기화

        Args:
            url: Home Assistant URL (포트 포함)
            token: Long-lived Access Token
        """
        self.url = url.rstrip("/")
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    async def call_service(
        self,
        domain: str,
        service: str,
        entity_id: Optional[str] = None,
        service_data: Optional[Dict] = None
    ) -> Dict:
        """
        Home Assistant 서비스 호출

        Args:
            domain: 도메인 (예: light, switch, climate)
            service: 서비스 이름 (예: turn_on, turn_off)
            entity_id: 대상 entity ID
            service_data: 추가 서비스 데이터

        Returns:
            API 응답

        Examples:
            >>> await client.call_service("light", "turn_on", "light.living_room")
            >>> await client.call_service("light", "turn_on", "light.bedroom", {"brightness": 255})
        """
        url = f"{self.url}/api/services/{domain}/{service}"

        data = service_data or {}
        if entity_id:
            data["entity_id"] = entity_id

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self.headers, json=data) as response:
                response.raise_for_status()
                return await response.json()

    async def get_state(self, entity_id: str) -> HomeAssistantEntity:
        """
        Entity 상태 조회

        Args:
            entity_id: Entity ID (예: light.living_room)

        Returns:
            HomeAssistantEntity 객체

        Examples:
            >>> state = await client.get_state("light.living_room")
            >>> print(state.state)  # "on" or "off"
            >>> print(state.attributes["brightness"])  # 밝기 값
        """
        url = f"{self.url}/api/states/{entity_id}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                response.raise_for_status()
                data = await response.json()

                return HomeAssistantEntity(
                    entity_id=data["entity_id"],
                    state=data["state"],
                    attributes=data.get("attributes", {}),
                    last_changed=data["last_changed"],
                    last_updated=data["last_updated"]
                )

    async def get_states(self) -> List[HomeAssistantEntity]:
        """
        모든 Entity 상태 조회

        Returns:
            HomeAssistantEntity 리스트
        """
        url = f"{self.url}/api/states"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                response.raise_for_status()
                data = await response.json()

                return [
                    HomeAssistantEntity(
                        entity_id=item["entity_id"],
                        state=item["state"],
                        attributes=item.get("attributes", {}),
                        last_changed=item["last_changed"],
                        last_updated=item["last_updated"]
                    )
                    for item in data
                ]

    # ========================================
    # 편의 메서드 (자주 사용하는 서비스)
    # ========================================

    async def turn_on_light(self, entity_id: str, **kwargs) -> Dict:
        """
        조명 켜기

        Args:
            entity_id: Light entity ID (예: light.living_room)
            **kwargs: 추가 옵션 (brightness, color_temp, rgb_color 등)

        Returns:
            API 응답
        """
        return await self.call_service("light", "turn_on", entity_id, kwargs or None)

    async def turn_off_light(self, entity_id: str) -> Dict:
        """
        조명 끄기

        Args:
            entity_id: Light entity ID

        Returns:
            API 응답
        """
        return await self.call_service("light", "turn_off", entity_id)

    async def toggle_light(self, entity_id: str) -> Dict:
        """
        조명 토글 (켜짐 ↔ 꺼짐)

        Args:
            entity_id: Light entity ID

        Returns:
            API 응답
        """
        return await self.call_service("light", "toggle", entity_id)

    async def turn_on_switch(self, entity_id: str) -> Dict:
        """
        스위치 켜기

        Args:
            entity_id: Switch entity ID (예: switch.speaker_outlet)

        Returns:
            API 응답
        """
        return await self.call_service("switch", "turn_on", entity_id)

    async def turn_off_switch(self, entity_id: str) -> Dict:
        """
        스위치 끄기

        Args:
            entity_id: Switch entity ID

        Returns:
            API 응답
        """
        return await self.call_service("switch", "turn_off", entity_id)

    async def is_on(self, entity_id: str) -> bool:
        """
        Entity가 켜져 있는지 확인

        Args:
            entity_id: Entity ID

        Returns:
            True if on, False if off
        """
        state = await self.get_state(entity_id)
        return state.state == "on"

    async def health_check(self) -> bool:
        """
        Home Assistant API 연결 확인

        Returns:
            True if healthy, False otherwise
        """
        try:
            url = f"{self.url}/api/"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    data = await response.json()
                    return data.get("message") == "API running."
        except Exception:
            return False


# ============================================================
# 테스트 및 디버깅용 헬퍼 함수
# ============================================================

async def test_api_client():
    """
    API 클라이언트 테스트 함수
    """
    import os
    from dotenv import load_dotenv

    load_dotenv()

    client = HomeAssistantAPIClient(
        url=os.getenv("HOMEASSISTANT_URL", "http://localhost:8124"),
        token=os.getenv("HOMEASSISTANT_TOKEN")
    )

    # Health check
    if await client.health_check():
        print("✅ Home Assistant API 연결 성공")
    else:
        print("❌ Home Assistant API 연결 실패")
        return

    # 모든 entity 조회
    states = await client.get_states()
    print(f"\n📊 총 {len(states)}개의 entity 발견")

    # Light entity만 필터링
    lights = [s for s in states if s.entity_id.startswith("light.")]
    print(f"💡 조명: {len(lights)}개")
    for light in lights[:5]:  # 처음 5개만 출력
        print(f"  - {light.entity_id}: {light.state}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_api_client())
