"""
SmartThings API 기본 기능 테스트

실제 SmartThings API를 사용하여 장치 연결 및 제어를 테스트합니다.
"""

import pytest
import asyncio
import aiohttp
import pysmartthings


@pytest.mark.integration
@pytest.mark.asyncio
async def test_smartthings_connection(smartthings_token):
    """SmartThings API 연결 테스트"""
    async with aiohttp.ClientSession() as session:
        api = pysmartthings.SmartThings(_token=smartthings_token, session=session)

        # API 객체가 생성되었는지 확인
        assert api is not None
        assert api._token == smartthings_token


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_all_devices(smartthings_token):
    """모든 장치 목록 가져오기 테스트"""
    async with aiohttp.ClientSession() as session:
        api = pysmartthings.SmartThings(_token=smartthings_token, session=session)

        # 장치 목록 가져오기
        devices = await api.get_devices()

        # 장치가 존재하는지 확인
        assert len(devices) > 0
        print(f"\n총 {len(devices)}개의 장치 발견:")

        for device in devices:
            print(f"  - {device.label} ({device.device_id})")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_specific_device(smartthings_token, real_device_ids):
    """특정 장치 정보 가져오기 테스트"""
    device_id = real_device_ids["speaker"]

    async with aiohttp.ClientSession() as session:
        api = pysmartthings.SmartThings(_token=smartthings_token, session=session)

        # 특정 장치 가져오기
        device = await api.get_device(device_id)

        # 장치 정보 확인
        assert device is not None
        assert device.device_id == device_id
        assert device.label is not None

        print(f"\n장치 정보:")
        print(f"  - 이름: {device.label}")
        print(f"  - ID: {device.device_id}")
        print(f"  - 타입: {device.type}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_device_status(smartthings_token, real_device_ids):
    """장치 상태 조회 테스트"""
    device_id = real_device_ids["speaker"]

    async with aiohttp.ClientSession() as session:
        api = pysmartthings.SmartThings(_token=smartthings_token, session=session)

        # 장치 가져오기
        device = await api.get_device(device_id)

        # 상태 새로고침
        await device.status.refresh()

        # 상태 정보 확인
        print(f"\n{device.label} 상태:")

        # Switch 상태
        if hasattr(device.status, "switch"):
            print(f"  - Switch: {device.status.switch}")
            assert device.status.switch in ["on", "off"]

        # Power 상태
        if hasattr(device.status, "power"):
            print(f"  - Power: {device.status.power}W")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_all_device_details(smartthings_token, real_device_ids):
    """모든 테스트 장치의 상세 정보 확인"""
    async with aiohttp.ClientSession() as session:
        api = pysmartthings.SmartThings(_token=smartthings_token, session=session)

        print("\n=== 모든 장치 상세 정보 ===")

        for name, device_id in real_device_ids.items():
            try:
                device = await api.get_device(device_id)
                await device.status.refresh()

                print(f"\n[{name}] {device.label}")
                print(f"  Device ID: {device.device_id}")
                print(f"  Type: {device.type}")

                # Components 정보
                if hasattr(device, "components"):
                    print(f"  Components: {list(device.components.keys())}")

                    # Main component의 capabilities
                    if "main" in device.components:
                        main_comp = device.components["main"]
                        if hasattr(main_comp, "capabilities"):
                            caps = [str(cap) for cap in main_comp.capabilities[:5]]
                            print(f"  Capabilities: {', '.join(caps)}...")

                # 상태 정보
                print(f"  Status:")
                if hasattr(device.status, "switch"):
                    print(f"    - Switch: {device.status.switch}")
                if hasattr(device.status, "power"):
                    print(f"    - Power: {device.status.power}W")
                if hasattr(device.status, "energy"):
                    print(f"    - Energy: {device.status.energy}kWh")

            except Exception as e:
                print(f"\n[{name}] 오류: {str(e)}")


@pytest.mark.integration
@pytest.mark.safe
@pytest.mark.asyncio
async def test_device_control_switch(smartthings_token, real_device_ids):
    """
    장치 제어 테스트 (Switch on/off)

    주의: 이 테스트는 실제로 장치를 제어합니다!
    """
    device_id = real_device_ids["speaker"]

    async with aiohttp.ClientSession() as session:
        api = pysmartthings.SmartThings(_token=smartthings_token, session=session)

        device = await api.get_device(device_id)

        # 현재 상태 확인
        await device.status.refresh()
        initial_state = device.status.switch

        print(f"\n{device.label} 제어 테스트:")
        print(f"  현재 상태: {initial_state}")

        # 상태 반전 테스트
        target_state = "off" if initial_state == "on" else "on"
        print(f"  목표 상태: {target_state}")

        # 명령 실행
        result = await device.command(
            component_id="main",
            capability="switch",
            command=target_state
        )

        print(f"  명령 결과: {result}")

        # 잠시 대기 (장치가 상태를 변경할 시간)
        await asyncio.sleep(2)

        # 상태 재확인
        await device.status.refresh()
        new_state = device.status.switch

        print(f"  변경된 상태: {new_state}")

        # 원래 상태로 복구
        if new_state != initial_state:
            await device.command(
                component_id="main",
                capability="switch",
                command=initial_state
            )
            await asyncio.sleep(2)
            print(f"  복구 완료: {initial_state}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_device_capabilities(smartthings_token, real_device_ids, device_capabilities):
    """각 장치의 capabilities 확인"""
    async with aiohttp.ClientSession() as session:
        api = pysmartthings.SmartThings(_token=smartthings_token, session=session)

        print("\n=== 장치별 Capabilities 확인 ===")

        for name, device_id in real_device_ids.items():
            device = await api.get_device(device_id)

            print(f"\n[{name}] {device.label}:")

            if hasattr(device, "components") and "main" in device.components:
                main_comp = device.components["main"]
                if hasattr(main_comp, "capabilities"):
                    actual_caps = [str(cap).split(".")[-1].replace("'", "").replace(">", "")
                                   for cap in main_comp.capabilities]

                    # 주요 capability 확인
                    has_switch = any("switch" in cap.lower() for cap in actual_caps)
                    print(f"  - Has switch capability: {has_switch}")

                    # 처음 10개 capability 출력
                    print(f"  - Capabilities: {', '.join(actual_caps[:10])}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multiple_devices_status(smartthings_token, test_safe_devices, real_device_ids):
    """여러 장치의 상태를 동시에 조회"""
    async with aiohttp.ClientSession() as session:
        api = pysmartthings.SmartThings(_token=smartthings_token, session=session)

        print("\n=== 안전한 장치들의 현재 상태 ===")

        for device_name in test_safe_devices:
            device_id = real_device_ids[device_name]
            device = await api.get_device(device_id)
            await device.status.refresh()

            print(f"\n[{device_name}] {device.label}:")
            print(f"  - Switch: {device.status.switch if hasattr(device.status, 'switch') else 'N/A'}")
            print(f"  - Power: {device.status.power if hasattr(device.status, 'power') else 'N/A'}W")


if __name__ == "__main__":
    # 개별 실행용
    pytest.main([__file__, "-v", "-s"])