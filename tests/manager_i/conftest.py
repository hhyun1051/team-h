"""
Pytest configuration and fixtures for Manager I tests

실제 SmartThings 장치를 사용한 테스트
"""

import os
import sys
from pathlib import Path
from typing import Dict
import pytest
import asyncio
from dotenv import load_dotenv

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 프로젝트 루트의 .env 파일 로드
load_dotenv(project_root / ".env")


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def smartthings_token():
    """SmartThings API 토큰 (환경변수에서 로드)"""
    token = os.getenv("SMARTTHINGS_TOKEN")
    if not token:
        pytest.skip("SMARTTHINGS_TOKEN 환경변수가 설정되지 않았습니다.")
    return token


@pytest.fixture
def real_device_ids():
    """
    실제 장치 ID들
    /root/team-h/tests/모든장치정보.txt 참고
    """
    return {
        # Smart Plugs (ZIGBEE)
        "speaker": "d5ae3413-10a4-4a03-b5e3-eaa0bee64db4",
        "vertical_monitor": "55ca4824-3237-411b-88fd-efb549927553",

        # Entertainment (OCF)
        "projector": "f28bb22f-4768-685b-076b-b9514941498c",

        # Air Quality (OCF)
        "air_purifier": "0897d30e-5cb2-5566-13d5-7de7394061d1",
    }


@pytest.fixture
def device_config(real_device_ids):
    """
    Manager I Agent를 위한 장치 설정
    실제 장치를 방별로 매핑
    """
    return {
        # 스피커를 거실 스피커 콘센트로 사용
        "living_room_speaker_outlet": real_device_ids["speaker"],

        # 프로젝터를 거실 불로 사용 (테스트용)
        "living_room_light": real_device_ids["projector"],

        # 세로모니터를 안방 불로 사용 (테스트용)
        "bedroom_light": real_device_ids["vertical_monitor"],

        # 공기청정기를 화장실 불로 사용 (테스트용)
        "bathroom_light": real_device_ids["air_purifier"],
    }


@pytest.fixture
def device_capabilities():
    """각 장치의 capabilities 정보"""
    return {
        "speaker": ["switch", "powerMeter", "energyMeter"],
        "vertical_monitor": ["switch", "powerMeter", "energyMeter"],
        "projector": [
            "switch",
            "audioVolume",
            "audioMute",
            "tvChannel",
            "mediaInputSource",
        ],
        "air_purifier": [
            "switch",
            "airConditionerFanMode",
            "airQualitySensor",
            "dustSensor",
        ],
    }


@pytest.fixture
def sample_test_scenarios():
    """테스트 시나리오들"""
    return [
        {
            "name": "Turn off speaker",
            "user_input": "거실 스피커 꺼줘",
            "expected_device": "speaker",
            "expected_action": "off",
        },
        {
            "name": "Turn on living room light",
            "user_input": "거실 불 켜줘",
            "expected_device": "projector",
            "expected_action": "on",
        },
        {
            "name": "Turn off bedroom light",
            "user_input": "안방 불 꺼줘",
            "expected_device": "vertical_monitor",
            "expected_action": "off",
        },
        {
            "name": "Check device status",
            "user_input": "거실 불 상태 확인해줘",
            "expected_device": "projector",
            "expected_action": "status",
        },
    ]


@pytest.fixture(scope="session")
def test_safe_devices():
    """
    안전하게 테스트할 수 있는 장치들
    실제로 on/off를 반복해도 문제없는 장치들만 포함
    """
    return [
        "speaker",  # 스피커 콘센트
        "vertical_monitor",  # 모니터 콘센트
    ]


# Pytest markers
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, no external dependencies)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests with real SmartThings API"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take significant time to run"
    )
    config.addinivalue_line(
        "markers", "safe: Tests that are safe to run repeatedly"
    )
    config.addinivalue_line(
        "markers", "dangerous: Tests that should be run carefully (e.g., shutdown)"
    )


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--run-dangerous",
        action="store_true",
        default=False,
        help="Run dangerous tests (e.g., actual system shutdown)"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options"""
    # Skip dangerous tests by default
    if not config.getoption("--run-dangerous"):
        skip_dangerous = pytest.mark.skip(
            reason="need --run-dangerous option to run (potentially destructive)"
        )
        for item in items:
            if "dangerous" in item.keywords:
                item.add_marker(skip_dangerous)


@pytest.fixture
def openai_api_key():
    """OpenAI API Key for LLM tests"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not configured in .env")
    return api_key