"""
Team-H Agent Assert 테스트

최소 하나 이상의 매니저가 활성화되어야 한다는 것을 검증하는 테스트
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.team_h import TeamHAgent


def test_all_managers_disabled():
    """모든 매니저가 비활성화된 경우 AssertionError 발생 확인"""
    print("\n[TEST 1] 모든 매니저 비활성화 시도...")
    try:
        agent = TeamHAgent(
            enable_manager_i=False,
            enable_manager_m=False,
            enable_manager_s=False,
        )
        print("❌ FAILED: AssertionError가 발생하지 않았습니다!")
        return False
    except AssertionError as e:
        print(f"✅ PASSED: 예상된 AssertionError 발생")
        print(f"   에러 메시지: {e}")
        return True
    except Exception as e:
        print(f"❌ FAILED: 예상치 못한 에러 발생: {e}")
        return False


def test_only_manager_m_enabled():
    """Manager M만 활성화된 경우 정상 동작 확인"""
    print("\n[TEST 2] Manager M만 활성화...")
    try:
        agent = TeamHAgent(
            enable_manager_i=False,
            enable_manager_m=True,
            enable_manager_s=False,
        )
        print("✅ PASSED: Manager M만으로 정상 초기화됨")
        print(f"   활성화된 매니저: Manager M")
        print(f"   Router agent created: {agent.router_agent is not None}")
        return True
    except Exception as e:
        print(f"❌ FAILED: 초기화 중 에러 발생: {e}")
        return False


def test_manager_i_and_s_enabled():
    """Manager I와 S가 활성화된 경우 정상 동작 확인"""
    print("\n[TEST 3] Manager I, S 활성화 (M 비활성화)...")
    try:
        # Manager I를 위한 더미 설정 (실제로는 SmartThings 토큰 필요)
        agent = TeamHAgent(
            enable_manager_i=True,
            enable_manager_m=False,
            enable_manager_s=True,
            smartthings_token="dummy_token",
            device_config={"light": "dummy_id"},
            tavily_api_key="dummy_key",
        )
        print("✅ PASSED: Manager I, S로 정상 초기화됨")
        print(f"   활성화된 매니저: Manager I, Manager S")
        print(f"   Router agent created: {agent.router_agent is not None}")
        return True
    except Exception as e:
        print(f"❌ FAILED: 초기화 중 에러 발생: {e}")
        return False


def test_all_managers_enabled():
    """모든 매니저가 활성화된 경우 정상 동작 확인"""
    print("\n[TEST 4] 모든 매니저 활성화...")
    try:
        agent = TeamHAgent(
            enable_manager_i=True,
            enable_manager_m=True,
            enable_manager_s=True,
            smartthings_token="dummy_token",
            device_config={"light": "dummy_id"},
            tavily_api_key="dummy_key",
        )
        print("✅ PASSED: 모든 매니저로 정상 초기화됨")
        print(f"   활성화된 매니저: Manager I, Manager M, Manager S")
        print(f"   Router agent created: {agent.router_agent is not None}")
        return True
    except Exception as e:
        print(f"❌ FAILED: 초기화 중 에러 발생: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Team-H Agent Assert 테스트 시작")
    print("=" * 60)

    results = []

    # 테스트 실행
    results.append(("모든 매니저 비활성화", test_all_managers_disabled()))
    results.append(("Manager M만 활성화", test_only_manager_m_enabled()))
    results.append(("Manager I, S 활성화", test_manager_i_and_s_enabled()))
    results.append(("모든 매니저 활성화", test_all_managers_enabled()))

    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\n총 {passed}/{total} 테스트 통과")

    if passed == total:
        print("\n🎉 모든 테스트가 통과했습니다!")
    else:
        print(f"\n⚠️ {total - passed}개의 테스트가 실패했습니다.")
