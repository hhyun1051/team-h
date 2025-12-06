"""
Streamlit 간단 인증 모듈

Cookie + 파일 기반 기기 인증으로 한 번만 비밀번호 입력
- Cookie: 브라우저별 고유 기기 ID 저장 (영구 유지)
- 파일: 인증된 기기 목록 저장 (서버 재시작해도 유지)

쿠키 비동기 처리:
- 쿠키 설정은 즉시 반영되지 않음
- rerun 후 다음 실행에서 쿠키 읽기
- 2단계 프로세스: 설정 → rerun → 읽기
"""

import streamlit as st
import hashlib
import uuid
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict
import extra_streamlit_components as stx


# ============================================================================
# 파일 기반 영구 저장소
# ============================================================================

AUTH_FILE = Path.home() / ".team_h_auth.json"


def load_auth_store() -> Dict[str, str]:
    """인증 정보 파일 로드"""
    if not AUTH_FILE.exists():
        return {}

    try:
        with open(AUTH_FILE, "r") as f:
            data = json.load(f)
            # ISO 문자열을 datetime으로 변환하지 않고 그대로 유지
            return data
    except (json.JSONDecodeError, IOError):
        return {}


def save_auth_store(auth_store: Dict[str, str]) -> None:
    """인증 정보 파일 저장"""
    try:
        # 디렉토리 생성 (없으면)
        AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)

        with open(AUTH_FILE, "w") as f:
            json.dump(auth_store, f, indent=2)

        # 파일 권한 설정 (소유자만 읽기/쓰기)
        AUTH_FILE.chmod(0o600)
    except IOError as e:
        st.warning(f"⚠️ 인증 정보 저장 실패: {e}")


# 쿠키 매니저 전역 인스턴스 (앱 시작 시 한 번만 초기화)
_cookie_manager = None


def get_cookie_manager():
    """쿠키 매니저 싱글톤"""
    global _cookie_manager
    if _cookie_manager is None:
        _cookie_manager = stx.CookieManager()
    return _cookie_manager


def get_device_fingerprint() -> str:
    """
    기기 고유 식별자 생성 및 관리 (Cookie 기반)

    브라우저 쿠키에 기기 ID 저장
    2단계 프로세스로 비동기 문제 해결

    전략:
    1. session_state 확인 (빠른 접근)
    2. 쿠키에서 device_id 로드
    3. 없으면:
       a. 새 device_id 생성
       b. 쿠키에 저장 요청
       c. cookie_needs_init 플래그 설정
       d. rerun (다음 실행에서 쿠키 읽기 가능)

    Returns:
        기기 고유 ID
    """
    # 1. session_state에 이미 있으면 재사용
    if "device_id" in st.session_state and st.session_state.device_id:
        return st.session_state.device_id

    # 2. 쿠키 매니저 초기화
    cookie_manager = get_cookie_manager()

    # 3. 쿠키에서 device_id 로드 시도 (직접 get 사용)
    device_id = cookie_manager.get("team_h_device_id")

    # 4. 쿠키에 없으면 새로 생성하고 설정
    if not device_id:
        # 쿠키 초기화가 필요한지 확인 (무한 루프 방지)
        if not st.session_state.get("cookie_init_attempted", False):
            # 새 device_id 생성
            device_id = f"device_{uuid.uuid4().hex[:12]}"

            # 쿠키에 저장 (다음 rerun에서 읽을 수 있음)
            cookie_manager.set(
                "team_h_device_id",
                device_id,
                expires_at=datetime.now() + timedelta(days=365),
                key="set_device_id"
            )

            # 초기화 시도 플래그 설정
            st.session_state.cookie_init_attempted = True
            st.session_state.device_id = device_id

            # 쿠키가 설정되도록 rerun
            st.rerun()
        else:
            # 이미 초기화 시도했는데도 쿠키 없음
            # 임시 device_id 사용 (쿠키 비활성화 상태)
            device_id = st.session_state.get("device_id", f"temp_{uuid.uuid4().hex[:8]}")

    # 5. session_state에 캐싱
    st.session_state.device_id = device_id

    return device_id


def hash_password(password: str) -> str:
    """비밀번호 해시화"""
    return hashlib.sha256(password.encode()).hexdigest()


def show_debug_info(show_always: bool = False):
    """
    개발자 디버깅 정보 표시

    Args:
        show_always: True면 항상 표시, False면 인증된 경우에만 표시
    """
    # 인증되지 않았고 show_always=False면 표시 안 함
    if not show_always and not st.session_state.get("current_auth", False):
        return

    device_id = get_device_fingerprint()

    with st.expander("🔧 개발자 정보"):
        st.code(f"기기 ID: {device_id}")

        # 쿠키 정보
        cookie_manager = get_cookie_manager()
        device_cookie = cookie_manager.get("team_h_device_id")
        all_cookies = cookie_manager.get_all()
        st.json({
            "team_h_device_id (직접 get)": device_cookie,
            "all_cookies (get_all)": all_cookies
        })

        # 인증 파일 정보
        auth_store = load_auth_store()
        st.json({"auth_store": auth_store})

        st.info("💡 쿠키가 브라우저에 저장되어 다음에 자동 로그인됩니다!")
        st.caption("비밀번호를 잊으셨다면 서버 관리자에게 문의하세요")


def check_auth(password_hash: str, expiry_days: int = 365) -> bool:
    """
    기기 인증 확인 (파일 기반 영구 저장)

    Args:
        password_hash: 비밀번호 해시 (SHA256)
        expiry_days: 인증 유효 기간 (일)

    Returns:
        인증 성공 여부
    """
    # 현재 세션에서 이미 인증됨
    if st.session_state.get("current_auth", False):
        show_debug_info()  # 디버그 정보 표시
        return True

    # 기기 ID 확인
    device_id = get_device_fingerprint()

    # 파일에서 인증 정보 로드
    auth_store = load_auth_store()

    # 이 기기가 이전에 인증되었는지 확인
    if device_id in auth_store:
        auth_time_str = auth_store[device_id]

        try:
            # ISO 형식 문자열을 datetime으로 변환
            auth_time = datetime.fromisoformat(auth_time_str)

            # 인증 만료 확인
            if datetime.now() - auth_time < timedelta(days=expiry_days):
                st.session_state.current_auth = True
                show_debug_info()  # 디버그 정보 표시
                return True
            else:
                # 만료된 인증 정보 삭제
                del auth_store[device_id]
                save_auth_store(auth_store)
        except (ValueError, TypeError):
            # 잘못된 시간 형식이면 삭제
            del auth_store[device_id]
            save_auth_store(auth_store)

    # 인증 UI 표시
    st.markdown("## 🔐 기기 인증")
    st.caption("한 번만 입력하면 이 기기에서 1년간 자동 로그인됩니다")

    col1, col2 = st.columns([3, 1])

    with col1:
        password = st.text_input(
            "비밀번호",
            type="password",
            key="auth_password",
            placeholder="비밀번호를 입력하세요"
        )

    with col2:
        st.write("")  # 버튼 정렬용
        login_btn = st.button("🔓 로그인", use_container_width=True)

    if login_btn:
        if hash_password(password) == password_hash:
            # 인증 성공 - 파일에 기기 등록
            auth_store = load_auth_store()
            auth_store[device_id] = datetime.now().isoformat()
            save_auth_store(auth_store)

            st.session_state.current_auth = True
            st.success("✅ 인증 성공! 이 기기는 1년간 자동 로그인됩니다")
            st.rerun()
        else:
            st.error("❌ 비밀번호가 틀렸습니다")

    return False


def simple_auth(password: str = "teamh2024", expiry_days: int = 365) -> bool:
    """
    간단한 기기 인증 (비밀번호 1개)

    Args:
        password: 인증 비밀번호
        expiry_days: 인증 유효 기간 (기본: 365일)

    Returns:
        인증 성공 여부

    Example:
        >>> from streamlits.core.auth import simple_auth
        >>>
        >>> if not simple_auth(password="your_password"):
        >>>     st.stop()
        >>>
        >>> # 인증 성공 후 코드
        >>> st.write("인증된 사용자만 볼 수 있습니다!")
    """
    password_hash = hash_password(password)
    return check_auth(password_hash, expiry_days)


def logout():
    """현재 기기 인증 해제 (파일에서도 삭제)"""
    device_id = get_device_fingerprint()

    # 파일에서 삭제
    auth_store = load_auth_store()
    if device_id in auth_store:
        del auth_store[device_id]
        save_auth_store(auth_store)

    # 세션에서도 삭제
    st.session_state.current_auth = False

    st.success("✅ 로그아웃 완료")
    st.rerun()


def show_auth_status():
    """인증 상태 표시 (사이드바용) - 파일에서 로드"""
    if st.session_state.get("current_auth", False):
        device_id = get_device_fingerprint()

        # 파일에서 인증 시간 로드
        auth_store = load_auth_store()
        auth_time_str = auth_store.get(device_id)

        if auth_time_str:
            try:
                auth_time = datetime.fromisoformat(auth_time_str)
                days_left = 365 - (datetime.now() - auth_time).days

                st.sidebar.success(f"✅ 인증됨 ({days_left}일 남음)")

                if st.sidebar.button("🔓 로그아웃", use_container_width=True):
                    logout()
            except (ValueError, TypeError):
                pass
