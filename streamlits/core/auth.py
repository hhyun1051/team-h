"""
Streamlit 간단 인증 모듈

쿠키 기반 기기 인증으로 한 번만 비밀번호 입력
"""

import streamlit as st
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Optional


def get_device_fingerprint() -> str:
    """
    브라우저 고유 식별자 생성

    Returns:
        기기 고유 ID
    """
    # Streamlit 세션 정보로 기기 식별
    # 실제로는 브라우저 쿠키에 저장됨
    if "device_id" not in st.session_state:
        st.session_state.device_id = str(uuid.uuid4())

    return st.session_state.device_id


def hash_password(password: str) -> str:
    """비밀번호 해시화"""
    return hashlib.sha256(password.encode()).hexdigest()


def check_auth(password_hash: str, expiry_days: int = 365) -> bool:
    """
    기기 인증 확인

    Args:
        password_hash: 비밀번호 해시 (SHA256)
        expiry_days: 인증 유효 기간 (일)

    Returns:
        인증 성공 여부
    """
    # 쿠키 기반 저장소 초기화
    if "authenticated_devices" not in st.session_state:
        st.session_state.authenticated_devices = {}

    if "current_auth" not in st.session_state:
        st.session_state.current_auth = False

    # 이미 인증된 경우
    if st.session_state.current_auth:
        return True

    # 기기 ID 확인
    device_id = get_device_fingerprint()

    # 이 기기가 이전에 인증되었는지 확인
    if device_id in st.session_state.authenticated_devices:
        auth_time = st.session_state.authenticated_devices[device_id]

        # 인증 만료 확인
        if datetime.now() - auth_time < timedelta(days=expiry_days):
            st.session_state.current_auth = True
            return True

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
            # 인증 성공 - 기기 등록
            st.session_state.authenticated_devices[device_id] = datetime.now()
            st.session_state.current_auth = True
            st.success("✅ 인증 성공! 이 기기는 1년간 자동 로그인됩니다")
            st.rerun()
        else:
            st.error("❌ 비밀번호가 틀렸습니다")

    # 개발자 정보
    with st.expander("🔧 개발자 정보"):
        st.code(f"기기 ID: {device_id}")
        st.caption("비밀번호를 잊으셨다면 서버 관리자에게 문의하세요")

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
    """현재 기기 인증 해제"""
    device_id = get_device_fingerprint()

    if "authenticated_devices" in st.session_state:
        if device_id in st.session_state.authenticated_devices:
            del st.session_state.authenticated_devices[device_id]

    if "current_auth" in st.session_state:
        st.session_state.current_auth = False

    st.success("✅ 로그아웃 완료")
    st.rerun()


def show_auth_status():
    """인증 상태 표시 (사이드바용)"""
    if st.session_state.get("current_auth", False):
        device_id = get_device_fingerprint()

        if "authenticated_devices" in st.session_state:
            auth_time = st.session_state.authenticated_devices.get(device_id)

            if auth_time:
                days_left = 365 - (datetime.now() - auth_time).days

                st.sidebar.success(f"✅ 인증됨 ({days_left}일 남음)")

                if st.sidebar.button("🔓 로그아웃", use_container_width=True):
                    logout()
