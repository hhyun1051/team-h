"""
FastAPI 백엔드 클라이언트

SSE (Server-Sent Events) 기반 스트리밍 클라이언트
"""

import json
import requests
from typing import Dict, Any, Iterator, Optional, List


class FastAPIClient:
    """FastAPI 백엔드와 통신하는 클라이언트"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Args:
            base_url: FastAPI 서버 주소 (기본: http://localhost:8000)
        """
        self.base_url = base_url.rstrip("/")

    def chat_stream(
        self,
        message: str,
        thread_id: str,
        user_id: str = "default_user",
        session_id: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        채팅 메시지를 전송하고 스트리밍 응답을 받음

        Args:
            message: 사용자 메시지
            thread_id: 대화 스레드 ID (상태 식별용)
            user_id: 사용자 ID
            session_id: Langfuse 세션 ID (없으면 thread_id 사용)

        Yields:
            SSE 이벤트 딕셔너리
        """
        url = f"{self.base_url}/chat/stream"

        payload = {
            "message": message,
            "thread_id": thread_id,
            "user_id": user_id,
            "session_id": session_id,
        }

        # None 값 제거
        payload = {k: v for k, v in payload.items() if v is not None}

        # SSE 스트림 요청
        with requests.post(url, json=payload, stream=True, timeout=300) as response:
            response.raise_for_status()

            # SSE 이벤트 파싱
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("data: "):
                    data_str = line[6:]  # "data: " 제거
                    try:
                        data = json.loads(data_str)
                        yield data
                    except json.JSONDecodeError:
                        print(f"[⚠️] Failed to parse SSE data: {data_str}")
                        continue

    def resume_stream(
        self,
        thread_id: str,
        decisions: List[Dict[str, Any]],
        user_id: str = "default_user",
        session_id: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        HITL 인터럽트를 재개하고 스트리밍 응답을 받음

        Args:
            thread_id: 대화 스레드 ID (인터럽트된 대화 식별)
            decisions: 승인/거부 결정 리스트
            user_id: 사용자 ID
            session_id: Langfuse 세션 ID

        Yields:
            SSE 이벤트 딕셔너리
        """
        url = f"{self.base_url}/chat/resume"

        payload = {
            "thread_id": thread_id,
            "decisions": decisions,
            "user_id": user_id,
            "session_id": session_id,
        }

        # None 값 제거
        payload = {k: v for k, v in payload.items() if v is not None}

        # SSE 스트림 요청
        with requests.post(url, json=payload, stream=True, timeout=300) as response:
            response.raise_for_status()

            # SSE 이벤트 파싱
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                        yield data
                    except json.JSONDecodeError:
                        print(f"[⚠️] Failed to parse SSE data: {data_str}")
                        continue

    def get_state(self, thread_id: str) -> Dict[str, Any]:
        """
        특정 thread의 현재 상태 조회

        Args:
            thread_id: 대화 스레드 ID

        Returns:
            그래프 상태 딕셔너리
        """
        url = f"{self.base_url}/state/{thread_id}"

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        return response.json()
