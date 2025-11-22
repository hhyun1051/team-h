# manager_t.py

"""
Manager T Agent - 캘린더 및 알림 관리 에이전트

Manager T는 시간 관리와 일정 관리를 담당하는 에이전트입니다:
- Google Calendar 연동 (일정 CRUD)
- 자연어 시간 파싱 ("내일 아침", "다음주 월요일" 등)
- 스마트 알림 및 일정 요약
- 반복 일정 관리

ManagerBase를 상속받아 공통 로직을 재사용합니다.
HumanInTheLoopMiddleware를 통해 일정 생성/수정/삭제 작업에 대한 승인을 요구합니다.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import pytz

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Agents import (__init__.py 활용)
from agents import ManagerBase
from agents.context import TeamHContext
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool, ToolRuntime
from pydantic import BaseModel, Field

# Google Calendar API
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLE_AVAILABLE = True
except ImportError:
    print("⚠️  Google API libraries not installed. Please run: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")
    GOOGLE_AVAILABLE = False


# ============================================================================
# Google Calendar 설정
# ============================================================================

SCOPES = ['https://www.googleapis.com/auth/calendar']
KST = pytz.timezone('Asia/Seoul')


# ============================================================================
# Pydantic 스키마
# ============================================================================

class CalendarEventSchema(BaseModel):
    """일정 생성/수정용 스키마"""
    title: str = Field(description="일정 제목")
    start_time: str = Field(description="시작 시간 (ISO 8601 형식)")
    end_time: Optional[str] = Field(default=None, description="종료 시간 (없으면 시작+1시간)")
    description: Optional[str] = Field(default="", description="상세 설명")
    location: Optional[str] = Field(default="", description="장소")
    reminders: List[int] = Field(default=[30], description="알림 시간 (분 단위)")


# ============================================================================
# Manager T 클래스
# ============================================================================

class ManagerT(ManagerBase):
    """Manager T 에이전트 클래스 - 캘린더 및 시간 관리 전문"""

    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        temperature: float = 0.7,
        google_credentials_path: Optional[str] = None,
        google_token_path: Optional[str] = None,
        calendar_id: str = "primary",
        additional_tools: Optional[List] = None,
        middleware: Optional[List] = None,
    ):
        """
        Manager T 에이전트 초기화

        Args:
            model_name: 사용할 LLM 모델 이름 (기본값: gpt-4o-mini)
            temperature: 모델 temperature 설정
            google_credentials_path: Google OAuth credentials.json 경로
            google_token_path: Google OAuth token.json 저장 경로
            calendar_id: 사용할 Google Calendar ID (기본값: primary)
            additional_tools: 핸드오프 등 추가 툴 리스트
            middleware: 외부에서 주입할 middleware 리스트
        """
        if not GOOGLE_AVAILABLE:
            raise ImportError("Google API libraries are required for Manager T. Please install them first.")

        # HITL 미들웨어 생성
        hitl_middleware = HumanInTheLoopMiddleware(
            interrupt_on={
                # 쓰기/수정/삭제 작업만 승인 필요
                "create_calendar_event": True,
                "update_calendar_event": True,
                "delete_calendar_event": True,
            },
            description_prefix="📅 Calendar operation pending approval",
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
            # Google Calendar 초기화를 위한 파라미터 전달
            google_credentials_path=google_credentials_path,
            google_token_path=google_token_path,
            calendar_id=calendar_id,
        )

        # 추가 초기화 메시지
        print(f"    - Calendar ID: {self.calendar_id}")
        print(f"    - HITL: Enabled for write operations")

    def _pre_init_hook(self, **kwargs):
        """Google Calendar 서비스 초기화 (툴 생성 전에 필요)"""
        self.google_credentials_path = kwargs.get("google_credentials_path") or os.getenv("GOOGLE_CALENDAR_CREDENTIALS_PATH", "credentials.json")
        self.google_token_path = kwargs.get("google_token_path") or os.getenv("GOOGLE_CALENDAR_TOKEN_PATH", "/root/team-h/.credentials/calendar_token.json")
        self.calendar_id = kwargs.get("calendar_id", "primary")

        # Google Calendar 서비스 생성
        try:
            self.calendar_service = self._get_calendar_service()
            print(f"[✅] Google Calendar API connected")
        except Exception as e:
            print(f"[⚠️] Google Calendar API connection failed: {e}")
            print(f"[⚠️] Manager T will have limited functionality")
            self.calendar_service = None


    def _get_calendar_service(self):
        """Google Calendar API 서비스 생성"""
        creds = None
        token_path = self.google_token_path

        # 토큰 파일이 있으면 로드
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)

        # 토큰이 없거나 만료되었으면 새로 인증
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.google_credentials_path):
                    raise FileNotFoundError(
                        f"Google credentials file not found at {self.google_credentials_path}. "
                        "Please download credentials.json from Google Cloud Console."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.google_credentials_path, SCOPES
                )
                # 고정 포트 사용 (8080) - Google Cloud Console의 redirect URI와 일치
                creds = flow.run_local_server(port=8080)

            # 토큰 저장
            os.makedirs(os.path.dirname(token_path), exist_ok=True)
            with open(token_path, 'w') as token:
                token.write(creds.to_json())

        service = build('calendar', 'v3', credentials=creds)
        return service

    def _list_events_internal(
        self,
        start_date: str,
        end_date: str,
        max_results: int = 10
    ) -> str:
        """
        Internal method to list calendar events.
        This is called by multiple @tool functions.
        """
        if not self.calendar_service:
            return "❌ Google Calendar service is not available. Please check authentication."

        try:
            print(f"[DEBUG] _list_events_internal called")
            print(f"[DEBUG] API request - timeMin: {start_date}, timeMax: {end_date}")

            # 이벤트 조회
            events_result = self.calendar_service.events().list(
                calendarId=self.calendar_id,
                timeMin=start_date,
                timeMax=end_date,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()

            events = events_result.get('items', [])

            if not events:
                return f"📅 해당 기간에 일정이 없습니다."

            # 결과 포맷팅
            formatted_events = [f"📅 일정 목록 ({len(events)}개):\n"]
            for i, event in enumerate(events, 1):
                title = event.get('summary', '(제목 없음)')
                start = event.get('start', {}).get('dateTime', event.get('start', {}).get('date'))
                end = event.get('end', {}).get('dateTime', event.get('end', {}).get('date'))
                event_id = event.get('id')

                # 시간 파싱 및 포맷팅
                try:
                    start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                    # KST로 변환
                    start_dt_kst = start_dt.astimezone(KST)

                    # 종료 시간도 파싱
                    if end:
                        end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
                        end_dt_kst = end_dt.astimezone(KST)

                        # 시작-종료 시간 표시
                        formatted_events.append(
                            f"{i}. 📌 {title}\n"
                            f"   ⏰ 시작: {start_dt_kst.strftime('%Y-%m-%d %H:%M')}\n"
                            f"   ⏰ 종료: {end_dt_kst.strftime('%Y-%m-%d %H:%M')}\n"
                            f"   🆔 {event_id}"
                        )
                    else:
                        # 종료 시간 없음 (시작 시간만)
                        formatted_events.append(
                            f"{i}. 📌 {title}\n"
                            f"   ⏰ {start_dt_kst.strftime('%Y-%m-%d %H:%M')}\n"
                            f"   🆔 {event_id}"
                        )

                except Exception as e:
                    formatted_events.append(
                        f"{i}. 📌 {title}\n"
                        f"   ⏰ {start}\n"
                        f"   🆔 {event_id}"
                    )

            return "\n\n".join(formatted_events)

        except HttpError as error:
            if error.resp.status == 401:
                return "⚠️ 인증이 만료되었습니다. 다시 로그인해주세요."
            elif error.resp.status == 403:
                return "⚠️ 권한이 부족합니다. Calendar 권한을 확인해주세요."
            else:
                return f"❌ 일정 조회 실패: {error}"
        except Exception as e:
            return f"❌ 일정 조회 중 오류 발생: {str(e)}"

    def _create_tools(self) -> List:
        """캘린더 관리 관련 툴 생성"""

        @tool
        def create_calendar_event(
            title: str,
            start_time: str,
            end_time: Optional[str] = None,
            description: Optional[str] = "",
            location: Optional[str] = "",
            reminders_minutes: Optional[List[int]] = None,
            runtime: ToolRuntime[TeamHContext] = None
        ) -> str:
            """
            Create a new event in Google Calendar.

            Args:
                title: Event title (e.g., "빨래하기")
                runtime: Automatically injected runtime context
                start_time: Start time in ISO 8601 format (e.g., "2025-11-15T09:00:00+09:00")
                end_time: End time (optional, defaults to start_time + 1 hour)
                description: Event description
                location: Event location
                reminders_minutes: Reminder times in minutes (e.g., [30, 60] for 30min and 1hr before)

            Returns:
                Confirmation message with event ID
            """
            print(f"[DEBUG] create_calendar_event called with title='{title}', start_time='{start_time}'")

            if not self.calendar_service:
                print(f"[DEBUG] Calendar service is None!")
                return "❌ Google Calendar service is not available. Please check authentication."

            try:
                # ISO 8601 형식의 시간을 파싱
                start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))

                # end_time이 없으면 start_time + 1시간
                if end_time:
                    end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                else:
                    end_dt = start_dt + timedelta(hours=1)

                # 이벤트 구조 생성
                event = {
                    'summary': title,
                    'description': description,
                    'location': location,
                    'start': {
                        'dateTime': start_dt.isoformat(),
                        'timeZone': 'Asia/Seoul',
                    },
                    'end': {
                        'dateTime': end_dt.isoformat(),
                        'timeZone': 'Asia/Seoul',
                    },
                }
                print(f"[DEBUG] Event object created: {event}")

                # 알림 설정
                if reminders_minutes:
                    event['reminders'] = {
                        'useDefault': False,
                        'overrides': [
                            {'method': 'popup', 'minutes': minutes}
                            for minutes in reminders_minutes
                        ],
                    }
                else:
                    # 기본 30분 전 알림
                    event['reminders'] = {
                        'useDefault': False,
                        'overrides': [{'method': 'popup', 'minutes': 30}],
                    }
                print(f"[DEBUG] Reminders configured")

                # 이벤트 생성
                print(f"[DEBUG] Calling Google Calendar API...")
                event_result = self.calendar_service.events().insert(
                    calendarId=self.calendar_id,
                    body=event
                ).execute()
                print(f"[DEBUG] API call successful! Result: {event_result}")

                event_id = event_result.get('id')
                event_link = event_result.get('htmlLink')

                success_msg = (
                    f"✅ 일정이 등록되었습니다!\n"
                    f"📌 제목: {title}\n"
                    f"⏰ 시작: {start_dt.strftime('%Y-%m-%d %H:%M')}\n"
                    f"⏰ 종료: {end_dt.strftime('%Y-%m-%d %H:%M')}\n"
                    f"🔗 링크: {event_link}\n"
                    f"🆔 ID: {event_id}"
                )
                print(f"[DEBUG] Success message created: {success_msg}")
                return success_msg

            except HttpError as error:
                print(f"[DEBUG] HttpError occurred: {error}")
                print(f"[DEBUG] Error status: {error.resp.status}")
                print(f"[DEBUG] Error details: {error.resp}")
                if error.resp.status == 401:
                    return "⚠️ 인증이 만료되었습니다. 다시 로그인해주세요."
                elif error.resp.status == 403:
                    return "⚠️ 권한이 부족합니다. Calendar 권한을 확인해주세요."
                else:
                    return f"❌ 일정 등록 실패: {error}"
            except Exception as e:
                print(f"[DEBUG] Exception occurred: {type(e).__name__}: {str(e)}")
                import traceback
                print(f"[DEBUG] Traceback:\n{traceback.format_exc()}")
                return f"❌ 일정 등록 중 오류 발생: {str(e)}"

        @tool
        def list_calendar_events(
            start_date: str,
            end_date: str,
            max_results: int = 10
        , runtime: ToolRuntime[TeamHContext] = None) -> str:
            """
            List calendar events within a date range.

            Args:
                start_date: Start date in ISO 8601 format (e.g., "2025-11-14T00:00:00+09:00")
                end_date: End date in ISO 8601 format
                max_results: Maximum number of events to return (default: 10)

            Returns:
                Formatted list of events
            """
            return self._list_events_internal(start_date, end_date, max_results)

        @tool
        def update_calendar_event(
            event_id: str,
            title: Optional[str] = None,
            start_time: Optional[str] = None,
            end_time: Optional[str] = None,
            description: Optional[str] = None
        , runtime: ToolRuntime[TeamHContext] = None) -> str:
            """
            Update an existing calendar event.

            Args:
                event_id: Google Calendar Event ID
                title: New title (optional)
                start_time: New start time in ISO 8601 format (optional)
                end_time: New end time (optional)
                description: New description (optional)

            Returns:
                Confirmation message
            """
            if not self.calendar_service:
                return "❌ Google Calendar service is not available. Please check authentication."

            try:
                # 기존 이벤트 가져오기
                event = self.calendar_service.events().get(
                    calendarId=self.calendar_id,
                    eventId=event_id
                ).execute()

                # 업데이트할 필드만 수정
                if title:
                    event['summary'] = title
                if start_time:
                    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    event['start'] = {
                        'dateTime': start_dt.isoformat(),
                        'timeZone': 'Asia/Seoul',
                    }
                if end_time:
                    end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                    event['end'] = {
                        'dateTime': end_dt.isoformat(),
                        'timeZone': 'Asia/Seoul',
                    }
                if description:
                    event['description'] = description

                # 이벤트 업데이트
                updated_event = self.calendar_service.events().update(
                    calendarId=self.calendar_id,
                    eventId=event_id,
                    body=event
                ).execute()

                return (
                    f"✅ 일정이 수정되었습니다!\n"
                    f"📌 제목: {updated_event.get('summary')}\n"
                    f"🆔 ID: {event_id}"
                )

            except HttpError as error:
                if error.resp.status == 404:
                    return f"⚠️ 일정을 찾을 수 없습니다 (ID: {event_id})"
                else:
                    return f"❌ 일정 수정 실패: {error}"
            except Exception as e:
                return f"❌ 일정 수정 중 오류 발생: {str(e)}"

        @tool
        def delete_calendar_event(event_id: str, runtime: ToolRuntime[TeamHContext] = None) -> str:
            """
            Delete a calendar event.

            Args:
                event_id: Google Calendar Event ID to delete

            Returns:
                Confirmation message
            """
            if not self.calendar_service:
                return "❌ Google Calendar service is not available. Please check authentication."

            try:
                # 이벤트 삭제
                self.calendar_service.events().delete(
                    calendarId=self.calendar_id,
                    eventId=event_id
                ).execute()

                return f"✅ 일정이 삭제되었습니다 (ID: {event_id})"

            except HttpError as error:
                if error.resp.status == 404:
                    return f"⚠️ 일정을 찾을 수 없습니다 (ID: {event_id})"
                elif error.resp.status == 410:
                    return f"⚠️ 이미 삭제된 일정입니다 (ID: {event_id})"
                else:
                    return f"❌ 일정 삭제 실패: {error}"
            except Exception as e:
                return f"❌ 일정 삭제 중 오류 발생: {str(e)}"

        @tool
        def get_current_datetime(runtime: ToolRuntime[TeamHContext] = None) -> str:
            """
            Get the current date and time in KST (Korea Standard Time).

            IMPORTANT: Always call this tool FIRST before processing any time-related requests
            to ensure you have the accurate current time.

            Returns:
                Current date and time information with examples for relative time parsing
            """
            now_kst = datetime.now(KST)

            return f"""📅 현재 시간 정보 (KST - Asia/Seoul):
- 현재 날짜: {now_kst.strftime('%Y-%m-%d')} ({now_kst.strftime('%A')})
- 현재 시간: {now_kst.strftime('%H:%M:%S')}
- ISO 8601: {now_kst.isoformat()}

상대적 시간 표현 해석 기준:
- "오늘" (today) = {now_kst.strftime('%Y-%m-%d')}
- "내일" (tomorrow) = {(now_kst + timedelta(days=1)).strftime('%Y-%m-%d')}
- "모레" (day after tomorrow) = {(now_kst + timedelta(days=2)).strftime('%Y-%m-%d')}
- "오늘 오후 1시" = {now_kst.strftime('%Y-%m-%d')}T13:00:00+09:00
- "내일 아침" = {(now_kst + timedelta(days=1)).strftime('%Y-%m-%d')}T09:00:00+09:00

현재 연도: {now_kst.year} (일정 생성 시 반드시 이 연도 사용!)"""

        @tool
        def get_today_events(runtime: ToolRuntime[TeamHContext] = None) -> str:
            """
            Get today's calendar events.

            Returns:
                Formatted list of today's events
            """
            # 오늘 날짜 범위 생성 (KST 기준)
            now = datetime.now(KST)
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            today_end = now.replace(hour=23, minute=59, second=59, microsecond=999999)

            print(f"[DEBUG] get_today_events called")
            print(f"[DEBUG] Current time (KST): {now}")
            print(f"[DEBUG] Search range: {today_start.isoformat()} ~ {today_end.isoformat()}")

            return self._list_events_internal(
                start_date=today_start.isoformat(),
                end_date=today_end.isoformat(),
                max_results=20
            )

        @tool
        def get_tomorrow_events(runtime: ToolRuntime[TeamHContext] = None) -> str:
            """
            Get tomorrow's calendar events.

            Returns:
                Formatted list of tomorrow's events
            """
            # 내일 날짜 범위 생성 (KST 기준)
            now = datetime.now(KST)
            tomorrow = now + timedelta(days=1)
            tomorrow_start = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
            tomorrow_end = tomorrow.replace(hour=23, minute=59, second=59, microsecond=999999)

            return self._list_events_internal(
                start_date=tomorrow_start.isoformat(),
                end_date=tomorrow_end.isoformat(),
                max_results=20
            )

        @tool
        def get_week_events(runtime: ToolRuntime[TeamHContext] = None) -> str:
            """
            Get this week's calendar events.

            Returns:
                Formatted list of this week's events
            """
            # 이번주 범위 생성 (KST 기준, 월요일 시작)
            now = datetime.now(KST)
            # 월요일까지 되돌리기
            days_since_monday = now.weekday()
            week_start = (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
            week_end = (week_start + timedelta(days=6)).replace(hour=23, minute=59, second=59, microsecond=999999)

            return self._list_events_internal(
                start_date=week_start.isoformat(),
                end_date=week_end.isoformat(),
                max_results=50
            )

        return [
            get_current_datetime,  # 현재 시간 확인 (항상 먼저 호출)
            create_calendar_event,
            list_calendar_events,
            update_calendar_event,
            delete_calendar_event,
            get_today_events,
            get_tomorrow_events,
            get_week_events,
        ]


def create_manager_t_agent(**kwargs) -> ManagerT:
    """
    Manager T 에이전트 생성 헬퍼 함수

    Args:
        **kwargs: ManagerT 초기화 파라미터

    Returns:
        ManagerT 인스턴스
    """
    return ManagerT(**kwargs)


# 싱글톤 인스턴스 (선택적 사용)
_manager_t_agent_instance = None


def get_manager_t_agent(**kwargs) -> ManagerT:
    """
    Manager T 에이전트 싱글톤 인스턴스 반환

    Args:
        **kwargs: ManagerT 초기화 파라미터 (처음 생성 시에만 사용됨)

    Returns:
        ManagerT 인스턴스
    """
    global _manager_t_agent_instance
    if _manager_t_agent_instance is None:
        _manager_t_agent_instance = ManagerT(**kwargs)
    return _manager_t_agent_instance
