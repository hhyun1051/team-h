# manager_s.py

"""
Manager S Agent - 웹 검색 에이전트

Manager S는 외부 웹 검색을 담당하는 에이전트입니다:
- Tavily Search API를 통한 웹 검색
- 실시간 정보 조회
- 검색 결과 요약

ManagerBase를 상속받아 공통 로직을 재사용합니다.
"""

import sys
from pathlib import Path
from typing import Optional, List

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from agents.base_manager import ManagerBase
from langchain.tools import tool

# Tavily Search
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
except ImportError:
    print("⚠️  langchain-community가 설치되지 않았습니다. pip install langchain-community를 실행하세요.")
    TavilySearchResults = None


class ManagerS(ManagerBase):
    """Manager S 에이전트 클래스 - 웹 검색 전문"""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        tavily_api_key: Optional[str] = None,
        max_results: int = 5,
        additional_tools: Optional[List] = None,
    ):
        """
        Manager S 에이전트 초기화

        Args:
            model_name: 사용할 LLM 모델 이름 (기본값: gpt-4o-mini)
            temperature: 모델 temperature 설정
            tavily_api_key: Tavily API 키
            max_results: 검색 결과 최대 개수
            additional_tools: 핸드오프 등 추가 툴 리스트
        """
        # 특수 파라미터 검증 및 저장
        if not tavily_api_key:
            raise ValueError("Tavily API key is required")

        self.tavily_api_key = tavily_api_key
        self.max_results = max_results

        # 베이스 클래스 초기화 (공통 로직)
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            additional_tools=additional_tools,
            middleware=None,  # 검색은 안전한 작업이므로 HITL 불필요
        )

        # 추가 초기화 메시지
        print(f"    - Max results: {self.max_results}")

    def _create_tools(self) -> List:
        """검색 관련 툴 생성"""

        @tool
        def search_web(query: str, max_results: Optional[int] = None) -> str:
            """
            Search the web for information using Tavily Search API.

            Args:
                query: Search query string
                max_results: Maximum number of results to return (default: uses agent's max_results)

            Returns:
                Formatted search results with titles, URLs, and snippets
            """
            try:
                # TavilySearchResults 초기화
                if TavilySearchResults is None:
                    return "❌ Tavily Search tool is not available. Please install langchain-community."

                search_tool = TavilySearchResults(
                    api_key=self.tavily_api_key,
                    max_results=max_results or self.max_results,
                )

                # 검색 실행
                results = search_tool.invoke({"query": query})

                if not results:
                    return f"No results found for query: '{query}'"

                # 결과 포맷팅
                formatted_results = []
                for i, result in enumerate(results, 1):
                    title = result.get("title", "No title")
                    url = result.get("url", "No URL")
                    content = result.get("content", "No content")

                    formatted_results.append(
                        f"### Result {i}: {title}\n"
                        f"**URL:** {url}\n"
                        f"**Content:** {content}\n"
                    )

                return "\n".join(formatted_results)

            except Exception as e:
                return f"❌ Error during web search: {str(e)}"

        @tool
        def search_news(query: str, max_results: Optional[int] = None) -> str:
            """
            Search for news articles using Tavily Search API.

            Args:
                query: News search query string
                max_results: Maximum number of results to return (default: uses agent's max_results)

            Returns:
                Formatted news results with titles, URLs, and snippets
            """
            try:
                # TavilySearchResults 초기화 (news 모드)
                if TavilySearchResults is None:
                    return "❌ Tavily Search tool is not available. Please install langchain-community."

                search_tool = TavilySearchResults(
                    api_key=self.tavily_api_key,
                    max_results=max_results or self.max_results,
                    search_depth="advanced",  # 뉴스 검색 시 더 깊은 검색
                )

                # 검색 실행
                results = search_tool.invoke({"query": query})

                if not results:
                    return f"No news found for query: '{query}'"

                # 결과 포맷팅
                formatted_results = []
                for i, result in enumerate(results, 1):
                    title = result.get("title", "No title")
                    url = result.get("url", "No URL")
                    content = result.get("content", "No content")

                    formatted_results.append(
                        f"### News {i}: {title}\n"
                        f"**URL:** {url}\n"
                        f"**Content:** {content}\n"
                    )

                return "\n".join(formatted_results)

            except Exception as e:
                return f"❌ Error during news search: {str(e)}"

        return [search_web, search_news]


def create_manager_s_agent(**kwargs) -> ManagerS:
    """
    Manager S 에이전트 생성 헬퍼 함수

    Args:
        **kwargs: ManagerS 초기화 파라미터

    Returns:
        ManagerS 인스턴스
    """
    return ManagerS(**kwargs)


# 싱글톤 인스턴스 (선택적 사용)
_manager_s_agent_instance = None


def get_manager_s_agent(**kwargs) -> ManagerS:
    """
    Manager S 에이전트 싱글톤 인스턴스 반환

    Args:
        **kwargs: ManagerS 초기화 파라미터 (처음 생성 시에만 사용됨)

    Returns:
        ManagerS 인스턴스
    """
    global _manager_s_agent_instance
    if _manager_s_agent_instance is None:
        _manager_s_agent_instance = ManagerS(**kwargs)
    return _manager_s_agent_instance
