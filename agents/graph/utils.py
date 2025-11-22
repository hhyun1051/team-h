"""
Utility functions for Team-H Graph
"""

from typing import Optional
from .graph import TeamHGraph


def create_team_h_graph(**kwargs) -> TeamHGraph:
    """
    Team-H Graph 생성 헬퍼 함수

    Args:
        **kwargs: TeamHGraph 초기화 파라미터

    Returns:
        TeamHGraph 인스턴스
    """
    return TeamHGraph(**kwargs)


# 싱글톤 인스턴스
_team_h_graph_instance: Optional[TeamHGraph] = None


def get_team_h_graph(**kwargs) -> TeamHGraph:
    """
    Team-H Graph 싱글톤 인스턴스

    Args:
        **kwargs: TeamHGraph 초기화 파라미터 (첫 호출 시에만 사용)

    Returns:
        TeamHGraph 싱글톤 인스턴스
    """
    global _team_h_graph_instance
    if _team_h_graph_instance is None:
        _team_h_graph_instance = TeamHGraph(**kwargs)
    return _team_h_graph_instance
