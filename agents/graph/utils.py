"""
Utility functions for Team-H Graph
"""

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
