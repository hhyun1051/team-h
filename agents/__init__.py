"""
Team-H Agents Package

모든 Manager 에이전트를 제공합니다.

사용 예:
    from agents import ManagerI, ManagerM, ManagerS
"""

from agents.base_manager import ManagerBase
from agents.manager_i import ManagerI
from agents.manager_m import ManagerM
from agents.manager_s import ManagerS
from agents.manager_t import ManagerT

# TeamHGraph는 선택적으로 import (의존성 문제 방지)
try:
    from agents.team_h_graph import TeamHGraph
    __all__ = [
        "ManagerBase",
        "ManagerI",
        "ManagerM",
        "ManagerS",
        "ManagerT",
        "TeamHGraph",
    ]
except ImportError:
    __all__ = [
        "ManagerBase",
        "ManagerI",
        "ManagerM",
        "ManagerS",
        "ManagerT",
    ]

__version__ = "0.1.0"
