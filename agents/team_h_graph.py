"""
Team-H Graph - Backward Compatibility Wrapper

⚠️ DEPRECATED: This module is kept for backward compatibility.
Please use the new modular structure:
    from agents.graph import TeamHGraph, create_team_h_graph, get_team_h_graph

New structure:
    agents/graph/
    ├── __init__.py       # Public API
    ├── state.py          # TeamHState, AgentRouting
    ├── nodes.py          # Node execution logic
    ├── graph.py          # TeamHGraph class
    └── utils.py          # Factory functions
"""

# Import everything from the new modular structure
from agents.graph import (
    TeamHGraph,
    TeamHState,
    AgentRouting,
    create_team_h_graph,
    get_team_h_graph,
)

__all__ = [
    "TeamHGraph",
    "TeamHState",
    "AgentRouting",
    "create_team_h_graph",
    "get_team_h_graph",
]
