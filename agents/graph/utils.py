from typing import Optional, Any
# Circular import 방지를 위해 내부에서 import 하거나, TYPE_CHECKING 사용
# 여기서는 TeamHGraph 클래스를 import 해야 하는데, graph.py가 아직 없으므로
# 나중에 graph.py에서 utils를 import하지 않도록 주의.
# 보통 Factory는 Graph 클래스를 알아야 함.

# 싱글톤 인스턴스 저장소
_team_h_graph_instance = None

def create_team_h_graph(**kwargs) -> Any:
    """
    Team-H Graph 생성 헬퍼 함수
    
    Args:
        **kwargs: TeamHGraph 생성자에 전달할 인자
        
    Returns:
        TeamHGraph 인스턴스
    """
    from agents.graph.graph import TeamHGraph
    return TeamHGraph(**kwargs)

def get_team_h_graph(**kwargs) -> Any:
    """
    Team-H Graph 싱글톤 인스턴스 가져오기 (없으면 생성)
    
    Args:
        **kwargs: 생성 시 전달할 인자 (이미 생성된 경우 무시됨)
        
    Returns:
        TeamHGraph 인스턴스
    """
    global _team_h_graph_instance
    if _team_h_graph_instance is None:
        _team_h_graph_instance = create_team_h_graph(**kwargs)
    return _team_h_graph_instance
