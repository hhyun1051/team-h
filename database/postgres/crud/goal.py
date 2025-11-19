from sqlalchemy.orm import Session
from database.postgres.models import Goal
from datetime import datetime, timezone
from typing import List, Optional

# Goal CRUD operations
def create_goal(db: Session, title: str, content: str, due_date: Optional[datetime] = None, measurement_method: Optional[str] = None) -> Goal:
    """새로운 목표를 생성합니다."""
    db_goal = Goal(
        title=title,
        content=content,
        due_date=due_date,
        created_at=datetime.now(timezone.utc),
        measurement_method=measurement_method,
        delete_yn=False
    )
    db.add(db_goal)
    db.commit()
    db.refresh(db_goal)
    return db_goal

def get_goal(db: Session, goal_index: int) -> Optional[Goal]:
    """ID로 목표를 조회합니다."""
    return db.query(Goal).filter(Goal.index == goal_index, Goal.delete_yn == False).first()

def get_goals(db: Session, skip: int = 0, limit: int = 100) -> List[Goal]:
    """모든 목표를 조회합니다."""
    return db.query(Goal).filter(Goal.delete_yn == False).offset(skip).limit(limit).all()

def update_goal(db: Session, goal_index: int, title: Optional[str] = None, content: Optional[str] = None, due_date: Optional[datetime] = None, measurement_method: Optional[str] = None) -> Optional[Goal]:
    """목표를 업데이트합니다."""
    db_goal = get_goal(db, goal_index)
    if db_goal:
        if title is not None:
            db_goal.title = title
        if content is not None:
            db_goal.content = content
        if due_date is not None:
            db_goal.due_date = due_date
        if measurement_method is not None:
            db_goal.measurement_method = measurement_method
        db.commit()
        db.refresh(db_goal)
    return db_goal

def delete_goal(db: Session, goal_index: int) -> Optional[Goal]:
    """목표를 삭제(delete_yn을 True로 설정)합니다."""
    db_goal = get_goal(db, goal_index)
    if db_goal:
        db_goal.delete_yn = True
        db.commit()
        db.refresh(db_goal)
    return db_goal
