from sqlalchemy.orm import Session
from database.postgres.models import Progress
from datetime import datetime, timezone
from typing import List, Optional

# Progress CRUD operations
def create_progress(db: Session, goal_id: int, progress_content: str) -> Progress:
    """새로운 진행 상황을 기록합니다."""
    db_progress = Progress(
        goal_id=goal_id,
        progress_content=progress_content,
        logged_at=datetime.now(timezone.utc),
        delete_yn=False
    )
    db.add(db_progress)
    db.commit()
    db.refresh(db_progress)
    return db_progress

def get_progress(db: Session, progress_id: int) -> Optional[Progress]:
    """ID로 진행 상황을 조회합니다."""
    return db.query(Progress).filter(Progress.progress_id == progress_id, Progress.delete_yn == False).first()

def get_progresses_by_goal(db: Session, goal_id: int, skip: int = 0, limit: int = 100) -> List[Progress]:
    """특정 목표의 모든 진행 상황을 조회합니다."""
    return db.query(Progress).filter(Progress.goal_id == goal_id, Progress.delete_yn == False).offset(skip).limit(limit).all()

def update_progress(db: Session, progress_id: int, progress_content: Optional[str] = None) -> Optional[Progress]:
    """진행 상황을 업데이트합니다."""
    db_progress = get_progress(db, progress_id)
    if db_progress:
        if progress_content is not None:
            db_progress.progress_content = progress_content
        db.commit()
        db.refresh(db_progress)
    return db_progress

def delete_progress(db: Session, progress_id: int) -> Optional[Progress]:
    """진행 상황을 삭제(delete_yn을 True로 설정)합니다."""
    db_progress = get_progress(db, progress_id)
    if db_progress:
        db_progress.delete_yn = True
        db.commit()
        db.refresh(db_progress)
    return db_progress
