"""
매니저 H - PostgreSQL 데이터베이스 모델 정의
SQLAlchemy ORM을 사용한 Progress 테이블 정의
"""

from datetime import datetime
from sqlalchemy import Column, Integer, Text, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from database.postgres.models.base import Base


class Progress(Base):
    """진행 상황 테이블 - 목표의 진행 내역을 기록"""
    
    __tablename__ = "progress"
    
    progress_id = Column(Integer, primary_key=True, autoincrement=True, comment="진행 상황 기록 고유 번호")
    goal_id = Column(Integer, ForeignKey("goals.index"), nullable=False, comment="목표 FK")
    progress_content = Column(Text, nullable=False, comment="진행 상황 상세 내용")
    logged_at = Column(DateTime, nullable=False, default=datetime.utcnow, comment="진행 상황 기록 날짜")
    delete_yn = Column(Boolean, nullable=False, default=False, comment="삭제 여부")
    
    # Goal과의 관계 설정 (N:1)
    goal = relationship("Goal", back_populates="progresses")
    
    def __repr__(self):
        return f"<Progress(progress_id={self.progress_id}, goal_id={self.goal_id}, logged_at={self.logged_at})>"
