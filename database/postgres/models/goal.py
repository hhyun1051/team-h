"""
매니저 H - PostgreSQL 데이터베이스 모델 정의
SQLAlchemy ORM을 사용한 Goals 테이블 정의
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean
from sqlalchemy.orm import relationship
from database.postgres.models.base import Base


class Goal(Base):
    """목표 테이블 - 사용자의 SMART 목표를 저장"""
    
    __tablename__ = "goals"
    
    index = Column(Integer, primary_key=True, autoincrement=True, comment="목표 고유 식별 번호")
    title = Column(String(255), nullable=False, comment="목표 제목")
    content = Column(Text, nullable=False, comment="목표 상세 내용")
    due_date = Column(DateTime, nullable=True, comment="목표 마감 기한")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, comment="목표 생성 날짜")
    measurement_method = Column(Text, nullable=True, comment="목표 측정 방법")
    delete_yn = Column(Boolean, nullable=False, default=False, comment="삭제 여부")
    
    # Progress와의 관계 설정 (1:N)
    progresses = relationship("Progress", back_populates="goal", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Goal(index={self.index}, title='{self.title}', due_date={self.due_date})>"
