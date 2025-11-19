import pytest
from sqlalchemy.orm import Session

# 실제 PostgreSQL 연결을 위한 engine과 SessionLocal을 가져옵니다.
from database.postgres.core.connection import engine, SessionLocal
from database.postgres.models import Base
from database.postgres.crud import goal as goal_crud, progress as progress_crud

@pytest.fixture(name="db", scope="function")
def session_fixture():
    """테스트 함수마다 독립적인 데이터베이스 세션을 제공하고,
    테스트 전후로 테이블을 생성하고 삭제하여 깨끗한 환경을 보장합니다.
    """
    # 테스트 시작 전: 모든 테이블 생성
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        # 테스트 종료 후: 모든 테이블 삭제
        Base.metadata.drop_all(bind=engine)


def test_create_goal(db: Session):
    title = "테스트 목표"
    content = "테스트 목표 내용"
    goal = goal_crud.create_goal(db, title=title, content=content)

    assert goal.index is not None
    assert goal.title == title
    assert goal.content == content
    assert goal.delete_yn is False

def test_get_goal(db: Session):
    title = "조회 테스트 목표"
    content = "조회 테스트 내용"
    created_goal = goal_crud.create_goal(db, title=title, content=content)

    fetched_goal = goal_crud.get_goal(db, created_goal.index)

    assert fetched_goal is not None
    assert fetched_goal.index == created_goal.index
    assert fetched_goal.title == title

def test_get_goals(db: Session):
    goal_crud.create_goal(db, title="목표1", content="내용1")
    goal_crud.create_goal(db, title="목표2", content="내용2")

    goals = goal_crud.get_goals(db)
    assert len(goals) == 2
    assert goals[0].title == "목표1"

def test_update_goal(db: Session):
    title = "업데이트 전"
    content = "업데이트 전 내용"
    goal = goal_crud.create_goal(db, title=title, content=content)

    updated_title = "업데이트 후"
    updated_goal = goal_crud.update_goal(db, goal.index, title=updated_title)

    assert updated_goal is not None
    assert updated_goal.title == updated_title
    assert updated_goal.content == content

def test_delete_goal(db: Session):
    title = "삭제 테스트"
    content = "삭제 테스트 내용"
    goal = goal_crud.create_goal(db, title=title, content=content)

    deleted_goal = goal_crud.delete_goal(db, goal.index)

    assert deleted_goal is not None
    assert deleted_goal.delete_yn is True

    fetched_goal = goal_crud.get_goal(db, goal.index)
    assert fetched_goal is None  # 삭제된 목표는 조회되지 않아야 함

def test_create_progress(db: Session):
    goal = goal_crud.create_goal(db, title="진행 상황 목표", content="내용")
    progress_content = "첫 진행 상황"
    progress = progress_crud.create_progress(db, goal_id=goal.index, progress_content=progress_content)

    assert progress.progress_id is not None
    assert progress.goal_id == goal.index
    assert progress.progress_content == progress_content
    assert progress.delete_yn is False

def test_get_progress(db: Session):
    goal = goal_crud.create_goal(db, title="진행 상황 목표", content="내용")
    created_progress = progress_crud.create_progress(db, goal_id=goal.index, progress_content="내용")

    fetched_progress = progress_crud.get_progress(db, created_progress.progress_id)

    assert fetched_progress is not None
    assert fetched_progress.progress_id == created_progress.progress_id


def test_get_progresses_by_goal(db: Session):
    goal1 = goal_crud.create_goal(db, title="목표1", content="내용1")
    goal2 = goal_crud.create_goal(db, title="목표2", content="내용2")

    progress_crud.create_progress(db, goal_id=goal1.index, progress_content="목표1 진행1")
    progress_crud.create_progress(db, goal_id=goal1.index, progress_content="목표1 진행2")
    progress_crud.create_progress(db, goal_id=goal2.index, progress_content="목표2 진행1")

    progresses_goal1 = progress_crud.get_progresses_by_goal(db, goal1.index)
    assert len(progresses_goal1) == 2
    assert progresses_goal1[0].progress_content == "목표1 진행1"

def test_update_progress(db: Session):
    goal = goal_crud.create_goal(db, title="진행 상황 목표", content="내용")
    progress = progress_crud.create_progress(db, goal_id=goal.index, progress_content="업데이트 전")

    updated_content = "업데이트 후"
    updated_progress = progress_crud.update_progress(db, progress.progress_id, progress_content=updated_content)

    assert updated_progress is not None
    assert updated_progress.progress_content == updated_content

def test_delete_progress(db: Session):
    goal = goal_crud.create_goal(db, title="진행 상황 목표", content="내용")
    progress = progress_crud.create_progress(db, goal_id=goal.index, progress_content="삭제될 진행 상황")

    deleted_progress = progress_crud.delete_progress(db, progress.progress_id)

    assert deleted_progress is not None
    assert deleted_progress.delete_yn is True

    fetched_progress = progress_crud.get_progress(db, progress.progress_id)
    assert fetched_progress is None  # 삭제된 진행 상황은 조회되지 않아야 함
