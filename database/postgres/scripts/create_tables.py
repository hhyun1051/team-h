from database.postgres.core.connection import engine
from database.postgres.models import Base, Goal, Progress

def create_db_tables():
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")

if __name__ == "__main__":
    create_db_tables()