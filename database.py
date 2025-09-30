from sqlalchemy import create_engine, Column, Integer, String, Date
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime

# Define the database file
DATABASE_URL = "sqlite:///tasks.db"

# Create the database engine
engine = create_engine(DATABASE_URL)

# Session maker to interact with the DB
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for our models
Base = declarative_base()

# Define the Task model
class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True, index=True)
    description = Column(String, index=True)
    assignee = Column(String)
    due_date_str = Column(String) # Storing due date as string for simplicity
    status = Column(String, default="To Do")

# Function to create the database and table
def create_db_and_tables():
    Base.metadata.create_all(bind=engine)

# To run this file directly and create the DB
if __name__ == "__main__":
    print("Creating database and tasks table...")
    create_db_and_tables()
    print("Database created successfully.")