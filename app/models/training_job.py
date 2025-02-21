from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from typing import Optional

Base = declarative_base()

class TrainingJob(Base):
    __tablename__ = 'train_training_job'
    # Primary Key: ID
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    # Job Name
    job_name: Mapped[str] = mapped_column(String(300), nullable=False)

    # Status
    status: Mapped[str] = mapped_column(String(300), nullable=False)

    # Timestamp columns (nullable)
    started_at: Mapped[Optional[DateTime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    ended_at: Mapped[Optional[DateTime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)

    # Algorithm used for training
    algo: Mapped[str] = mapped_column(String(300), nullable=False)

    # Foreign Key: Dataset Image ID
    dataset_img_id: Mapped[int] = mapped_column(Integer,  nullable=False)

    # Foreign Key: User ID
    user_id: Mapped[int] = mapped_column(Integer,  nullable=False)

    # Additional fields (nullable)
    result: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    parameter_settings: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    training_log: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    training_log_history: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    
    
