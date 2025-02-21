from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class TrainedModel(Base):
    __tablename__ = 'train_trainedmodel'

    # Primary Key
    id = Column(Integer, primary_key=True, index=True)

    # Other fields
    model_file = Column(String(100), nullable=True)
    description = Column(String(300), nullable=True)
    status = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)
    key_attributes = Column(Text, nullable=False)
    class_label = Column(Text, nullable=False)
    dataset_id = Column(Integer, nullable=True)
    user_id = Column(Integer, nullable=False)
    dataset_img_id = Column(Integer, nullable=True)

    # You can add methods or other attributes here if needed
