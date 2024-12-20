from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class DatasetImg(Base):
    __tablename__ = 'train_dataset_img'

    # Primary Key
    id = Column(Integer, primary_key=True, index=True)

    # Other fields
    data_name = Column(String(300), nullable=False)
    data_path = Column(String(300), nullable=False)
    metainfo = Column(Text, nullable=False)
    processed_at = Column(DateTime(timezone=True), nullable=False)
    delete_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(300), nullable=False)
    user_id = Column(Integer, nullable=False)
    extracted_path = Column(Text, nullable=False)
    data_path_test = Column(String(300), nullable=False)
    extracted_path_test = Column(Text, nullable=False)

    # You can add methods or other attributes here if needed
