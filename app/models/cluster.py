from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Cluster(Base):
    __tablename__ = 'train_cluster'

    # Primary Key
    id = Column(Integer, primary_key=True, index=True)

    # Other fields
    name = Column(String(300), nullable=False)
    status = Column(String(300), nullable=False)

    # Relationship to ClusterNode (one-to-many)
    #nodes = relationship("ClusterNode", back_populates="cluster")

    def __repr__(self):
        return f"<Cluster(id={self.id}, name={self.name}, status={self.status})>"

 