
from sqlalchemy import Column, Integer, String, ForeignKey 
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
Base = declarative_base()
 
class ClusterNode(Base):
    __tablename__ = 'train_clusternode'

    # Primary Key
    id = Column(Integer, primary_key=True, index=True)

    # Other fields
    node_type = Column(String(10), nullable=False)  # Choices like 'worker', 'ps'
    ip_address = Column(String(15), nullable=False)
    port = Column(Integer, nullable=False)
    cluster_id = Column(Integer, nullable=False)

    # Foreign Key to Cluster (Many-to-One relationship)
    # cluster_id = Column(Integer, ForeignKey('train_cluster.id'), nullable=True)

    # Relationship to Cluster
    #cluster = relationship("Cluster", back_populates="nodes")

    def __repr__(self):
        return f"<ClusterNode(id={self.id}, node_type={self.node_type}, ip_address={self.ip_address}, port={self.port})>"
