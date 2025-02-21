from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models.cluster import Cluster  # Import the Cluster model

class ClusterDAO:

    @staticmethod
    async def save(db: AsyncSession, cluster: Cluster):
        # Add the new cluster to the session
        db.add(cluster)
        await db.commit()
        await db.refresh(cluster)
        return cluster

    @staticmethod
    async def fetch_all_clusters(db: AsyncSession):
        # Correct query to select all rows from the train_cluster table
        query = select(Cluster)  # This is correct for SQLAlchemy ORM models
        result = await db.execute(query)
        return result.scalars().all()  # Fetch all rows and return them as a list

    @staticmethod
    async def fetch_cluster_by_id(db: AsyncSession, cluster_id: int):
        # Query to fetch a single cluster by ID
        query = select(Cluster).filter(Cluster.id == cluster_id)
        result = await db.execute(query)
        return result.scalars().one_or_none()  # Returns the first result or None if not found

    @staticmethod
    async def update_cluster_status(db: AsyncSession, cluster_id: int, status: str):
        # Query to update the cluster's status
        query = select(Cluster).filter(Cluster.id == cluster_id)
        result = await db.execute(query)
        cluster = result.scalars().one_or_none()
        if cluster:
            cluster.status = status  # Update the status
            await db.commit()  # Commit the changes to the database
            await db.refresh(cluster)
        return cluster
