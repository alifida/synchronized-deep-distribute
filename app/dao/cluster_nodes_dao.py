from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models.cluster_node import ClusterNode  
from sqlalchemy import and_


class ClusterNodesDAO:
 
    @staticmethod
    async def fetch_workers_by_cluster_id(db: AsyncSession, cluster_id: int):
        # Query to fetch a single cluster by ID
        query = select(ClusterNode).filter(and_(ClusterNode.cluster_id == cluster_id, ClusterNode.node_type == 'worker'))

        result = await db.execute(query)
        return result.scalars().all()  # Returns the first result or None if not found

     