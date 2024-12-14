import asyncio
from app.helpers.weights_helper import aggregate
class ParameterService:
    weights_store = {}  # In-memory storage for weights by job_id

    @staticmethod
    async def aggregate_weights(job_id: str, worker_weights: dict):
        """
        Aggregate weights from a worker and update in memory.
        """
        if job_id not in ParameterService.weights_store:
            ParameterService.weights_store[job_id] = worker_weights
        else:
            existing_weights = ParameterService.weights_store[job_id]
            existing_weights = aggregate(existing_weights, worker_weights)
            ParameterService.weights_store[job_id] = existing_weights
        return True

    @staticmethod
    async def get_aggregated_weights(job_id: str):
        """
        Return aggregated weights for the given job_id.
        """
        return ParameterService.weights_store.get(job_id)
