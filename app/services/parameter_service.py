import asyncio
from app.helpers.weights_helper import aggregate, aggregate_gradients
class ParameterService:
    weights_store = {}  # In-memory storage for weights by job_id
    gradients_store = {}  # In-memory storage for gradients by job_id
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



    @staticmethod
    async def aggregate_gradients(job_id: str, worker_gradients: dict):
        """
        Aggregate gradients from a worker and update in memory.
        """
        if job_id not in ParameterService.gradients_store:
            ParameterService.gradients_store[job_id] = worker_gradients
        else:
            existing_gradients = ParameterService.gradients_store[job_id]
            existing_gradients = aggregate_gradients(existing_gradients, worker_gradients)
            ParameterService.gradients_store[job_id] = existing_gradients
        return True

    @staticmethod
    async def get_aggregated_gradients(job_id: str):
        """
        Return aggregated gradients for the given job_id.
        """
        return ParameterService.gradients_store.get(job_id)



