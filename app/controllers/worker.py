from fastapi import APIRouter, BackgroundTasks, HTTPException
from app.services.worker_service import WorkerService

w_router = APIRouter()

@w_router.post("/init-training")
async def init_training(job_id: str, init_params: dict):
    """
    Initialize the worker with training parameters and the first image.
    """
    success = await WorkerService.initialize_training(job_id, init_params)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to initialize training.")
    return {"status": "success", "message": "Worker initialized and training started."}
