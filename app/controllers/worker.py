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

@w_router.post("/test")
async def test():

    from app.helpers.augmentation import augment_tb_images, generate_image_json
    #augment_tb_images("/media/ali/workspace/dataset/tb_dataset/tb_dataset/Tuberculosis","/media/ali/workspace/dataset/tb_dataset/tb_dataset/Tuberculosis/generated",91)
    directory_path = '/media/ali/workspace/dataset/tb_dataset/tb_dataset'
    url_prefix = 'http://localhost:9000/datasets/tb_dataset/tb_dataset'
    return generate_image_json(directory_path, url_prefix)
