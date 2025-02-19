from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.dao.training_job_dao import TrainingJobDAO
from app.db.database import get_db  # Import the database session dependency

from app.services.parameter_service import ParameterService


training_job_router = APIRouter()
 



 

@training_job_router.post("/start/{job_id}")
async def init_training(job_id:int, data: dict):
    """
    
    """
    print (data)

    if not True:
        raise HTTPException(status_code=400, detail="Failed to initialize training.")
    
    await ParameterService.start_training_job(data)
    
    return {"status": "success", "message": "Worker initialized and training started."}



@training_job_router.get("/")
async def get_all_training_jobs(db: AsyncSession = Depends(get_db)):
    jobs = await TrainingJobDAO.fetch_all_training_jobs(db)
    return jobs

@training_job_router.get("/{job_id}")
async def get_training_job_by_id(job_id: int, db: AsyncSession = Depends(get_db)):
    job = await TrainingJobDAO.fetch_training_job_by_id(db, job_id)
    if not job: 
        raise HTTPException(status_code=404, detail="Training job not found")
    return job

@training_job_router.patch("/{job_id}/status")
async def update_training_job_status(job_id: int, status: str, db: AsyncSession = Depends(get_db)):
    await TrainingJobDAO.update_training_job_status(db, job_id, status)
    return {"message": "Training job status updated"}

