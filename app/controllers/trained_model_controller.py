from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.dao.trained_model_dao import TrainedModelDAO
from app.db.database import get_db  # Import the database session dependency

trained_model_router = APIRouter()
 

@trained_model_router.get("/")
async def get_all(db: AsyncSession = Depends(get_db)):
    models = await TrainedModelDAO.fetch_all_trained_models(db)
    return models

@trained_model_router.get("/{dataset_id}")
async def get_training_dataset_by_id(dataset_id: int, db: AsyncSession = Depends(get_db)):
    models = await TrainedModelDAO.fetch_by_dataset(db, dataset_id)
    if not models:
        raise HTTPException(status_code=404, detail="Training job not found")
    return models
