from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models.trained_model import TrainedModel

class TrainedModelDAO:
    @staticmethod
    async def fetch_all_trained_models(db: AsyncSession):
        # Correct query to select all rows from the train_training_job table
        query = select(TrainedModel)  # This is correct for SQLAlchemy ORM models
        result = await db.execute(query)
        return result.scalars().all()  # Fetch all rows and return them as a list @staticmethod

    @staticmethod
    async def fetch_by_dataset(db: AsyncSession, dataset_img_id: int):

        # Query to fetch a single job by ID
        query = select(TrainedModel).filter(TrainedModel.id == dataset_img_id)
        result = await db.execute(query)
        return result.scalars().all()

    @staticmethod
    async def fetch_trained_model_by_id(db: AsyncSession, model_id: int):
        # Query to fetch a single job by ID
        query = select(TrainedModel).filter(TrainedModel.id == model_id)
        result = await db.execute(query)
        return result.scalar_one_or_none()  # Returns the first result or None if not found

    @staticmethod
    async def save_trained_model(db: AsyncSession, trained_model: TrainedModel):
        # Add the new trained model to the session
        db.add(trained_model)
        await db.commit()  # Commit the transaction to save to the database
        await db.refresh(trained_model)  # Refresh to get the id if it's auto-generated (like `serial`)
        return trained_model  # Return the saved model with its ID

    @staticmethod
    async def update_trained_model(db: AsyncSession, model_id: int, updated_data: dict):
        # Fetch the existing trained model from the database
        query = select(TrainedModel).filter(TrainedModel.id == model_id)
        result = await db.execute(query)
        trained_model = result.scalar_one_or_none()

        if trained_model:
            # Update the fields based on the provided updated data
            for key, value in updated_data.items():
                if hasattr(trained_model, key):
                    setattr(trained_model, key, value)

            await db.commit()  # Commit the update to the database
            await db.refresh(trained_model)  # Refresh to ensure we have the latest data
            return trained_model
        else:
            return None  # If no model found with the given id