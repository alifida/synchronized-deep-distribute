from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update
from app.models.dataset_Img import DatasetImg

class DatasetImgDAO:
    @staticmethod
    async def fetch_all_dataset_images(db: AsyncSession):
        query = select(DatasetImg)
        result = await db.execute(query)
        return result.scalars().all()

    @staticmethod
    async def fetch_dataset_image_by_id(db: AsyncSession, id: int):
        query = select(DatasetImg).filter(DatasetImg.id == id)
        result = await db.execute(query)
        return result.scalar_one_or_none()  # Returns the first result or None if not found

    @staticmethod
    async def save_dataset_image(db: AsyncSession, dataset_img: DatasetImg):
        db.add(dataset_img)  # Add the new dataset image to the session
        await db.commit()  # Commit the transaction to save to the database
        await db.refresh(dataset_img)  # Refresh to get the ID if it's auto-generated
        return dataset_img  # Return the saved dataset image

    @staticmethod
    async def update_dataset_image(db: AsyncSession, id: int, updated_data: dict):
        # Fetch the existing dataset image by ID
        query = select(DatasetImg).filter(DatasetImg.id == id)
        result = await db.execute(query)
        dataset_img = result.scalar_one_or_none()

        if dataset_img:
            # Update the fields based on the provided updated data
            for key, value in updated_data.items():
                if hasattr(dataset_img, key):
                    setattr(dataset_img, key, value)

            await db.commit()  # Commit the changes to the database
            await db.refresh(dataset_img)  # Refresh to ensure we have the latest data
            return dataset_img
        else:
            return None  # If no dataset image found with the given ID
