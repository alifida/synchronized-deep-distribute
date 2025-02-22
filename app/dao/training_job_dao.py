from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models.training_job import TrainingJob

class TrainingJobDAO:

    @staticmethod
    async def save(db: AsyncSession, training_job: TrainingJob):
        # Add the new trained model to the session
        db.add(training_job)
        await db.commit()
        await db.refresh(training_job)
        return training_job

    @staticmethod
    async def fetch_all_training_jobs(db: AsyncSession):
        # Correct query to select all rows from the train_training_job table
        query = select(TrainingJob)  # This is correct for SQLAlchemy ORM models
        result = await db.execute(query)
        return result.scalars().all()  # Fetch all rows and return them as a list

    @staticmethod
    async def fetch_training_job_by_id(db: AsyncSession, job_id: int):
        # Query to fetch a single job by ID
        query = select(TrainingJob).filter(TrainingJob.id == job_id)
        result = await db.execute(query)
        return result.scalars().one_or_none()  # Returns the first result or None if not found

    @staticmethod
    async def update_training_job_status(db: AsyncSession, job_id: int, status: str):
        # Query to update the job's status
        query = select(TrainingJob).filter(TrainingJob.id == job_id)
        result = await db.execute(query)
        job = result.scalars().one_or_none()
        if job:
            job.status = status  # Update the status
            await db.commit()  # Commit the changes to the database
