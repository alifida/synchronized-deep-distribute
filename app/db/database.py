from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from app.config.settings import DATABASE_URL


# Create Async Engine
engine = create_async_engine(DATABASE_URL, poolclass=NullPool, echo=True)

# Create Async Session
AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
