import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:window@localhost/deep-distribute")

# Redis connection string
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

PARAMETER_SERVER_URL = os.getenv("PARAMETER_SERVER_URL", "http://localhost:9000/ps")
SUBMIT_WEIGHTS_URL = f"{PARAMETER_SERVER_URL}/submit-weights"
GET_WEIGHTS_URL = f"{PARAMETER_SERVER_URL}/get-weights"

