import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:window@localhost/deep-distribute")
#DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:szabist@192.168.10.120:5432/deep-distribute")

# Redis connection string
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


#DATASET_HOST = os.getenv("DATASET_HOST", "http://localhost:89/tmp/")

PARAMETER_SERVER_URL = os.getenv("PARAMETER_SERVER_URL", "/ps") #root context real url will be injected at runtime
SUBMIT_WEIGHTS_URL = f"{PARAMETER_SERVER_URL}/submit-weights"
GET_WEIGHTS_URL = f"{PARAMETER_SERVER_URL}/get-weights"

SUBMIT_GRADIENTS_URL = f"{PARAMETER_SERVER_URL}/submit-gradients"
GET_GRADIENTS_URL = f"{PARAMETER_SERVER_URL}/get-gradients"


SUBMIT_METRICS_URL = f"{PARAMETER_SERVER_URL}/submit-metrics"
SUBMIT_STATS_URL = f"{PARAMETER_SERVER_URL}/submit-stats"

#TRAINED_MODELS_DIR ="/media/ali/workspace/models"

