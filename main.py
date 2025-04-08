from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncpg
#from models import TrainDatasetImg, TrainTrainingJob
from app.controllers.parameter_server_controller import ps_router
from app.controllers.worker_controller import w_router
from app.controllers.training_job_controller import training_job_router
from app.controllers.trained_model_controller import trained_model_router
import os
from app.services.redis_service import redis_client
from pathlib import Path
from app._kafka.producer import init_producer
from app._kafka.consumer import start_consumer_thread
import logging
#Disable gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = FastAPI()
#app.mount("/static", StaticFiles(directory="app/static"), name="static")
#app.mount("/datasets", StaticFiles(directory="/media/ali/workspace/dataset"), name="datasets") # ubuntu
#app.mount("/datasets", StaticFiles(directory="E:/workspace/python/deep-distribute/deepdistribute/media"), name="datasets") #windows
#app.mount("/models", StaticFiles(directory="/media/ali/workspace/models"), name="datasets")


DATASET_DIR = Path(__file__).parent / "static/tmp/datasets"
MODELS_DIR = Path(__file__).parent / "static/trained_models"



logging.basicConfig(level=logging.WARNING)  # Set default logging level to WARNING
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)  # Suppress SQL query logs
logging.getLogger('asyncpg').setLevel(logging.WARNING)  # Optional, suppress asyncpg logs as well


task_queue = redis_client.get_task_queue()



init_producer()
start_consumer_thread()
# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database pool
@app.on_event("startup")
async def startup():
    app.state.db = await asyncpg.create_pool(
        user="postgres",
        password="window",
        database="deep-distribute",
        host="localhost",
        port=5432
    )

@app.on_event("shutdown")
async def shutdown():
    await app.state.db.close()




# Include Routers
app.include_router(ps_router, prefix="/ps", tags=["Parameter Server"])
app.include_router(w_router, prefix="/worker", tags=["Worker"])
app.include_router(training_job_router, prefix="/training-jobs", tags=["Training Jobs"])
app.include_router(trained_model_router, prefix="/trained-models", tags=["Trained Models"])



# Health Check Endpoint
@app.get("/")
async def health_check():
    return {"message": "API is up and running"}
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)
