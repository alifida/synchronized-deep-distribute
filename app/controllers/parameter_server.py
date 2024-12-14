from fastapi import APIRouter, HTTPException, Depends, Request, Response

import numpy as np
from app.services.parameter_service import ParameterService
import gzip
import pickle
from app.helpers.weights_helper import serialize_weights,deserialize_weights

ps_router = APIRouter()

@ps_router.post("/submit-weights/{job_id}")
async def submit_weights(request: Request, job_id: str):
    """
    Endpoint to receive weights from a worker and aggregate them.
    """
    try:
        # Read binary gzipped payload
        compressed_data = await request.body()
        worker_weights = deserialize_weights(compressed_data)
        #worker_weights = payload.get("worker_weights")
        # Call the aggregation logic
        success = await ParameterService.aggregate_weights(job_id, worker_weights)

        if not success:
            raise HTTPException(status_code=400, detail="Failed to aggregate weights.")
        return {"status": "success", "message": "Weights aggregated successfully."}

    except (gzip.BadGzipFile, pickle.PickleError) as e:
        raise HTTPException(status_code=400, detail=f"Failed to process weights: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")



@ps_router.get("/get-weights/{job_id}")
async def get_weights(job_id: str):
    """
    Endpoint to return aggregated weights for a job.
    """
    try:
        # Fetch weights from the parameter service
        weights = await ParameterService.get_aggregated_weights(job_id)

        if weights is None:
            return {"status": "success", "weights": None}
        weights = serialize_weights(weights)



        #test---------------------starts
        #weights = deserialize_weights(weights)
        #test------------------------ ends

        # Return the compressed data with the appropriate headers
        return Response(
            content=weights,
            media_type="application/octet-stream",  # This indicates binary content
            headers={"Content-Encoding": "gzip"},  # Indicating the content is gzipped
            status_code=200
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@ps_router.get("/get-weights____/{job_id}")
async def get_weights_____(job_id: str):
    """
    Endpoint to return aggregated weights for a job.
    """
    try:
        # Fetch weights from the parameter service
        weights = await ParameterService.get_aggregated_weights(job_id)
        if weights is None:
            return {"status": "success", "weights": None}

        # Serialize and compress weights before sending the response
        weights_serialized = pickle.dumps(weights)  # Serialize weights
        weights_compressed = gzip.compress(weights_serialized)  # Compress weights

        return {"status": "success", "weights": weights_compressed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")