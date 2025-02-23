from fastapi import APIRouter, HTTPException, Depends, Request, Response

import numpy as np
from app.services.parameter_service import ParameterService
import gzip
import pickle
from app.helpers.weights_helper import serialize_weights,deserialize_weights

ps_router = APIRouter()




@ps_router.post("/submit-gradients/{worker_id}/{job_id}")
async def submit_gradients(request: Request, worker_id: str, job_id: str):
    """
        Endpoint to receive gradients from a worker and aggregate them.
        """
    #print(f"-----submit_gradients___by worker --{worker_id}--{job_id}")
    try:
        # Read binary gzipped payload
        compressed_data = await request.body()
        worker_gradients = deserialize_weights(compressed_data)

        # Call the aggregation logic
        success = await ParameterService.aggregate_gradients(worker_id, job_id, worker_gradients)

        if not success:
            raise HTTPException(status_code=400, detail="Failed to aggregate gradients.")
        return {"status": "success", "message": "Gradients aggregated successfully."}

    except (gzip.BadGzipFile, pickle.PickleError) as e:
        raise HTTPException(status_code=400, detail=f"Failed to process gradients: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@ps_router.get("/get-gradients/{worker_id}/{job_id}")
async def get_gradients(worker_id: str, job_id: str):
    """
      Endpoint to return aggregated gradients for a job.
      """
    try:
        # Fetch gradients from the parameter service
        gradients = await ParameterService.get_aggregated_gradients(worker_id, job_id)

        if gradients is None:
            return {"status": "success", "gradients": None}
        gradients = serialize_weights(gradients)


        # Return the compressed data with the appropriate headers
        return Response(
            content=gradients,
            media_type="application/octet-stream",  # This indicates binary content
            headers={"Content-Encoding": "gzip"},  # Indicating the content is gzipped
            status_code=200
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")



@ps_router.post("/submit-weights/{worker_id}/{job_id}")
async def submit_weights(request: Request, worker_id: str,  job_id: str):
    """
    Endpoint to receive weights from a worker and aggregate them.
    """
    #print(f"-----sumbmit_weight___by worker --{worker_id}--{job_id}")
    try:
        # Read binary gzipped payload
        compressed_data = await request.body()
        worker_weights = deserialize_weights(compressed_data)
        #worker_weights = payload.get("worker_weights")
        # Call the aggregation logic
        success = await ParameterService.aggregate_weights(worker_id, job_id, worker_weights)

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


@ps_router.post("/submit-metrics/{worker_id}/{job_id}")
async def submit_metrics(request: Request, worker_id: str, job_id: str, metrics: dict):
    """
    Endpoint to receive metrics from a worker and aggregate them.
    """
    #print(f"----- submit_metrics by worker -- {worker_id} -- job {job_id}")
    try:
        # Parse the incoming JSON data dynamically as a dictionary
        #metrics = await request.json()

        # Validate that the required keys are present
        #required_keys = ["accuracy", "precision", "recall", "f1", "auc"]
        #if not all(key in metrics for key in required_keys):
        #    raise HTTPException(status_code=400, detail="Missing required metric fields.")

        # Call the aggregate_metrics method to store and aggregate the metrics
        aggregated_metrics = await ParameterService.aggregate_metrics(worker_id, job_id, metrics)

        if aggregated_metrics:
            return {"message": "Metrics submitted and aggregated successfully",
                    "aggregated_metrics": aggregated_metrics}
        else:
            raise HTTPException(status_code=500, detail="Failed to aggregate metrics.")

    except Exception as e:
        print(f"Error while submitting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@ps_router.get("/get-metrics/{job_id}")
async def get_metrics(job_id: str):
    """
    Endpoint to return aggregated metrics for a job.
    """
    try:
        # Retrieve the overall aggregated metrics for the given job_id
        aggregated_metrics = await ParameterService.get_overall_metrics(job_id)

        if aggregated_metrics:
            return {"message": "Aggregated metrics retrieved successfully", "aggregated_metrics": aggregated_metrics}
        else:
            raise HTTPException(status_code=404, detail=f"No aggregated metrics found for job {job_id}.")

    except Exception as e:
        print(f"Error while retrieving metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@ps_router.post("/submit-stats/{worker_id}/{job_id}")
async def submit_stats(request: Request, worker_id: str, job_id: str, stats: dict):
    """
    Endpoint to receive stats from a worker and aggregate them.
    """
    #print(f"----- submit_stats by worker -- {worker_id} -- job {job_id}")
    try:

        status = await ParameterService.update_stats(worker_id, job_id, stats)
        return {"status": 200,"message": status}

    except Exception as e:
        print(f"Error while submitting stats: {str(e)}")
        #raise HTTPException(.status_code=500, detail=f"An unexpected error occurred: {e}")
        return {"status": 500, "message": f"{e}"}


@ps_router.post("/get-stats/{job_id}")
async def get_stats(request: Request, job_id: str, stats: dict):
    """
    Endpoint returns stats of a job
    """
    #print(f"----- get_stats for -- job {job_id}")
    try:

        stats = await ParameterService.get_stats( job_id)
        return stats

    except Exception as e:
        print(f"Error while getting stats: {str(e)}")
        #raise HTTPException(.status_code=500, detail=f"An unexpected error occurred: {e}")
        return {"status": 500, "message": f"{e}"}

@ps_router.get("/get-metrics/{job_id}")
async def get_metrics(job_id: str):
    """
    Endpoint to return aggregated metrics for a job.
    """
    try:
        # Retrieve the overall aggregated metrics for the given job_id
        aggregated_metrics = await ParameterService.get_overall_metrics(job_id)

        if aggregated_metrics:
            return {"message": "Aggregated metrics retrieved successfully",
                    "aggregated_metrics": aggregated_metrics}
        else:
            raise HTTPException(status_code=404, detail=f"No aggregated metrics found for job {job_id}.")

    except Exception as e:
        print(f"Error while retrieving metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")