import json
import aiohttp
import gzip
import pickle
import numpy as np
from app.config.settings import SUBMIT_WEIGHTS_URL, GET_WEIGHTS_URL, SUBMIT_GRADIENTS_URL, GET_GRADIENTS_URL, SUBMIT_METRICS_URL, SUBMIT_STATS_URL
from app.services.redis_service import redis_client
from app.helpers.weights_helper import serialize_weights, deserialize_weights
import tensorflow as tf







async def submit_weights_task(parameter_server_url,worker_id: str, job_id: str, weights: list):

    queue = redis_client.get_task_queue()
    queue.enqueue(submit_weights,parameter_server_url, worker_id, job_id, weights, result_ttl=0)

async def fetch_latest_weights_task(parameter_server_url, worker_id: str, job_id: str):

    queue = redis_client.get_task_queue()
    queue.enqueue(fetch_latest_weights,parameter_server_url, worker_id, job_id, result_ttl=0)

async def submit_weights(parameter_server_url, worker_id: str, job_id: str, weights: list):
    """
    Submit the updated weights to the parameter server with compression for efficiency.
    """
    #print("------submit_weights called ----------------")
    try:
        # Normalize to numpy arrays before serialization
        normalized_weights = [weight.numpy() if isinstance(weight, tf.Variable) else weight for weight in weights]
        # Serialize and compress weights
        compressed_weights = serialize_weights(normalized_weights)
        #print (compressed_weights)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{parameter_server_url}{SUBMIT_WEIGHTS_URL}/{worker_id}/{job_id}",
                data=compressed_weights,  # Send compressed binary data
                headers={"Content-Encoding": "gzip", "Content-Type": "application/octet-stream"}
            ) as response:
                if response.status == 200:
                    pass
                    #print("-----------5")
                    #print(f"Weights successfully submitted for job {job_id}")
                else:
                    #print("-----------6")
                    #print(f"Error submitting weights for job {job_id}: {response.status}")
                    response_text = await response.text()
                    #print(f"Response: {response_text}")
    except Exception as e:
        print(f"Exception occurred while submitting weights for job {job_id}: {e}")



async def fetch_latest_weights(parameter_server_url, worker_id: str, job_id: str):
    """
    Fetch the latest aggregated weights from the parameter server and update in Redis.
    """
    print("------fetch_latest_weights called ----------------")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{parameter_server_url}{GET_WEIGHTS_URL}/{worker_id}/{job_id}") as response:
                if response.status == 200:
                    content_encoding = response.headers.get("Content-Encoding", "")
                    if "gzip" in content_encoding:
                        # Read the binary (gzip-compressed) data from the response
                        latest_weights = await response.read()
                        latest_weights = deserialize_weights(latest_weights)

                        redis_client.update_weights(job_id, latest_weights)
                        #
                        # Convert to tf.Variable for consistency
                        #latest_weights = [tf.Variable(weight) for weight in latest_weights]
                        #print(f"Latest weights updated in context for job {job_id}")
                    else:
                        pass
                        #print(f"No weights received from parameter server for job {job_id}")


                else:
                    #print(f"Error fetching weights for job {job_id}: {response.status}")
                    response_text = await response.text()
                    #print(f"Response: {response_text}")
        except (gzip.BadGzipFile, pickle.PickleError) as e:
            print(f"Error decompressing or deserializing weights for job {job_id}: {e}")
        except Exception as e:
            print(f"Exception occurred while fetching weights for job {job_id}: {e}")














''' ================== GRADIENTS==================='''
gradients_store={}


async def submit_gradients(parameter_server_url, worker_id: str, job_id: str, gradients: list):
    """
    Submit the difference in gradients to the parameter server to reduce data transmission size.
    """
    print("------submit_gradients with difference called ----------------")
    
    try:
        # Normalize gradients to numpy arrays
        normalized_gradients = [grad.numpy() if isinstance(grad, tf.Tensor) else grad for grad in gradients]

        # Get the latest stored gradients

        previous_gradients = gradients_store.get(job_id, {}).get("latest",None)
        
        if previous_gradients is not None:
            # Compute gradient differences (delta)
            print("****  debug: previous_gradients is not None  ****")
            delta_gradients = [curr - prev for curr, prev in zip(normalized_gradients, previous_gradients)]
        else:
            print("****  debug: previous_gradients is None  ****")
            delta_gradients = normalized_gradients  # First-time submission sends full gradients

        # Apply sparsification (optional: remove small values to compress further)
        sparsity_threshold = 1e-5
        delta_gradients = [np.where(np.abs(grad) > sparsity_threshold, grad, 0) for grad in delta_gradients]

        # Serialize and compress the delta gradients
        serialized_gradients = serialize_weights(delta_gradients)

        from app._kafka.producer import produce_submit_gradients
        if produce_submit_gradients(worker_id, job_id, serialized_gradients):
            gradients_store[job_id]= {"latest":normalized_gradients, "consumed":False}
        else:
            raise Exception("Failed to produce event")

    except Exception as e:
        print(f"Exception occurred while submitting gradients for job {job_id}: {e}")


async def submit_gradients_http(parameter_server_url, worker_id: str, job_id: str, gradients: list):
    """
    Submit the difference in gradients to the parameter server to reduce data transmission size.
    """
    print("------submit_gradients with difference called ----------------")

    try:
        # Normalize gradients to numpy arrays
        normalized_gradients = [grad.numpy() if isinstance(grad, tf.Tensor) else grad for grad in gradients]

        # Get the latest stored gradients
        previous_gradients = gradients_store.get(job_id, {}).get("latest",None)

        if previous_gradients is not None:
            # Compute gradient differences (delta)
            print("****  debug: previous_gradients is not None  ****")
            delta_gradients = [curr - prev for curr, prev in zip(normalized_gradients, previous_gradients)]
        else:
            print("****  debug: previous_gradients is None  ****")
            delta_gradients = normalized_gradients  # First-time submission sends full gradients

        # Apply sparsification (optional: remove small values to compress further)
        sparsity_threshold = 1e-5
        delta_gradients = [np.where(np.abs(grad) > sparsity_threshold, grad, 0) for grad in delta_gradients]

        # Serialize and compress the delta gradients
        compressed_gradients = serialize_weights(delta_gradients)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{parameter_server_url}{SUBMIT_GRADIENTS_URL}/{worker_id}/{job_id}",
                    data=compressed_gradients,
                    headers={"Content-Encoding": "gzip", "Content-Type": "application/octet-stream"}
            ) as response:
                if response.status == 200:
                    print(f"Delta gradients successfully submitted for job {job_id}")
                    # Update stored gradients
                    gradients_store[job_id] = {"latest":normalized_gradients}
                else:
                    print(f"Error submitting gradients for job {job_id}: {response.status}")
                    response_text = await response.text()
                    print(f"Response: {response_text}")

    except Exception as e:
        print(f"Exception occurred while submitting gradients for job {job_id}: {e}")

def fetch_ps_gradients(job_id):
    if not gradients_store.get(job_id, {}).get("consumed",True):
        return  gradients_store.get(job_id, {}).get("latest",None)
    return None
def reset_ps_gradients(job_id):
    gradients_store[job_id] = {"consumed":True}

async def fetch_latest_gradients(parameter_server_url, worker_id: str, job_id: str):
    """
    Fetch the latest aggregated gradients from the parameter server and update in Redis.
    """
    print("------fetch_latest_gradients called ----------------")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{parameter_server_url}{GET_GRADIENTS_URL}/{worker_id}/{job_id}") as response:
                if response.status == 200:
                    content_encoding = response.headers.get("Content-Encoding", "")
                    if "gzip" in content_encoding:
                        # Read the binary (gzip-compressed) data from the response
                        latest_gradients = await response.read()
                        latest_gradients = deserialize_weights(latest_gradients)  # Deserialize gradients

                        #redis_client.update_gradients(job_id, latest_gradients)
                        print(f"Latest gradients updated in context for job {job_id}")
                        gradients_store[job_id] = {"latest":latest_gradients, "consumed":False}
                        return latest_gradients
                    else:
                        print(f"No gradients received from parameter server for job {job_id}")
                else:
                    print(f"Error fetching gradients for job {job_id}: {response.status}")
                    response_text = await response.text()
                    print(f"Response: {response_text}")
        except Exception as e:
            print(f"Exception occurred while fetching gradients for job {job_id}: {e}")
    return None





''' ================== Metrics==================='''
async def submit_metrics(parameter_server_url, worker_id: str, job_id: str, metrics: dict):
    """
    Submit the real-time metrics to the parameter server.
    """
    print("------submit_metrics called ----------------")
    try:
        # Convert any numpy.float32 values to native Python float
        metrics = {key: float(value) if isinstance(value, np.float32) else value for key, value in metrics.items()}
        metrics_bytes = json.dumps(metrics).encode('utf-8')
        from app._kafka.producer import produce_worker_metrics
        res = produce_worker_metrics(worker_id,job_id, metrics_bytes)
        if res == True:
            print (f"submit_metrics produced for job_id: {job_id}; worker_id: {worker_id}")
        else:
            print(f"Failed: submit_metrics  for job_id: {job_id}; worker_id: {worker_id}")

    except Exception as e:
        print(f"Exception occurred while submitting metrics for job {job_id}: {e}")

async def submit_metrics_http(parameter_server_url, worker_id: str, job_id: str, metrics: dict):
    """
    Submit the real-time metrics to the parameter server.
    """
    print("------submit_metrics called ----------------")
    try:
        # Convert any numpy.float32 values to native Python float
        metrics = {key: float(value) if isinstance(value, np.float32) else value for key, value in metrics.items()}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{parameter_server_url}{SUBMIT_METRICS_URL}/{worker_id}/{job_id}",
                    json=metrics,  # Use json to send as JSON
            ) as response:
                if response.status == 200:
                    print(f"Metrics successfully submitted for job {job_id}")
                else:
                    print(f"Error submitting metrics for job {job_id}: {response.status}")
                    response_text = await response.text()
                    print(f"Response: {response_text}")
    except Exception as e:
        print(f"Exception occurred while submitting metrics for job {job_id}: {e}")



''' ================== Metrics==================='''

async def submit_stats(parameter_server_url, worker_id: str, job_id: str, stats: dict):
    """
    Submit the real-time stats to the parameter server.
    """
    print("------submit_stats called ----------------")
    try:

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{parameter_server_url}{SUBMIT_STATS_URL}/{worker_id}/{job_id}",
                    json=stats,  # Use json to send as JSON
            ) as response:
                if response.status == 200:
                    print(f"stats successfully submitted for job {job_id}")
                else:
                    print(f"Error submitting stats for job {job_id}: {response.status}")
                    response_text = await response.text()
                    print(f"Response: {response_text}")
    except Exception as e:
        print(f"Exception occurred while submitting stats for job {job_id}: {e}")
