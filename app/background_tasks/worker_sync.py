import aiohttp
import gzip
import pickle
import numpy as np
from app.config.settings import SUBMIT_WEIGHTS_URL, GET_WEIGHTS_URL, SUBMIT_GRADIENTS_URL, GET_GRADIENTS_URL, SUBMIT_METRICS_URL, SUBMIT_STATS_URL
from app.services.redis_service import redis_client
from app.helpers.weights_helper import serialize_weights, deserialize_weights, sparsify_gradients
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
async def submit_gradients(parameter_server_url, worker_id:str, job_id: str, gradients: list):
    """
    Submit the computed gradients to the parameter server with compression for efficiency.
    """
    print("------submit_gradients called ----------------")
    try:


        #gradients = sparsify_gradients(gradients)
        # Normalize gradients to numpy arrays before serialization
        normalized_gradients = [grad.numpy() if isinstance(grad, tf.Tensor) else grad for grad in gradients]
        # Serialize and compress gradients
        compressed_gradients = serialize_weights(normalized_gradients)  # Using same serialization as weights
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{parameter_server_url}{SUBMIT_GRADIENTS_URL}/{worker_id}/{job_id}",
                data=compressed_gradients,  # Send compressed binary data
                headers={"Content-Encoding": "gzip", "Content-Type": "application/octet-stream"}
            ) as response:
                if response.status == 200:
                    print(f"Gradients successfully submitted for job {job_id}")
                else:
                    print(f"Error submitting gradients for job {job_id}: {response.status}")
                    response_text = await response.text()
                    print(f"Response: {response_text}")
    except Exception as e:
        print(f"Exception occurred while submitting gradients for job {job_id}: {e}")

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

                        redis_client.update_gradients(job_id, latest_gradients)
                        print(f"Latest gradients updated in context for job {job_id}")
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
