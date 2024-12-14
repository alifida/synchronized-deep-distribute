import aiohttp
import gzip
import pickle

from app.config.settings import SUBMIT_WEIGHTS_URL, GET_WEIGHTS_URL
from app.services.redis_service import redis_client
from app.helpers.weights_helper import serialize_weights, deserialize_weights
import tensorflow as tf




import threading
import asyncio

# Define the function to run the async function in the event loop
def run_fetch_latest_weights(job_id):
    loop = asyncio.new_event_loop()  # Create a new event loop for the thread
    asyncio.set_event_loop(loop)  # Set the event loop for this thread
    loop.run_until_complete(fetch_latest_weights(job_id))  # Run the async function


def run_submit_weights(job_id, weights):
    loop = asyncio.new_event_loop()  # Create a new event loop for the thread
    asyncio.set_event_loop(loop)  # Set the event loop for this thread
    loop.run_until_complete(submit_weights(job_id, weights))  # Run the async function



async def submit_weights(job_id: str, weights: list):
    """
    Submit the updated weights to the parameter server with compression for efficiency.
    """
    print("------submit_weights called ----------------")
    try:
        # Normalize to numpy arrays before serialization
        normalized_weights = [weight.numpy() if isinstance(weight, tf.Variable) else weight for weight in weights]

        # Serialize and compress weights
        compressed_weights = serialize_weights(normalized_weights)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{SUBMIT_WEIGHTS_URL}/{job_id}",
                data=compressed_weights,  # Send compressed binary data
                headers={"Content-Encoding": "gzip", "Content-Type": "application/octet-stream"}
            ) as response:
                if response.status == 200:
                    print(f"Weights successfully submitted for job {job_id}")
                else:
                    print(f"Error submitting weights for job {job_id}: {response.status}")
                    response_text = await response.text()
                    print(f"Response: {response_text}")
    except Exception as e:
        print(f"Exception occurred while submitting weights for job {job_id}: {e}")



async def fetch_latest_weights(job_id: str):
    """
    Fetch the latest aggregated weights from the parameter server and update in Redis.
    """
    print("------fetch_latest_weights called ----------------")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{GET_WEIGHTS_URL}/{job_id}") as response:
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
                        print(f"Latest weights updated in context for job {job_id}")
                    else:
                        print(f"No weights received from parameter server for job {job_id}")


                else:
                    print(f"Error fetching weights for job {job_id}: {response.status}")
                    response_text = await response.text()
                    print(f"Response: {response_text}")
        except (gzip.BadGzipFile, pickle.PickleError) as e:
            print(f"Error decompressing or deserializing weights for job {job_id}: {e}")
        except Exception as e:
            print(f"Exception occurred while fetching weights for job {job_id}: {e}")





            '''
async def fetch_latest_weights____(job_id: str):
    """
    Fetch the latest aggregated weights from the parameter server and update in Redis.
    """
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{GET_WEIGHTS_URL}/{job_id}") as response:
                if response.status == 200:
                    response_json = await response.json()

                    # Decompress and deserialize weights
                    compressed_weights = response_json['weights']
                    if compressed_weights:
                        decompressed_weights = gzip.decompress(compressed_weights)
                        latest_weights = pickle.loads(decompressed_weights)

                        # Update weights in Redis
                        redis_client.update_weights(job_id, latest_weights)
                        print(f"Latest weights updated in context for job {job_id}")
                else:
                    print(f"Error fetching weights for job {job_id}: {response.status}")
                    response_text = await response.text()
                    print(f"Response: {response_text}")
        except (gzip.BadGzipFile, pickle.PickleError) as e:
            print(f"Error decompressing or deserializing weights for job {job_id}: {e}")
        except Exception as e:
            print(f"Exception occurred while fetching weights for job {job_id}: {e}")

            '''