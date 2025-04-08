from confluent_kafka import Consumer, KafkaException, KafkaError
import threading
from app.config import settings

import json
import asyncio
from app.helpers.weights_helper import serialize_weights,deserialize_weights
from app.services.parameter_service import ParameterService
from app.helpers.util import  get_host

consumer_config = {}
consumer_config["bootstrap.servers"] = settings.KAFKA_BROKER_URL
consumer_config["group.id"] = get_host()
consumer_config["auto.offset.reset"] = "earliest"
consumer_config["enable.auto.commit"] = False
consumer_config["fetch.max.bytes"] = 104857600 # Match broker setting




WORKER_GRADIENTS_TOPIC = "worker_gradients"
PS_GRADIENTS_TOPIC = "ps_gradients"
WORKER_METRICS_TOPIC = "worker_metrics"
TEST1_TOPIC = "test1"

# Initialize the Kafka consumer
consumer = Consumer(consumer_config)
#consumer.subscribe([WORKER_GRADIENTS_TOPIC, WORKER_METRICS_TOPIC]) ## Uncomment this for Parameter server
#consumer.subscribe([PS_GRADIENTS_TOPIC]) ## Uncomment this for worker server
consumer.subscribe([PS_GRADIENTS_TOPIC, WORKER_GRADIENTS_TOPIC, WORKER_METRICS_TOPIC]) #Uncomment for both worker and ps on same machine
# This will run the consumer in a separate thread to keep the FastAPI server responsive
async def consume_messages():
    try:
        while True:

            # if not app.state.app_started:
            #    break

            msg = consumer.poll(timeout=1.0)  # Adjust the timeout as needed

            if msg is None:  # No message available
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition reached
                    continue
                else:
                    raise KafkaException(msg.error())

            topic = msg.topic()
            is_consumed = await consume_message(topic, msg)
            if is_consumed:
                consumer.commit(msg)
                print(f"KAFKA message consumed successfully; ${topic}  ")
            else:
                print(f"ERROR occured while consuming KAFKA message; ${topic};  ")

    except KeyboardInterrupt as ex:
        print(f"ERROR: {str(ex)}")

    except Exception as e:
        print(f"ERROR: {str(e)}")
    finally:
        consumer.close()


async def consume_message(topic, msg):
    func_name = f"consume_{topic.replace('-', '_')}"  # Replace dashes with underscores to make it a valid function name

    if func_name in globals():
        func = globals()[func_name]

        # Check if the function is a coroutine (async function)
        if asyncio.iscoroutinefunction(func):
            # If it's async, await the function
            return await func(msg)
        else:
            # If it's not async, just call the function
            return func(msg)
    else:
        print(f"Warning: No consumer function found for topic: {topic}")
        return False


async def consume_worker_metrics(msg):
    try:
        raw_data = msg.value()
        key = msg.key()
        if key:
            worker_id, job_id,partition = key.decode("utf-8").split(":")
        else:
            raise ValueError("Message for consume_worker_metrics key is missing.")

        if not raw_data:
            raise ValueError("Received empty message in consume_worker_metrics")
        metrics = json.loads(raw_data.decode('utf-8'))
        from app.services.parameter_service import ParameterService
        aggregated_metrics = await ParameterService.aggregate_metrics(worker_id, job_id, metrics)
        if aggregated_metrics:
            return True
    except Exception as e:
        print(f"ERROR consuming consume_worker_metrics message: {str(e)}")
    return False


async def consume_worker_gradients(msg):
    try:
        # Directly get raw bytes from Kafka message
        raw_data = msg.value()  # No JSON decoding
        if not raw_data:
            raise ValueError("Received empty message in consume_worker_gradients.")

        # Deserialize the received data
        worker_gradients = deserialize_weights(raw_data)

        # Extract metadata from message key (if needed)
        key = msg.key()
        if key:
            worker_id, job_id,partition = key.decode("utf-8").split(":")  # Assuming "worker_id:job_id" format
        else:
            raise ValueError("Message for consume_worker_gradients key is missing.")

        # Process gradients in ParameterService
        success = await ParameterService.aggregate_gradients(worker_id, job_id, worker_gradients)
        if success:
            #publish/produce aggrigated gradients for other workers
            await ParameterService.produce_latest_gradients(job_id)
        else:
            raise Exception("Failed to aggregate gradients.")

        return success

    except Exception as e:
        print(f"ERROR consuming consume_worker_gradients message: {str(e)}")
        return False



async def consume_ps_gradients(msg):
    try:
        # Directly get raw bytes from Kafka message
        raw_data = msg.value()
        if not raw_data:
            raise ValueError("Received empty message.")

        # Deserialize the received data
        ps_gradients = deserialize_weights(raw_data)
        job_id = msg.key()
        from app.background_tasks import worker_sync
        worker_sync.gradients_store[job_id] = {"latest":ps_gradients}

        return True

    except Exception as e:
        print(f"ERROR consuming message: {str(e)}")
        return False

is_thread_started = False


def start_consumer_thread():
    try:
        global is_thread_started

        if not is_thread_started:
            # Create and start a thread to run the async function
            threading.Thread(target=run_consumer_thread, daemon=True).start()
            is_thread_started = True

    except Exception as ex:
        print(
            f"An Error occurred while starting the consumer thread: {type(ex), ex}",
        )
        is_thread_started = False


def run_consumer_thread():
    """
    Runs the consume_messages function in an asyncio event loop.
    This is necessary because async functions can't be directly executed in threads.
    """
    loop = asyncio.new_event_loop()  # Create a new event loop for this thread
    asyncio.set_event_loop(loop)  # Set the loop for the current thread
    loop.run_until_complete(
        consume_messages()
    )  # Run the async function in the event loop

"""
@app.get("/kafka-consumer/thread/start")
async def kafka_consumer_status():
    start_consumer_thread()
    return {"message": f"Kafka Consumer status: {is_thread_started}"}
"""
