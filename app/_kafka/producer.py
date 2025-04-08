from confluent_kafka import Producer
from app.config import settings



producer_config = {
    "bootstrap.servers": settings.KAFKA_BROKER_URL,
    "compression.type": "snappy",  # Options: snappy, lz4, gzip, zstd
    "linger.ms": 10,  # Small delay to batch messages
    "batch.size": 1048576,  # 1MB batch size for efficiency
    "message.max.bytes": 104857600 # 100MB limit to match broker
}




# Global producer variable
producer = None
WORKER_GRADIENTS_TOPIC = "worker_gradients"
WORKER_METRICS_TOPIC = "worker_metrics"
PS_GRADIENTS_TOPIC = "ps_gradients"

TOTAL_PARTITIONS_PER_TOPIC =3
ACTIVE_PARTITIONS={}

def get_next_partition(topic):
    current = ACTIVE_PARTITIONS.get(topic,{}).get("current",-1)
    current+=1
    if current >=TOTAL_PARTITIONS_PER_TOPIC:
        current = 0
    ACTIVE_PARTITIONS[topic] = {"current":current}
    return str(current)

def init_producer():
    """Initialize the  Kafka producer."""
    global producer
    if producer is None:
        producer = Producer(producer_config)  # Initialize the producer only once
        print("Kafka Producer is initialized.")


def __produce_message(partition: str, topic: str, message: str):
    try:
        # Send  the message to the Kafka topic with acks=all to ensure it waits for acknowledgment
        if not producer:
            init_producer()
        partition_post_fix = get_next_partition(topic)
        producer.produce(
            topic,
            value=message,
            key=f"{partition}:{partition_post_fix}",
            # acks="all",
        )
        producer.flush()

        print(f"Kafka Produced message for topic: {topic}; ")

        # Return the response with metadata
        return {
            "status": "success",
            "message": "Message sent successfully",
            "meta": {"topic": topic, "partition": partition},
        }

    except Exception as e:
        print(
            f"Kafka Producer error for TOPIC: {topic};  ERROR: {str(e)} "
        )
        return {"status": "error", "message": str(e)}


def produce_message_test(partition: str, topic: str, message: str):
    try:
        # Send the message to the Kafka topic
        producer.produce(topic, value=message.encode("utf-8"), key=partition)
        producer.flush()
        print(f"Kafka Produced message for topic: {topic}; message: {message}")
        return {
            "status": "success",
            "message": f"Message sent for topic: {topic}; message: {message}",
        }

    except Exception as e:
        print(
            f"Kafka Producer error for TOPIC: {topic}; MESSAGE: {message}; ERROR: {str(e)} "
        )
        return {"status": "error", "message": str(e)}

def produce_worker_metrics(worker_id, job_id, data):
    res = __produce_message(
        partition=f"{worker_id}:{job_id}",
        topic=f"{WORKER_METRICS_TOPIC}",
        message=data,
    )
    if res["status"] == "success":
        return True
    else:
        return False

def produce_submit_gradients(worker_id, job_id, data):
    #from app.helpers.weights_helper import write_to_tmp_file
    #write_to_tmp_file(data)
    res = __produce_message(
        partition=f"{worker_id}:{job_id}",
        topic=f"{WORKER_GRADIENTS_TOPIC}",
        message=data,
    )

    if res["status"] == "success":
        return True
    else:
        return False




def produce_latest_gradients(job_id, data):
    #from app.helpers.weights_helper import write_to_tmp_file
    #write_to_tmp_file(data)
    res = __produce_message(
        partition=f"{job_id}",
        topic=f"{PS_GRADIENTS_TOPIC}",
        message=data,
    )

    if res["status"] == "success":
        return True
    else:
        return False


def get_topic_prefix():
    return ""
    """env = config.ENVIRONMENT_NAME.lower()
    if env == "prod":
        return "prod"
    elif env == "preprod":
        return "pre"  # pre is used in docflow
    elif env == "sandbox":
        return "sandbox"
    elif env == "demo":
        return "demo"
    elif env == "stage":
        return "stage"
    elif env == "dev":
        return "dev"
    elif env in ["pre-dev", "predev", "pre_dev"]:  # not known for now
        return "pre_dev"
    else:
        return "dev"
        """


def shutdown_producer():
    if producer:
        producer.flush()  # Flush any remaining messages before shutdown
        print("Kafka Producer is shutdown.")
    try:
        #cleanup_kafka_files()
        pass
    except Exception as ex:
        print(
            f"Kafka Error while cleaningup files cleanup_kafka_files() : ERROR: {str(ex)} "
        )
