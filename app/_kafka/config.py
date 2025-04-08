from app.config import settings
import tempfile
import os

# Kafka producer configuration


def create_temp_file(content, suffix=".pem"):
    # Create a temporary file to hold the content
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with open(temp_file.name, "w") as f:
        f.write(content)
    return temp_file.name

"""


def cleanup_kafka_files():
    for file_path in [
        kafka_config["ssl.certificate.location"],
        kafka_config["ssl.key.location"],
        kafka_config["ssl.ca.location"],
    ]:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted temporary file: {file_path}")

 
"""