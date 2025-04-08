from urllib.parse import urlparse
import os
from pathlib import Path
import random
import string
import time


def get_label_by_url(url):
    # Parse the URL
    parsed_url = urlparse(url)

    # Get the path part of the URL and split it by '/'
    path_parts = parsed_url.path.strip('/').split('/')

    # Return the second-to-last element, which is the parent directory
    if len(path_parts) >= 2:
        return path_parts[-2]
    return None  # If there is no parent directory



def prepare_path(path: str):
    # Convert the path string to a Path object
    dir_path = Path(path)
    
    # Check if the path exists, if not, create it along with intermediate directories
    if not dir_path.exists():
        # Create all directories in the path if they don't exist
        os.makedirs(dir_path, exist_ok=True)

def get_dataset_path(job_id):
    from main import DATASET_DIR
    path = f"{DATASET_DIR}/{job_id}/"
    prepare_path(path=path)
    return path





def generate_unique_string(length=3):
    # Combine the current time (in seconds) and a random string
    timestamp = int(time.time())  # Get the current time as a unique timestamp
    random_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

    # Create a unique string by combining the timestamp with the random string
    unique_string = f"{timestamp}_{random_string}"

    return unique_string

def get_host():
    import socket
    return socket.gethostname()

