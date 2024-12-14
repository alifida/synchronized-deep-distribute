import redis
import json
from typing import Optional, Any
from fastapi import HTTPException
from app.config.settings import REDIS_URL  # Import the Redis connection string
from app.helpers.weights_helper import serialize_weights, deserialize_weights
class RedisClientService:

    JOB_PREFIX="job_id_"

    def __init__(self, redis_url: str = REDIS_URL):
        try:
            self.redis = redis.from_url(redis_url, decode_responses=False, )
        except redis.RedisError as e:
            raise HTTPException(status_code=500, detail=f"Failed to connect to Redis: {str(e)}")

    def update_weights(self, job_id, weights):

        key = f"{self.JOB_PREFIX}{job_id}_latest_weights"

        """Sets data to Redis with optional TTL."""
        try:
            if weights:
                weights = serialize_weights(weights)
            return self.redis.set(key, weights)
        except redis.RedisError as e:
            raise HTTPException(status_code=500, detail=f"Redis Error: {str(e)}")



    def get_latest_weights(self, job_id):
        key = f"{self.JOB_PREFIX}{job_id}_latest_weights"
        try:
            weights = self.redis.get(key)
            if weights:
                return deserialize_weights(weights)
            return None
        except redis.RedisError as e:
            raise HTTPException(status_code=500, detail=f"Redis Error: {str(e)}")

    def clear_job_context(self, job_id):
        return self.__delete_by_prefix(f"{self.JOB_PREFIX}{job_id}")

    def save_job_context(self, job_id, job_context):
        return self.__set_data(f"{self.JOB_PREFIX}{job_id}_job_context", job_context)

    def get_job_context(self, job_id):
        return self.__get_data(f"{self.JOB_PREFIX}{job_id}_job_context")



    def store_image_data(self, job_id: str, url: str, image_data: bytes, ttl: Optional[int]=None) -> bool:
        """Stores the downloaded image data in Redis with a unique key (job_id + "_" + url)."""
        key = f"{self.JOB_PREFIX}{job_id}_{url}"
        try:
            if ttl:
                # Use the TTL parameter to set the time-to-live for the key
                return self.redis.setex(key, ttl, image_data)
            return self.redis.set(key, image_data)
        except redis.RedisError as e:
            raise HTTPException(status_code=500, detail=f"Redis Error: {str(e)}")


    def get_image_data(self, job_id: str, url: str) -> Optional[bytes]:
        """Fetches the image data from Redis using the unique key (job_id + "_" + url)."""
        key = f"{self.JOB_PREFIX}{job_id}_{url}"

        try:
            data = self.redis.get(key)
            if data:
                return data
            return None  # Return None if the key does not exist
        except redis.RedisError as e:
            raise HTTPException(status_code=500, detail=f"Redis Error: {str(e)}")









    def __delete_by_prefix(self, prefix):
        # Use SCAN to iterate over the keys
        cursor = 0
        while True:
            # SCAN command returns a cursor and a list of keys
            cursor, keys = self.redis.scan(cursor, match=f'{prefix}*')

            # Delete the keys that match the prefix
            if keys:
                self.redis.delete(*keys)
                print(f"Deleted keys: {keys}")

            # If cursor is 0, it means we've finished scanning
            if cursor == 0:
                break

        print("All matching keys deleted.")

    def __set_data(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Sets data to Redis with optional TTL."""
        try:
            if ttl:
                # Use the TTL parameter to set the time-to-live for the key
                return self.redis.setex(key, ttl, json.dumps(value))
            return self.redis.set(key, json.dumps(value))
        except redis.RedisError as e:
            raise HTTPException(status_code=500, detail=f"Redis Error: {str(e)}")

    def __get_data(self, key: str) -> Optional[Any]:
        """Gets data from Redis."""
        try:
            data = self.redis.get(key)
            if data:
                return json.loads(data)
            return None  # Return None if the key does not exist
        except redis.RedisError as e:
            raise HTTPException(status_code=500, detail=f"Redis Error: {str(e)}")



    def __delete_data(self, key: str) -> bool:
        """Deletes a key from Redis."""
        try:
            return self.redis.delete(key) > 0
        except redis.RedisError as e:
            raise HTTPException(status_code=500, detail=f"Redis Error: {str(e)}")


# Singleton instance of RedisHelper
redis_client = RedisClientService()
