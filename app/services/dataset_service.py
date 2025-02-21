import os
import aiohttp
from aiohttp import ClientSession
import shutil
from pathlib import Path
import asyncio
from .redis_service import redis_client
from app.helpers.util import get_dataset_path


class DatasetService:
    @staticmethod
    async def download_images(job_id: str, examples: list):
        """
        Download the images from the image_urls in parallel and add them to the job context.
        """
         
        # Create an aiohttp session for downloading images
        async with aiohttp.ClientSession() as session:
            dataset_dir = get_dataset_path(job_id)
            for example in examples:
                await DatasetService.fetch_image_to_disk(session, example["url"], job_id, dataset_dir)
        return True
    
    

    async def download_images_wait_for_all(job_id: str, image_urls: list[str]):
        """
        Download the images from the image_urls in parallel and add them to the job context.
        """
        # Create an aiohttp session for downloading images
        async with aiohttp.ClientSession() as session:
            # List of tasks to download images
            download_tasks = [
                DatasetService.fetch_image(session, url, job_id)
                for url in image_urls
            ]
            # Wait for all downloads to complete (downloads happen in the background)
            await asyncio.gather(*download_tasks)

        return True

    @staticmethod
    async def fetch_image(session: ClientSession, url: str, job_id: str):
        """
        Fetch an image from a URL, store it in the job context, and update the downloaded count.
        """
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    image_data = await response.read()
                    # Store the image data in Redis using job_id and url as the key
                    stored = redis_client.store_image_data(job_id, url, image_data)
                    if stored:
                        print(f"Image for {job_id}_{url} has been successfully stored in Redis.")
                    else:
                        print(f"Failed to store image {job_id}_{url} in Redis.")
                else:
                    print(f"Failed to download {url}. Status code: {response.status}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")


    @staticmethod
    async def fetch_image_to_disk(session: ClientSession, url: str, job_id: str, directory: str):
        """
        Fetch an image from a URL and save it to the disk.
        """
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    image_data = await response.read()
                    # Write the image data to a file on disk
                    image_filename = f"{url.split('/')[-1]}"
                    image_path = os.path.join(directory, image_filename)
                    with open(image_path, "wb") as f:
                        f.write(image_data)
                    print(f"Image saved to disk at {image_path}")
                else:
                    print(f"Failed to download {url}. Status code: {response.status}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")


    @staticmethod
    def load_image_from_disk(dataset_dir:str, job_id: str, url: str) -> bytes:
        """
        Loads an image from disk based on the provided job_id and url.
        
        Args:
            dataset_dir (str): The directory where dataset images are stored.
            job_id (str): The job ID associated with the image.
            url (str): The URL used to form the image filename.
            
        Returns:
            bytes: The image data in bytes if found, otherwise raises FileNotFoundError.
        """
        
        # Generate the image filename based on job_id and URL
        image_filename = f"{url.split('/')[-1]}"  # Using the last part of the URL as filename
        image_path = os.path.join(dataset_dir, image_filename)

        # Try to read the image from disk
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()  # Read the image data in binary
            return image_data
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found at {image_path}")

    @staticmethod
    def delete_dataset(dataset_dir: str):
        # Convert dataset_dir to a Path object
        dataset_path = Path(dataset_dir)

        # Check if the directory exists
        if dataset_path.exists() and dataset_path.is_dir():
            try:
                # Remove the directory and all its contents
                shutil.rmtree(dataset_path)
                print(f"Directory {dataset_dir} and all its contents have been deleted.")
            except Exception as e:
                print(f"Error while deleting the directory: {e}")
        else:
            print(f"The directory {dataset_dir} does not exist or is not a valid directory.")