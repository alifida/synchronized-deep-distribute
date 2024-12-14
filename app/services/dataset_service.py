
import aiohttp
from aiohttp import ClientSession
import asyncio
from .redis_service import redis_client


class DatasetService:
    @staticmethod
    async def download_images(job_id: str, examples: list):
        """
        Download the images from the image_urls in parallel and add them to the job context.
        """
        # Create an aiohttp session for downloading images
        async with aiohttp.ClientSession() as session:

            for example in examples:
                await DatasetService.fetch_image(session, example["url"], job_id)
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


