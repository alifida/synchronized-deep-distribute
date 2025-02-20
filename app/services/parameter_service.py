import re
import json
import aiohttp
from app.db.database import get_db
from app.config.settings import TRAINED_MODELS_DIR
from app.helpers.weights_helper import aggregate, aggregate_gradients
import pickle
import os
from app.services.keras_catalog_service import KerasCatalogService as kerasService
from app.dao.training_job_dao import TrainingJobDAO
from app.dao.trained_model_dao import TrainedModelDAO
from app.dao.dataset_Img_dao import DatasetImgDAO
from app.models.trained_model import TrainedModel
import datetime
from app.models.training_job import TrainingJob
from app.dao.cluster_nodes_dao import ClusterNodesDAO
from app.models.cluster_node import ClusterNode
from typing import List, Dict
import math
class ParameterService:
    weights_store = {}  # In-memory storage for weights by job_id
    gradients_store = {}  # In-memory storage for gradients by job_id
    metrics_store ={} # In-memory storage for metrics by job_id
    stats_store ={}


    async def start_training_job(data):
        init_params = data.get('init_params')
        job_data = init_params.get('job_data')
        parameter_settings = job_data.get('parameter_settings')
        #dataset_img = DatasetImgDAO.fetch_all_dataset_images(job_data.get('dataset_img'))
        dataset_details = init_params.get('dataset_details')
       
        dataset_host_url = dataset_details.get('host_url')
        del dataset_details['host_url'] #remove the host url from dataset
        train_ds_details = dataset_details.get('train')
        #classes = train_ds_details.get("class_names")
        classwise_details = train_ds_details.get("classwise_details")


        cluster_id = parameter_settings.get("cluster")
        async for db in get_db():
            workers =await ClusterNodesDAO.fetch_workers_by_cluster_id(db, cluster_id)

        print("-------------------------------------------")
        print("-------------------------------------------")
        print("-------------------------------------------")
        print("-------------------------------------------")

        print (json.dumps(classwise_details))

        subsets = await ParameterService.divide_dataset_classwise(classwise_details, workers, dataset_host_url)
        

        print("--------------------")

        response = await ParameterService.distribute_training_amoung_workers(subsets, workers, parameter_settings, f"job_id_{job_data['job_id']}")

    @staticmethod
    async def distribute_training_amoung_workers(subsets, workers, parameter_settings, job_id):
        # Clean the parameter settings
        parameter_settings['total_epoch'] = parameter_settings['epochs']
        
        del parameter_settings["dataset_id"]
        del parameter_settings["cluster"]
        del parameter_settings["strategy"]
        del parameter_settings["epochs"]
         
        payload = parameter_settings
       

        responses = {}

        # Create an aiohttp session for making asynchronous requests
        async with aiohttp.ClientSession() as session:
            # Iterate over the workers
            for worker in workers:
                worker_key = ParameterService.get_worker_key(worker)
                
                # Assign dataset to the payload based on worker key
                payload["dataset"] = subsets[worker_key]["dataset"]
                
                # Create the worker URL with the job and worker ID
                worker_url = f"http://{worker.ip_address}:{worker.port}/worker/init-training?worker_id=worker_{worker.id}&job_id={job_id}"
                
                try:
                    # Send the POST request to the worker's URL
                    async with session.post(worker_url, json=payload) as response:
                        
                        # Check if the request was successful (HTTP status code 200)
                        if response.status == 200:
                            responses[worker_key] = await response.json()  # Collect the response as a JSON object
                        else:
                            responses[worker_key] = {"error": f"Failed with status code {response.status}"}

                except aiohttp.ClientError as exc:
                    # Handle request errors
                    responses[worker_key] = {"error": f"Request failed: {exc}"}

        # Return the collected responses
        return responses



    @staticmethod
    async def divide_dataset_classwise(classwise_details: List[Dict], workers: List[ClusterNode], dataset_host_url:str):
        total_workers = len(workers)
        
        # Step 1: Create a mapping of class names to labels
        class_labels = {class_details["class_name"]: idx for idx, class_details in enumerate(classwise_details)}

        # Step 2: Prepare to divide the dataset for each worker
        worker_datasets = {f"w_{worker.ip_address}_{worker.port}": {"dataset": {"images": []}} for worker in workers}

        # Step 3: Divide the images of each class equally among the workers
        for class_details in classwise_details:
            class_name = class_details["class_name"]
            total_examples = class_details["total_examples"]
            preview_images = class_details["preview_images"]
            class_label = class_labels[class_name]
            
            # Calculate how many images each worker should get
            images_per_worker = math.ceil(total_examples / total_workers)
            
            # Step 4: Assign images to workers
            for idx, worker in enumerate(workers):
                worker_label = ParameterService.get_worker_key(worker)
                
                # Slice the list of preview images for this worker
                start_idx = idx * images_per_worker
                end_idx = start_idx + images_per_worker
                
                # Ensure we don't exceed the total examples
                worker_images = preview_images[start_idx:end_idx]
                
                # Append the images with the appropriate label
                for image_url in worker_images:
                    image_url = image_url.replace("\\", "/").replace("\\\\", "/")
                    worker_datasets[worker_label]["dataset"]["images"].append({
                        "url": f"{dataset_host_url}{image_url}",
                        "label": class_label
                    })
        
        # Step 5: Return the worker datasets
        #return list(worker_datasets.values())
        return worker_datasets

    def get_worker_key(worker):
        return f"w_{worker.ip_address}_{worker.port}"
    
    

    def get_worker_url_by_key(key):
        # Extract the worker's IP address and port from the key using regular expressions
        match = re.match(r"w_(\S+)_(\d+)", key)
        if match:
            ip_address = match.group(1)
            port = match.group(2)
            
            # Construct the URL using the extracted IP and port
            url = f"http://{ip_address}:{port}"
            return url
        else:
            raise ValueError("Invalid key format")


    @staticmethod
    async def aggregate_metrics(worker_id: str, job_id: str, metrics: dict):
        try:
            # Ensure the job exists in metrics_store; if not, create an entry for the job
            if job_id not in ParameterService.metrics_store:
                ParameterService.metrics_store[job_id] = {
                    "workers": {},
                    "overall_metrics": {}
                }

            # Store the metrics for the specific worker
            ParameterService.metrics_store[job_id]["workers"][worker_id] = {"metrics": metrics}

            # Now aggregate metrics from all workers
            all_worker_metrics = ParameterService.metrics_store[job_id]["workers"]

            # Initialize the aggregated metrics
            aggregated_metrics = {key: 0 for key in metrics.keys()}

            # Iterate over all workers' metrics and accumulate the values
            num_workers = len(all_worker_metrics)
            for worker in all_worker_metrics.values():
                worker_metrics = worker["metrics"]
                for key in aggregated_metrics:
                    aggregated_metrics[key] += worker_metrics.get(key, 0)

            # Average the accumulated metrics
            for key in aggregated_metrics:
                aggregated_metrics[key] /= num_workers

            # Store the averaged metrics in overall_metrics
            ParameterService.metrics_store[job_id]["overall_metrics"] = aggregated_metrics

            print(f"Aggregated metrics for job {job_id}: {aggregated_metrics}")
            return aggregated_metrics

        except Exception as e:
            print(f"Error aggregating metrics for job {job_id}: {str(e)}")
            return None


    @staticmethod
    async def get_overall_metrics(job_id: str):
        try:
            # Check if the job_id exists in the metrics_store
            if job_id not in ParameterService.metrics_store:
                raise ValueError(f"Job ID {job_id} not found in metrics store.")

            # Retrieve overall metrics for the specified job
            overall_metrics = ParameterService.metrics_store[job_id].get("overall_metrics", None)

            if overall_metrics is None:
                raise ValueError(f"No aggregated metrics found for job ID {job_id}.")

            # Return the overall aggregated metrics
            return overall_metrics

        except Exception as e:
            print(f"Error retrieving overall metrics for job {job_id}: {str(e)}")
            return None

    @staticmethod
    async def aggregate_weights(worker_id: str, job_id: str, worker_weights: dict):
        """
        Aggregate weights from a worker and update in memory.
        """
        if job_id not in ParameterService.weights_store:
            ParameterService.weights_store[job_id] = worker_weights
        else:
            existing_weights = ParameterService.weights_store[job_id]
            existing_weights = aggregate(existing_weights, worker_weights)
            ParameterService.weights_store[job_id] = existing_weights
        return True

    @staticmethod
    async def get_aggregated_weights( job_id: str):
        """
        Return aggregated weights for the given job_id.
        """
        return ParameterService.weights_store.get(job_id)



    @staticmethod
    async def aggregate_gradients(worker_id: str, job_id: str, worker_gradients: dict):
        """
        Aggregate gradients from a worker and update in memory.
        """
        if job_id not in ParameterService.gradients_store:
            ParameterService.gradients_store[job_id] = worker_gradients
        else:
            existing_gradients = ParameterService.gradients_store[job_id]
            existing_gradients = aggregate_gradients(existing_gradients, worker_gradients)
            ParameterService.gradients_store[job_id] = existing_gradients
        return True

    @staticmethod
    async def get_aggregated_gradients(worker_id:str, job_id: str):
        """
        Return aggregated gradients for the given job_id.
        """
        return ParameterService.gradients_store.get(job_id)

    @staticmethod
    async def update_stats(worker_id:str, job_id: str, stats):

        try:
            # Ensure the job exists in stats_store; if not, create an entry for the job
            if job_id not in ParameterService.stats_store:
                ParameterService.stats_store[job_id] = {
                    "workers": {},
                }

            # Store the stats for the specific worker
            ParameterService.stats_store[job_id]["workers"][worker_id] = {"stats": stats}

            if await ParameterService.all_workers_done(job_id):
                await ParameterService.save_trained_model(job_id)


            return "success"

        except Exception as e:
            print(f"Error aggregating stats for job {job_id}: {str(e)}")
            raise e

    @staticmethod
    async def get_stats(job_id: str):
        return ParameterService.stats_store[job_id]

    @staticmethod
    async def all_workers_done(job_id):
        job_stats = ParameterService.stats_store.get(job_id, None)
        total_workers = len(job_stats["workers"])
        completed_workers = 0
        failed_workers = 0

        if not job_stats:
            raise ValueError(f"No stats found for job {job_id}.")
        # Analyze worker stats
        workers = job_stats["workers"]
        for worker_id, worker_data in workers.items():
            stats = worker_data.get("stats", {})
            status = stats.get("status", "failed")
            if status == "completed":
                completed_workers += 1
            else:
                failed_workers += 1



        # Ensure all workers completed training
        if completed_workers + failed_workers < total_workers:
            return False
        else:
            return True

    @staticmethod
    async def save_trained_model(job_id: str):
        try:
            # Check if all workers have submitted stats
            job_stats = ParameterService.stats_store.get(job_id, None)
            if not job_stats:
                raise ValueError(f"No stats found for job {job_id}.")

            workers = job_stats["workers"]
            total_workers = len(workers)
            completed_workers = 0
            failed_workers = 0
            training_durations = []
            total_epochs = []

            # Analyze worker stats
            for worker_id, worker_data in workers.items():
                stats = worker_data.get("stats", {})
                status = stats.get("status", "failed")
                if status == "completed":
                    completed_workers += 1
                else:
                    failed_workers += 1
                training_durations.append(stats.get("training_duration", 0))
                total_epochs.append(stats.get("epochs", 0))

            # Get individual and overall metrics
            individual_metrics = {worker_id: worker_data.get("stats", {}).get("metrics", {}) for worker_id, worker_data in workers.items()}
            overall_metrics = await ParameterService.get_overall_metrics(job_id)

            # Aggregate weights to form the final model
            aggregated_weights = await ParameterService.get_aggregated_weights(job_id)
            if not aggregated_weights:
                raise ValueError(f"No weights found for job {job_id}.")

            # Save final model to a file
            model_file_path = f"{TRAINED_MODELS_DIR}/{job_id}_final_model.pkl"
            with open(model_file_path, "wb") as model_file:
                pickle.dump(aggregated_weights, model_file)

            # Prepare and return training details
            result = {
                "job_id": job_id,
                "total_workers": total_workers,
                "completed_workers": completed_workers,
                "failed_workers": failed_workers,
                "training_durations": training_durations,
                "total_epochs": total_epochs,
                "individual_metrics": individual_metrics,
                "overall_metrics": overall_metrics,
                "model_file_path": model_file_path
            }

            async for db in get_db():  # Await the async generator

                training_job = await TrainingJobDAO.fetch_training_job_by_id(db, job_id)
                if not training_job:
                    training_job = TrainingJob()
                    training_job.job_name = job_id
                    training_job.dataset_img_id=1
                    training_job.algo =''
                    training_job.user_id=1
                training_job.result = json.dumps(result)
                training_job.status = "COMPLETED"
                await TrainingJobDAO.save(db, training_job)
                current_time = datetime.datetime.now()
                dataset  = await DatasetImgDAO.fetch_dataset_image_by_id(db, training_job.dataset_img_id)
                class_labels = await get_class_labels(dataset.extracted_path)
                trained_model = TrainedModel(
                    model_file=model_file_path,
                    description=training_job.algo,
                    status="Temp",
                    created_at=current_time,
                    updated_at=current_time,
                    key_attributes="",
                    class_label=json.dumps(class_labels),
                    dataset_img_id=training_job.dataset_img_id,
                    user_id=dataset.user_id
                )

                await TrainedModelDAO.save_trained_model(db, trained_model)

        except Exception as e:
            print(f"Error in preparing trained model details for job {job_id}: {str(e)}")
            return {"status": "error", "message": str(e)}



def convert_pkl_to_h5(algo_name: str, pkl_file_path: str):
    try:

        # Get the model object from the dictionary
        base_model = kerasService.get_model_object(algo_name)
        #base_model = MODELS_DICT[algo_name](weights=None)  # Initialize model without weights

        # Load the .pkl file to get the weights
        with open(pkl_file_path, 'rb') as f:
            model_weights = pickle.load(f)

        # Load the weights into the model
        base_model.set_weights(model_weights)

        # Prepare the path for the .h5 file (same name as .pkl but with .h5 extension)
        h5_file_path = os.path.splitext(pkl_file_path)[0] + '.h5'

        # Save the model in .h5 format
        base_model.save(h5_file_path)

        return h5_file_path

    except Exception as e:
        print(f"Error during conversion: {e}")
        return None


async def get_class_labels(directory_path):
    classes=[]
    for label in os.listdir(directory_path):
        classes.append(os.path.join(directory_path, label))
    return classes