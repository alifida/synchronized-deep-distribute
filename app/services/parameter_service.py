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
from app.models.dataset_Img import DatasetImg


class ParameterService:
    weights_store = {}  # In-memory storage for weights by job_id
    gradients_store = {}  # In-memory storage for gradients by job_id
    metrics_store ={} # In-memory storage for metrics by job_id
    stats_store ={}
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
            return {
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

            db = get_db()
            TrainingJobDAO.update_training_job_status(db, job_id, "COMPLETED")
            traing_job = TrainingJobDAO.fetch_training_job_by_id(db, job_id)
            dataset  = DatasetImgDAO.fetch_dataset_image_by_id(traing_job.dataset_img_id)
            class_labels = get_class_labels(dataset.extracted_path)
            trained_model = TrainedModel(
                model_file=model_file_path,
                description=traing_job.algo,
                status="Temp",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                key_attributes="",
                class_label=class_labels,
                dataset_id=traing_job.dataset_img_id,
                user_id=dataset.user_id
            )

            TrainedModelDAO.save_trained_model(db, trained_model)

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