import asyncio
from app.background_tasks.worker_sync import fetch_ps_gradients,reset_ps_gradients, submit_gradients, submit_metrics, submit_weights, submit_stats
import tensorflow as tf
from collections import deque
from app.helpers.weights_helper import aggregate
from .redis_service import redis_client
from app.services.dataset_service import DatasetService
import time
from app.helpers.util import get_dataset_path
from app.db.database import get_db
from app.dao.dataset_Img_dao import DatasetImgDAO
from app.dao.training_job_dao import  TrainingJobDAO
import numpy as np

#from app.services.model_loader import load_model  # Hypothetical helper to load models dynamically
from app.services.keras_catalog_service import KerasCatalogService as kerasService

latest_gradients_store={}

class WorkerService:

    @staticmethod
    async def initialize_training(worker_id:str, job_id: str, init_params: dict):
        """
        Initialize worker with training parameters and start training.
        """
        try:
            # Save context for the job
            total_images = len(init_params["dataset"]["images"])
            classes = init_params["dataset"]["classes"]
            downloaded_images_count = 0     # Initially 0 images downloaded
            processed_images_count = 0            # Initially 0 images processed (for training)
            job_context = {
                "init_params": init_params,
                "total_images": total_images,
                "downloaded_images_count": downloaded_images_count,
                "processed_images_count": processed_images_count,
                "classes": classes
            }

            #save in redis
            redis_client.save_job_context(job_id, job_context)
            examples = init_params["dataset"]["images"]
            if (not len(examples)>0):
                raise ValueError("Error: No image provided in the dataset.")

            # Start downloading images in the background
            #asyncio.create_task(DatasetService.download_images(job_id, image_urls))
            download_started_at = time.time()
            await DatasetService.download_images(job_id, examples)
            download_ended_at =  time.time()
            # Start training in the background
            asyncio.create_task(WorkerService.start_training(worker_id, job_id))

            return True
        except Exception as e:
            print(f"Error during training for job {job_id}: {str(e)}")
            return False

    @staticmethod
    async def start_training(worker_id:str, job_id: str):
        """
        Start training using the provided model and images.
        """


        dataset_dir = get_dataset_path(job_id)

        training_stats = {
            "training_started_at": time.time(),
            "status" : "started"
        }
        try:
            # Retrieve job-specific context
            job_context = redis_client.get_job_context(job_id)

            init_params = job_context["init_params"]
            total_epoch = init_params.get("total_epoch", 1)
            parameter_sever_url = init_params["parameter_sever_url"]
            classes = job_context["classes"]


            # Initialize model
            learning_rate = init_params.get("learning_rate", 0.001)
            if isinstance(learning_rate, str):
                try:
                    learning_rate = float(learning_rate)
                except ValueError:
                    print("Error: Invalid value for epoch, unable to convert to float.")
                    learning_rate = 0.001  

            if isinstance(total_epoch, str):
                try:
                    total_epoch = int(total_epoch)
                except ValueError:
                    print("Error: Invalid value for learning_rate, unable to convert to float.")
                    total_epoch = 1
      

            # Load model dynamically based on model_name
            base_model = kerasService.get_model_object(init_params['algo_name'])  # Get pre-trained model
            base_model.trainable = True  # Allow fine-tuning

            """
            # Add classification head
            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(1)  # Binary classification output
            ])

            """

             # Build full model
            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(len(classes), activation="softmax")  # Multi-class classification
            ])
            optimizer = tf.keras.optimizers.Adam(learning_rate)
            loss_fn = tf.keras.losses.CategoricalCrossentropy()

            # Initialize metrics
            accuracy_metric = tf.keras.metrics.BinaryAccuracy()
            precision_metric = tf.keras.metrics.Precision()
            recall_metric = tf.keras.metrics.Recall()
            auc_metric = tf.keras.metrics.AUC()
            f1_score = tf.keras.metrics.Mean()  # Custom F1 score tracking (manual aggregation)


            
            first_time =True

            for epoch in range(total_epoch):
                print(f"Starting epoch {epoch + 1}/{total_epoch}")
                images_q = deque(init_params["dataset"]["images"])

                processed_images_count=0
                # Training loop
                while images_q:
                    # Simulate loading and preprocessing the image
                    example = images_q[0]

                    image = DatasetService.load_image_from_disk(dataset_dir,job_id,example["url"])
                    if not image:
                        # Image is not available/downloaded; move it to the end of the queue
                        images_q.append(images_q.popleft())
                    else:


                        image_tensor = tf.image.decode_image(image, channels=3)
                        image_tensor = tf.image.resize(image_tensor, (150, 150))
                        image_tensor = tf.expand_dims(image_tensor / 255.0, axis=0)  # Normalize and batch

                        # Fetch label
                        #label = init_params.get("label", 0)  # Assume binary label for simplicity
                        label = example["label"]
                        label_tensor = tf.one_hot(label, depth=len(classes))
                        label_tensor = tf.reshape(label_tensor, (1, len(classes)))
                        # Perform a training step
                        with tf.GradientTape() as tape:
                            predictions = model(image_tensor, training=True)
                            loss = loss_fn(label_tensor, predictions)
                        if first_time:
                            gradients = tape.gradient(loss, model.trainable_variables)
                            gradients = WorkerService.compress_gradients(gradients)
                            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                            first_time=False

                        # Update metrics
                        accuracy_metric.update_state(label_tensor, predictions)
                        precision_metric.update_state(label_tensor, predictions)
                        recall_metric.update_state(label_tensor, predictions)
                        auc_metric.update_state(label_tensor, predictions)

                        # Compute F1 Score
                        precision = precision_metric.result().numpy()
                        recall = recall_metric.result().numpy()
                        if precision + recall > 0:
                            f1 = 2 * (precision * recall) / (precision + recall)
                        else:
                            f1 = 0.0
                        f1_score.update_state(f1)

                        #print(f"Processed image. Loss: {loss.numpy()}")
                        if processed_images_count % 1 ==0:
                            await submit_gradients(parameter_sever_url, worker_id, job_id, gradients)
                         
                        # Fetch the latest aggregated weights and update the model
                        latest_gradients = fetch_ps_gradients(job_id)
                        if latest_gradients:
                            optimizer.apply_gradients(zip(latest_gradients, model.trainable_variables))
                            reset_ps_gradients(job_id)
                        # submit results
                        if init_params.get("post_realtime_results", True):
                            metrics = {
                                "accuracy" : accuracy_metric.result().numpy(),
                                "precision" : precision_metric.result().numpy(),
                                "recall" : recall_metric.result().numpy(),
                                "auc" : auc_metric.result().numpy(),
                                "f1_final" : f1_score.result().numpy(),
                            }
                            await submit_metrics(parameter_sever_url, worker_id,job_id, metrics)


                        images_q.popleft()
                        processed_images_count +=1


                        first_time = False
                        print(f"----************############### epoch: {epoch}/{total_epoch}-- example processed: {processed_images_count}")

            # Final metrics computation
            accuracy = accuracy_metric.result().numpy()
            precision = precision_metric.result().numpy()
            recall = recall_metric.result().numpy()
            auc = auc_metric.result().numpy()
            f1_final = f1_score.result().numpy()

            # Prepare metrics dictionary
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'auc': auc,
                'f1_score': f1_final
            }

            await submit_weights(parameter_sever_url, worker_id=worker_id, job_id=job_id, weights=model.get_weights())
            training_stats["status"] = "completed"
        except Exception as e:
            print(f"Error during training for job {job_id}: {str(e)}")
            training_stats["status"] = "error"
            return None
        finally:
            #training_stats ["epoch"] = epoch
            training_stats ["training_ended_at"] = time.time()
            await submit_stats(parameter_sever_url, worker_id=worker_id, job_id=job_id,  stats=training_stats)
            redis_client.clear_job_context(job_id)  # Clean up context
            DatasetService.delete_dataset(dataset_dir)

    @staticmethod
    def compress_gradients(gradients, bit_width=4, sparsity=0.9):
        """
        Apply quantization & sparsification to reduce gradient size.
        """
        # Quantization (Convert to 8-bit)
        try:
            scale = 2 ** bit_width - 1

            #  Filter out None gradients
            gradients = [grad for grad in gradients if grad is not None]

            #  Convert to NumPy before applying astype()
            gradients = [(grad.numpy() * scale).astype(np.int8) / scale for grad in gradients]

            # Sparsification (Zero out small gradients)
            for i in range(len(gradients)):
                grad = gradients[i]

                #  Compute size safely
                grad_size = np.size(grad)
                if grad_size == 0:  # Skip empty gradients
                    continue

                k = int(sparsity * grad_size)
                threshold = np.sort(np.abs(grad), axis=None)[k]  # Use NumPy sorting
                gradients[i] = np.where(np.abs(grad) < threshold, np.zeros_like(grad), grad)

            #  Convert back to TensorFlow tensors
            gradients = [tf.convert_to_tensor(grad, dtype=tf.float32) for grad in gradients]

            #  Gradient Clipping
            gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]

        except Exception as e:
            print(f"Error in compress_gradients, Exception: {str(e)}")
        return gradients