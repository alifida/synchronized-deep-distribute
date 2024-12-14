import asyncio
from app.background_tasks.worker_sync import submit_weights, fetch_latest_weights
import tensorflow as tf
from collections import deque
from app.helpers.weights_helper import aggregate
from .redis_service import redis_client
from app.services.dataset_service import DatasetService

import threading

#from app.services.model_loader import load_model  # Hypothetical helper to load models dynamically
from app.services.keras_catalog_service import KerasCatalogService as kerasService

class WorkerService:



    @staticmethod
    async def initialize_training(job_id: str, init_params: dict):
        """
        Initialize worker with training parameters and start training.
        """
        # Save context for the job
        total_images = len(init_params["dataset"]["images"])
        downloaded_images_count = 0     # Initially 0 images downloaded
        processed_images_count = 0            # Initially 0 images processed (for training)
        job_context = {
            "init_params": init_params,
            "total_images": total_images,
            "downloaded_images_count": downloaded_images_count,
            "processed_images_count": processed_images_count
        }

        #save in redis
        redis_client.save_job_context(job_id, job_context)
        examples = init_params["dataset"]["images"]

        # Start downloading images in the background
        #asyncio.create_task(DatasetService.download_images(job_id, image_urls))
        await DatasetService.download_images(job_id, examples)

        # Start training in the background
        asyncio.create_task(WorkerService.start_training(job_id))

        return True

    @staticmethod
    async def start_training(job_id: str):
        """
        Start training using the provided model and images.
        """
        try:
            # Retrieve job-specific context
            job_context = redis_client.get_job_context(job_id)

            init_params = job_context["init_params"]
            total_epoch = init_params.get("total_epoch", 1)
            total_images = job_context["total_images"]
            downloaded_images_count = job_context["downloaded_images_count"]
            #processed_images_count = job_context["processed_images_count"]

            # Initialize model
            learning_rate = init_params.get("learning_rate", 0.001)

            # Load model dynamically based on model_name
            base_model = kerasService.get_model_object(init_params['algo_name'])  # Get pre-trained model
            base_model.trainable = True  # Allow fine-tuning

            # Add classification head
            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(1)  # Binary classification output
            ])

            optimizer = tf.keras.optimizers.Adam(learning_rate)
            loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

            # Initialize metrics
            accuracy_metric = tf.keras.metrics.BinaryAccuracy()
            precision_metric = tf.keras.metrics.Precision()
            recall_metric = tf.keras.metrics.Recall()
            auc_metric = tf.keras.metrics.AUC()
            f1_score = tf.keras.metrics.Mean()  # Custom F1 score tracking (manual aggregation)

            for epoch in range(total_epoch):
                print(f"Starting epoch {epoch + 1}/{total_epoch}")
                images_q = deque(init_params["dataset"]["images"])

                processed_images_count=0
                # Training loop
                while images_q:
                    # Simulate loading and preprocessing the image
                    example = images_q[0]

                    image = redis_client.get_image_data(job_id, example["url"])
                    if not image:
                        # Image is not available/downloaded; move it to the end of the queue
                        images_q.append(images_q.popleft())
                    else:
                        # Fetch latest weights
                        await fetch_latest_weights(job_id=job_id)
                        #asyncio.create_task(fetch_latest_weights(job_id=job_id))
                        #fetch_weight_thread = threading.Thread(target=fetch_latest_weights, args=(job_id))
                        # Start the thread
                        #fetch_weight_thread.start()
                        # Preprocess image
                        image_tensor = tf.image.decode_image(image, channels=3)
                        image_tensor = tf.image.resize(image_tensor, (150, 150))
                        image_tensor = tf.expand_dims(image_tensor / 255.0, axis=0)  # Normalize and batch

                        # Fetch label
                        #label = init_params.get("label", 0)  # Assume binary label for simplicity
                        label = example["label"]
                        label_tensor = tf.constant([[label]], dtype=tf.float32)  # Match output shape (batch_size, 1)

                        # Perform a training step
                        with tf.GradientTape() as tape:
                            predictions = model(image_tensor, training=True)
                            loss = loss_fn(label_tensor, predictions)

                        gradients = tape.gradient(loss, model.trainable_variables)
                        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

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

                        print(f"Processed image. Loss: {loss.numpy()}")

                        # Submit updated weights to parameter server
                        await submit_weights(job_id=job_id, weights=model.trainable_variables)
                        #asyncio.create_task(submit_weights(job_id=job_id, weights=model.trainable_variables))

                        #submit_weight_thread = threading.Thread(target=submit_weights, args=(job_id, model.trainable_variables))
                        # Start the thread
                        #submit_weight_thread.start()

                        # Fetch the latest aggregated weights and update the model
                        latest_weights = redis_client.get_latest_weights(job_id)

                        if not latest_weights:
                            latest_weights = model.trainable_variables

                        # Aggregate weights from the parameter server with current iteration weights
                        latest_weights = aggregate(latest_weights, model.trainable_variables)
                        if latest_weights:
                            for var, latest in zip(model.trainable_variables, latest_weights):
                                var.assign(latest)
                        images_q.popleft()
                        processed_images_count +=1
                        print(f"------ example processed: {processed_images_count}")

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

            print(f"Training complete for job {job_id}. Metrics: {metrics}")
            return metrics

        except Exception as e:
            print(f"Error during training for job {job_id}: {str(e)}")
            return None
        finally:
            redis_client.clear_job_context(job_id)  # Clean up context



    @staticmethod
    async def start_training_______(job_id: str):
        """
        Start training using the provided model and images.
        """
        try:
            # Retrieve job-specific context
            job_context = redis_client.get_job_context(job_id)

            init_params = job_context["init_params"]
            total_epoch = init_params.get("total_epoch", 1)
            total_images = job_context["total_images"]
            downloaded_images_count = job_context["downloaded_images_count"]
            processed_images_count = job_context["processed_images_count"]
            image_urls_q = deque(init_params["dataset"]["image_urls"])

            # Initialize model
            learning_rate = init_params.get("learning_rate", 0.001)

            # Load model dynamically based on model_name
            base_model = kerasService.get_model_object(init_params['algo_name'])  # Get pre-trained model
            base_model.trainable = True  # Allow fine-tuning

            # Add classification head
            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(1)  # Binary classification output
            ])

            optimizer = tf.keras.optimizers.Adam(learning_rate)
            loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

            for epoch in range(total_epoch):
                print(f"Starting epoch {epoch + 1}/{total_epoch}")

                # Training loop
                while image_urls_q:
                    # Simulate loading and preprocessing the image
                    image = redis_client.get_image_data(job_id, image_urls_q[0])
                    if not image:
                        # Image is not available/downloaded; move it to the end of the queue
                        image_urls_q.append(image_urls_q.popleft())
                    else:
                        #asyncio.create_task(fetch_latest_weights(job_id=job_id))
                        await fetch_latest_weights(job_id=job_id)
                        # Preprocess image
                        image_tensor = tf.image.decode_image(image, channels=3)
                        image_tensor = tf.image.resize(image_tensor, (150, 150))
                        image_tensor = tf.expand_dims(image_tensor / 255.0, axis=0)  # Normalize and batch

                        # Fetch label
                        label = init_params.get("label", 0)  # Assume binary label for simplicity
                        label_tensor = tf.constant([[label]], dtype=tf.float32)  # Match output shape (batch_size, 1)

                        # Perform a training step
                        with tf.GradientTape() as tape:
                            predictions = model(image_tensor, training=True)
                            loss = loss_fn(label_tensor, predictions)

                        gradients = tape.gradient(loss, model.trainable_variables)
                        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                        print(f"Processed image. Loss: {loss.numpy()}")

                        # Submit updated weights to parameter server
                        #asyncio.create_task(submit_weights(job_id=job_id, weights=model.trainable_variables))
                        await submit_weights(job_id=job_id, weights=model.trainable_variables)
                        # Fetch the latest aggregated weights and update the model
                        latest_weights = redis_client.get_latest_weights(job_id)
                        if not latest_weights:
                            latest_weights = model.trainable_variables
                        # Aggregate weights from the parameter server with current iteration weights
                        latest_weights = aggregate(latest_weights, model.trainable_variables)
                        if latest_weights:
                            for var, latest in zip(model.trainable_variables, latest_weights):
                                var.assign(latest)
                        image_urls_q.popleft()

            print(f"Training complete for job {job_id}.")

        except Exception as e:
            print(f"Error during training for job {job_id}: {str(e)}")
        finally:
            redis_client.clear_job_context(job_id)  # Clean up context
