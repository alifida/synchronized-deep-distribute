import tensorflow as tf
import gzip
import pickle
import numpy as np


def aggregate(existing, fresh):
    """
    Aggregate weights using simple averaging.

    Args:
        existing (list): List of existing TensorFlow Variable objects or numpy arrays.
        fresh (list): List of fresh TensorFlow Variable objects or numpy arrays.

    Returns:
        list: Aggregated weights as TensorFlow Variables.
    """
    if len(existing) != len(fresh):
        raise ValueError("The length of existing and fresh weights must match.")

    aggregated_weights = []
    for existing_var, fresh_var in zip(existing, fresh):
        # Convert to numpy if needed
        existing_values = existing_var.numpy() if isinstance(existing_var, tf.Variable) else existing_var
        fresh_values = fresh_var.numpy() if isinstance(fresh_var, tf.Variable) else fresh_var

        # Ensure the shapes match
        if existing_values.shape != fresh_values.shape:
            raise ValueError(f"Shape mismatch: {existing_values.shape} vs {fresh_values.shape}")

        # Perform element-wise averaging
        aggregated_values = (existing_values + fresh_values) / 2

        # Always return tf.Variable for consistency
        aggregated_weights.append(tf.Variable(aggregated_values))

    return aggregated_weights



def serialize_weights(weights):
    # Convert weights to numpy arrays for serialization
    serializable_weights = [weight.numpy() if isinstance(weight, tf.Variable) else weight for weight in weights]
    return gzip.compress(pickle.dumps(serializable_weights))  # Serialize and compress

def deserialize_weights(weights):
    # Deserialize and convert back to TensorFlow Variables
    try:
        weights = gzip.decompress(weights)
    except gzip.BadGzipFile as e:
        pass
        #print ("parameter server returned weights are decompressed by fastAPI")
    deserialized_weights = pickle.loads(weights)
    return [tf.Variable(weight) for weight in deserialized_weights]  # Convert to tf.Variable


def normalize_weights(weights):
    """
    Ensure all weights are numpy arrays for aggregation.
    """
    return [w.numpy() if isinstance(w, tf.Variable) else np.array(w) for w in weights]
