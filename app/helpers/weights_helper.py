import os
import tensorflow as tf
import gzip
import pickle
import numpy as np
import tempfile



def aggregate(existing, fresh):
    """
    Aggregate weights using simple averaging.

    Args:
        existing (list): List of existing weight arrays.
        fresh (list): List of fresh weight arrays.

    Returns:
        list: Aggregated weights as NumPy arrays.
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

        aggregated_weights.append(aggregated_values)  # Returning NumPy array, not tf.Variable

    return aggregated_weights


def aggregate___(existing, fresh):
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



def serialize_weights(weights, write_to_file=False):
    # Convert weights to numpy arrays for serialization
    serializable_weights = [weight.numpy() if isinstance(weight, tf.Variable) else weight for weight in weights]
    serialized_data = gzip.compress(pickle.dumps(serializable_weights))  # Serialize and compress

    if write_to_file:
        file_path = write_to_tmp_file(serialized_data)
        print(file_path)
    return serialized_data

def write_to_tmp_file(data):
    # Ensure the directory exists; create it if not
    dir_name="/media/ali/DATADRIVE1/tmp"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Create a temporary file in the specified directory
    with tempfile.NamedTemporaryFile(delete=False, dir=dir_name) as tmp_file:
        tmp_file.write(data)
        tmp_file_path = tmp_file.name  # Save the file path

    return tmp_file_path


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



#======================= GRADIENTS ===========================
def aggregate_gradients(existing, fresh):
    """
    Aggregate gradients using simple averaging.

    Args:
        existing (list): List of existing gradients as TensorFlow Tensor objects or numpy arrays.
        fresh (list): List of fresh gradients as TensorFlow Tensor objects or numpy arrays.
m
    Returns:
        list: Aggregated gradients as TensorFlow Tensors.
    """
    if len(existing) != len(fresh):
        raise ValueError("The length of existing and fresh gradients must match.")

    aggregated_gradients = []
    for existing_grad, fresh_grad in zip(existing, fresh):
        # Convert to numpy if needed
        existing_values = existing_grad.numpy() if isinstance(existing_grad, tf.Tensor) else existing_grad
        fresh_values = fresh_grad.numpy() if isinstance(fresh_grad, tf.Tensor) else fresh_grad

        # Ensure the shapes match
        if existing_values.shape != fresh_values.shape:
            raise ValueError(f"Shape mismatch: {existing_values.shape} vs {fresh_values.shape}")

        # Perform element-wise averaging
        aggregated_values = (existing_values + fresh_values) / 2

        # Always return tf.Tensor for consistency
        aggregated_gradients.append(tf.convert_to_tensor(aggregated_values))

    return aggregated_gradients



def sparsify_gradients(gradients):
    """
    Sparsify gradients by extracting non-zero values and their indices.
    Args:
        gradients (list): List of gradient tensors.
    Returns:
        list: A list of tuples (indices, values, shape) for each gradient.
    """

    serialize_weights(gradients, True)


    sparse_gradients = []

    for grad in gradients:
        # Ensure the gradient is a numpy array
        grad_array = grad.numpy() if isinstance(grad, tf.Tensor) else grad

        # Get non-zero indices and values
        indices = np.nonzero(grad_array)
        values = grad_array[indices]
        shape = grad_array.shape

        # Store as a tuple
        sparse_gradients.append((indices, values, shape))

    quantized = quantize_gradients(sparse_gradients, np.int8)

    serialize_weights(quantized, True)
    return sparse_gradients




def quantize_gradients(gradients, dtype=np.float16):
    """
    Quantize the gradients to reduce communication size.

    Args:
        gradients (list of np.ndarray): List of gradients.
        dtype: Desired data type for quantization (e.g., np.float16 or np.int8).

    Returns:
        list of np.ndarray: Quantized gradients.
    """
    quantized_gradients = []
    for grad in gradients:
        if grad is not None:
            # Convert to the desired dtype
            quantized_gradients.append(grad.astype(dtype))
        else:
            quantized_gradients.append(None)
    return quantized_gradients

def dequantize_gradients(quantized_gradients, original_dtype=np.float32):
    """
    Convert quantized gradients back to the original precision.

    Args:
        quantized_gradients (list of np.ndarray): List of quantized gradients.
        original_dtype: Original precision of gradients (e.g., np.float32).

    Returns:
        list of np.ndarray: Dequantized gradients.
    """
    dequantized_gradients = []
    for grad in quantized_gradients:
        if grad is not None:
            # Convert back to the original dtype
            dequantized_gradients.append(grad.astype(original_dtype))
        else:
            dequantized_gradients.append(None)
    return dequantized_gradients