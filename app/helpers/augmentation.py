import json
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import array_to_img


def augment_tb_images(image_directory, output_directory, num_augmented_images=2800):
    """
    Generates augmented images from the provided directory of TB images.

    Args:
    - image_directory (str): Path to the directory containing the original TB images.
    - output_directory (str): Path where the augmented images will be saved.
    - num_augmented_images (int): Total number of augmented images to generate. Defaults to 2800.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Initialize the ImageDataGenerator with augmentation parameters
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Get all the images from the provided directory
    image_files = [f for f in os.listdir(image_directory) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Loop through each image and apply augmentation
    for i, image_file in enumerate(image_files):
        # Load the image
        img_path = os.path.join(image_directory, image_file)
        img = load_img(img_path)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)  # Add batch dimension

        # Generate augmented images and save them
        j = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=output_directory, save_prefix='aug__', save_format='jpeg'):
            j += 1
            if j >= num_augmented_images // len(image_files):
                break

        # If the number of required augmented images is reached, break out of the loop
        if (i + 1) * (num_augmented_images // len(image_files)) >= num_augmented_images:
            break

    print(f"Generated {num_augmented_images} augmented images in '{output_directory}'.")


def generate_image_json(directory_path, url_prefix):
    """
    Generate a JSON object containing URLs and labels for all images in a directory.

    Args:
    - directory_path (str): Path to the root directory containing image subdirectories.
    - url_prefix (str): The URL prefix to be used in the final image URLs.

    Returns:
    - A JSON object containing URLs and labels for all images.
    """
    images = []

    # Iterate through subdirectories (labels) and their files (images)
    for label in os.listdir(directory_path):
        label_path = os.path.join(directory_path, label)

        # Check if it is a directory (this will be the label, e.g., "Normal", "Tuberculosis")
        if os.path.isdir(label_path):
            for image_filename in os.listdir(label_path):
                # Only consider image files (e.g., .png, .jpg, .jpeg)
                if image_filename.endswith(('.png', '.jpg', '.jpeg')):
                    image_url = f"{url_prefix}/{label}/{image_filename}"
                    label_value = 0 if label.lower() == 'normal' else 1  # Assume 0 for 'Normal' and 1 for 'Tuberculosis'
                    images.append({"url": image_url, "label": label_value})

    # Create the final JSON structure
    result = {"images": images}

    # Print the JSON object
    return result # print(json.dumps(result, indent=4))


# Example usage
#directory_path = '/media/ali/workspace/dataset/tb_dataset_tiny'
#url_prefix = 'http://localhost:9000/datasets/tb_dataset_tiny'
#generate_image_json(directory_path, url_prefix)


# Example usage:
# augment_tb_images('/path/to/tb_images', '/path/to/augmented_tb_images', num_augmented_images=2800)
