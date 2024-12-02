import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_organize_images(input_dir):
    """
    Load images from the input directory and organize them into separate dictionaries
    for 'calib' and 'kodim02', preserving their variants.
    """
    calib_images = {}
    kodim02_images = {}

    # Iterate through files in the directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.tiff'):
            filepath = os.path.join(input_dir, filename)

            # Load the image as a grayscale array
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            # Parse the filename to identify the type and variant
            base_name = filename.rsplit('.', 1)[0]  # Remove extension
            parts = base_name.split('_')

            if parts[0] == 'calib':
                calib_images['_'.join(parts[1:])] = image
            elif parts[0] == 'kodim02':
                kodim02_images['_'.join(parts[1:])] = image

    return calib_images, kodim02_images


def save_images_to_npz(images_dict, output_path):
    """
    Save a dictionary of images to a single .npz file.
    """
    np.savez_compressed(output_path, **images_dict)


# Main function
if __name__ == "__main__":
    input_directory = "./server_prep/images/"  # Replace with your directory
    calib_output_path = "./server_prep/calib_images.npz"
    kodim02_output_path = "./server_prep/kodim02_images.npz"

    img = np.load(kodim02_output_path)
    mat = img['5_25']

    plt.imshow(mat)
    plt.show()

    # # Load and organize images
    # calib_images, kodim02_images = load_and_organize_images(input_directory)

    # # Save to npz files
    # save_images_to_npz(calib_images, calib_output_path)
    # save_images_to_npz(kodim02_images, kodim02_output_path)

    # print(f"Saved {len(calib_images)} calib images to {calib_output_path}")
    # print(f"Saved {len(kodim02_images)} kodim02 images to {kodim02_output_path}")
