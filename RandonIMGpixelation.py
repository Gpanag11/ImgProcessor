import os
import numpy as np
import glob
from PIL import Image
from Grayscale import to_grayscale
from Pixelate import prepare_image
import random
import torch


class RandomImagePixelationDataset:
    def __init__(self, image_dir, width_range, height_range, size_range):
        # Add checks for the range tuples
        for i in [width_range, height_range, size_range]:
            if i[0] < 2:
                raise ValueError("Minimum value of range must be at least 2.")
            if i[0] > i[1]:
                raise ValueError("Minimum value of range must not be greater than the maximum.")

        self.image_dir = image_dir
        self.width_range = width_range
        self.height_range = height_range
        self.size_range = size_range

        super(RandomImagePixelationDataset, self).__init__()

        self.image_paths = self.load_image_paths()

    def load_image_paths(self):
        """
        Load the paths of all images in the specified directory

        Returns:
        a list, A sorted list of full paths to each image in my directory
        """
        images_directory = os.path.abspath(self.image_dir)
        images = glob.glob(os.path.join(images_directory, '**', '*.jpg'), recursive=True)
        images.sort()
        return images

    def __getitem__(self, index):
        """
        Applies random pixelation to the image at the given index

        Parameters:
        index (int): Index of the image to be pixelated

        Returns:
        tuple: A tuple containing the pixelated image, an array indicating the known (unpixelated) areas,
               a copy of the target area, and the path of the original image
        """
        random.seed(index)

        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image_width, image_height = image.size

        np_image = np.array(image)

        grayscale_image = to_grayscale(np_image)

        # Randomly choose width and height, and clip if necessary
        width = random.randint(self.width_range[0], self.width_range[1])
        height = random.randint(self.height_range[0], self.height_range[1])
        width = min(width, image_width)
        height = min(height, image_height)

        # Randomly choose x and y within the valid range
        x = random.randint(0, image_width - width)
        y = random.randint(0, image_height - height)

        # Random size
        size = random.randint(self.size_range[0], self.size_range[1])

        # Call the prepare_image method
        pixelated_image, known_array, target_array = prepare_image(grayscale_image, x, y, width, height, size)

        return pixelated_image, known_array, target_array, image_path

    def __len__(self):
        """
        Returns the total number of images in the dataset
        """
        return len(self.image_paths)


def stack_with_padding(batch_as_list: list):
    """
    This function takes a batch of images, represented as a list of tuples. Each tuple contains a pixelated image and
    an array indicating known unpixelated areas, a target array, and an images file path. This function stacks these
    images into a single tensor, uses paddling them as necessary to match the largest image dimensions in the batch. The
    arrays are also stacked and the target arrays are converted to PyTorch tensors.

    The Parameters:
    batch_as_list (list): A list of tuples, in which each tuple contains a pixelated image (as a numpy array),
                          a known array (as a numpy array), a target array (as a numpy array), andd an image file path

    Returns:
    tuple: A tuple containing four elements:
           1. A PyTorch tensor of the stacked pixelated images, with padding as necessary.
           2. A PyTorch tensor of the stacked known arrays, with padding as necessary.
           3. A list of PyTorch tensors of the target arrays.
           4. A list of image file paths corresponding to each image in the batch.

    """
    n = len(batch_as_list)
    pixelated_images_dtype = batch_as_list[0][0].dtype
    known_arrays_dtype = batch_as_list[0][1].dtype
    shapes = []
    pixelated_images = []
    known_arrays = []
    target_arrays = []
    image_files = []

    for pixelated_image, known_array, target_array, image_file in batch_as_list:
        shapes.append(pixelated_image.shape)  # Equal to the shape of known array
        pixelated_images.append(pixelated_image)
        known_arrays.append(known_array)
        target_arrays.append(torch.from_numpy(target_array))
        image_files.append(image_file)

    max_shape = np.max(np.stack(shapes, axis=0), axis=0)
    stacked_pixelated_images = np.zeros(shape=(n, *max_shape), dtype=pixelated_images_dtype)
    stacked_known_arrays = np.ones(shape=(n, *max_shape), dtype=known_arrays_dtype)

    for i in range(n):
        channels, height, width = pixelated_images[i].shape
        stacked_pixelated_images[i, :channels, :height, :width] = pixelated_images[i]
        stacked_known_arrays[i, :channels, :height, :width] = known_arrays[i]

    return torch.from_numpy(stacked_pixelated_images), torch.from_numpy(
        stacked_known_arrays), target_arrays, image_files
