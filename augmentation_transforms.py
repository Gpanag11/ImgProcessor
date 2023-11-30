from collections.abc import Sequence
from typing import Union
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt


def create_augmentation_transforms(image_size: Union[int, Sequence[int]], seed: int):
    """
    Create a chain of randomly selected image augmentation transforms,
    a series of potential image transformations for augmentation, including random rotation,
    vertical flip, horizontal flip, and color jitter. It then randomly selects two of these transformations, based
    on the provided seed, to include in the transform chain.

    Parameters:
    image_size (Union[int, Sequence[int]]): The target size for resizing the images. Could be a single integer
                                            for square images, or a sequence of two integers with (width, height=)

    Returns:
    transforms.Compose: A composition of transforms including resizing, two randomly selected augmentations
    (from rotation, vertical flip, horizontal flip, color jitter), conversion to tensor,
    and dropout. This chain can be applied to images for augmentation


    """
    rng = np.random.default_rng(seed)
    torch.random.manual_seed(seed)

    image_transforms = [
        transforms.RandomRotation(degrees=180),
        transforms.RandomVerticalFlip(p=1),
        transforms.RandomHorizontalFlip(p=1),
        transforms.ColorJitter()
    ]

    rng.shuffle(image_transforms)
    selected_transforms = image_transforms[:2]

    transform_list = [transforms.Resize(size=image_size)] + \
                     selected_transforms + \
                     [transforms.ToTensor(), torch.nn.Dropout(p=0.01)]

    return transforms.Compose(transform_list)


def random_augmented_image(image: Image, transform_chain):
    return transform_chain(image)


def plot_images(original, transformed, save_path="../sheet/figures/2023_a6_ex1.pdf"):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[1].imshow(transforms.functional.to_pil_image(transformed))
    axes[1].set_title("Transformed Image")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    image_path = "images/test_image.jpg"
    image_size = 300
    seed = 3

    with Image.open(image_path) as image:
        transform_chain = create_augmentation_transforms(image_size, seed)
        transformed_image = random_augmented_image(image, transform_chain)
        plot_images(image, transformed_image)
