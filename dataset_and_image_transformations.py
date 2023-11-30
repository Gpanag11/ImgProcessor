import os
from typing import Tuple, Union
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from collections.abc import Sized



def load_and_transform_image(image_path: str, transform=None) -> Image.Image:
    with Image.open(image_path) as img:
        return img if transform is None else transform(img)


class SimpleImageDataset(Dataset):
    """
    Theclass provides an interface for accessing images from a directory, treating each image file
    as an individual data point

    Attributes:
    file_paths (list): A list of paths to image files found in the specified directory.

    Parameters:
    root_dir (str): The root directory from which to load images
    ext (str): The file extension of the images to load (default is "*.jpg")

    Methods:
    __getitem__(idx): Retrieves the image at the specified index in the dataset
    __len__(): Returns the number of images in the dataset.
    """
    def __init__(self, root_dir: str, ext: str = "*.jpg"):
        self.file_paths = sorted(
            [os.path.join(dp, f) for dp, dn, filenames in os.walk(root_dir) for f in filenames if f.endswith(ext)])

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        image = load_and_transform_image(self.file_paths[idx])
        return image, idx

    def __len__(self) -> int:
        return len(self.file_paths)


class AugmentedImageDataset(Dataset):
    """
    This class wraps around a base dataset, applying a specified transformation to each image. It is useful
    for data augmentation purposes where transformations can include resizing, normalization, and other
    torchvision transforms.

    Attributes:
    base_dataset (Dataset): The dataset to which the transformations will be applied
    transform (callable): The transformation function to be applied to each image

    Parameters:
    base_dataset (Dataset): The underlying dataset containing the original images
    transform (callable): A function or composition of functions from torchvisiondottransforms
                          that will be applied to each image.

    Methods:
    __getitem__(idx): Retrieves and applies the transformation to the image at the specified index.
    __len__(): Returns the number of items in the base dataset
    """
    def __init__(self, base_dataset: Dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        image, _ = self.base_dataset[idx]
        return self.transform(image), idx

    def __len__(self) -> int:
        return len(self.base_dataset)


if __name__ == "__main__":
    image_dir = "images"
    image_size = 300
    base_dataset = SimpleImageDataset(image_dir)
    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    augmented_dataset = AugmentedImageDataset(base_dataset, transform)

    for i in range(len(base_dataset)):
        original_img, _ = base_dataset[i]
        augmented_img, _ = augmented_dataset[i]

        plt.subplot(1, 2, 1)
        plt.imshow(original_img)
        plt.title("Original Image")

        plt.subplot(1, 2, 2)
        plt.imshow(transforms.functional.to_pil_image(augmented_img))
        plt.title("Augmented Image")

        plt.suptitle(f"Image {i}")
        plt.tight_layout()
        plt.savefig(f"transformed_image_{i}.png")
        plt.show()
