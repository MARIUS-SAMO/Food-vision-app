from torchvision import transforms
from typing import Tuple


def create_img_train_transform(size: Tuple[int, int]) -> transforms.Compose:
    """
    Creates a data transformation pipeline for training images.

    Args:
        size (Tuple[int, int]): A tuple specifying the desired size of the transformed images.

    Returns:
        data_transform: A torchvision.transforms.Compose object representing the transformation pipeline.
    """
    data_transform = transforms.Compose([
        transforms.Resize(size=size),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor()
    ])

    return data_transform


def create_img_test_transform(size: Tuple[int, int]) -> transforms.Compose:
    """
    Creates a data transformation pipeline for the testing images 

    Args:
        size (Tuple[int, int]): A tuple specifying the desired size of the transformed images.

    Returns:
        data_transform: A torchvision.transforms.Compose object representing the transformation pipeline.
    """

    data_transform = transforms.Compose(
        [
            transforms.Resize(size=size),
            transforms.ToTensor()
        ]
    )

    return data_transform
