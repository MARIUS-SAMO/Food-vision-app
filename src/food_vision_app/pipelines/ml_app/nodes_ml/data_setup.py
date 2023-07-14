
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def create_dataloaders(
        train_dir: str,
        test_dir: str,
        train_transform: transforms.Compose,
        test_tranform: transforms.Compose,
        batch_size: int,
        num_workers: int = 0
):
    """Creating training and testing dataloaders

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
        train_dir (str): The directory of your training images
        test_dir (str): The directory of your testing images
        train_transform (transforms.Compose): The transformation to apply in your raw training images
        test_tranform (transforms.Compose): The transformation to apply in your raw testing images
        batch_size (int): Number of samples per batch in the dataloader
        num_workers (int): The number of workers for the dataloader

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names)
    """

    # Use ImageFolder to create datasets
    train_data = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform,
    )
    test_data = datasets.ImageFolder(
        root=test_dir,
        transform=test_tranform
    )

    # Get class names labels
    class_dict = train_data.class_to_idx
    id_to_class = {x: y for y, x in class_dict.items()}

    # Buil training and testing dataloaders
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        num_workers=num_workers
    )

    return train_loader, test_loader, id_to_class
