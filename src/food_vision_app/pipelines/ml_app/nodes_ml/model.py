from torch import nn
import torch


class TinyVGG(nn.Module):
    """Create the TinyVGG architecture.

    This class replicate the TinyVGG computer vision model

    Args:
        input_dim (int): the number of input channel
        hidden_dim (int): the number of hidden channel for the features map
        output_dim (int): the number of output units
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.clf = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16*16*hidden_dim, out_features=output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(self.conv_block_1(x).shape)
        # print(self.conv_block_2(self.conv_block_1(x)).shape)
        return self.clf(self.conv_block_2(self.conv_block_1(x)))


def build_model(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Module:
    """
    Creates a TinyVGG model.

    Args:
        input_dim: The input dimension of the model.
        hidden_dim: The hidden dimension of the model.
        output_dim: The output dimension of the model.
        device: The target device to compute on (e.g., "cuda" or "cpu").

    Returns:
        model: A PyTorch model instance.
    """

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model = TinyVGG(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    ).to(device)

    return model
