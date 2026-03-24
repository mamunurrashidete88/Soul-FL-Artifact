import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


# ---------------------------------------------------------------------------
# ResNet-18 for CIFAR-10 
# ---------------------------------------------------------------------------
class CIFARResNet18(nn.Module):

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.model = resnet18(weights=None, num_classes=num_classes)
        # Replace stem for small images
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.maxpool = nn.Identity()  # type: ignore[assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# CNN for FEMNIST
# ---------------------------------------------------------------------------
class FEMNISTNet(nn.Module):

    def __init__(self, num_classes: int = 62):
        super().__init__()
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # → 32×14×14

            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # → 64×7×7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_model(dataset: str, num_classes: int) -> nn.Module:
    if dataset == "cifar10":
        return CIFARResNet18(num_classes=num_classes)
    elif dataset == "femnist":
        return FEMNISTNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Gradient utilities
# ---------------------------------------------------------------------------
def get_flat_gradients(model: nn.Module) -> torch.Tensor:
    """Flatten all parameter gradients into a single vector."""
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().view(-1))
        else:
            grads.append(torch.zeros(p.numel(), device=next(model.parameters()).device))
    return torch.cat(grads)


def get_flat_params(model: nn.Module) -> torch.Tensor:
    """Flatten all parameters into a single vector."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model: nn.Module, flat: torch.Tensor) -> None:
    """Load a flat parameter vector back into the model."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat[offset: offset + numel].view_as(p))
        offset += numel


def compute_gradient_update(
    global_params: torch.Tensor,
    local_params: torch.Tensor,
) -> torch.Tensor:

    return local_params - global_params


def model_size_mb(model: nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
