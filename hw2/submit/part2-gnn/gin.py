from typing import Protocol, runtime_checkable

import beartype
import toolkit as tk
import torch
import torch.nn.functional as F
import torch_geometric.nn as tgnn
import torchmetrics as tm
import wandb
from jaxtyping import Float, jaxtyped
from loguru import logger
from torch import nn, optim
from torch_geometric.data import Dataset
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from tqdm.rich import tqdm

type DeviceLikeType = str | torch.device | int


@runtime_checkable
class BatchData(Protocol):
    @property
    def batch(self) -> Float[torch.Tensor, " num_nodes"]: ...
    @property
    def batch_size(self) -> int: ...
    @property
    def edge_index(self) -> Float[torch.Tensor, "2 num_edges"]: ...
    @property
    def num_edges(self) -> int: ...
    @property
    def num_features(self) -> int: ...
    @property
    def num_graphs(self) -> int: ...
    @property
    def num_node_features(self) -> int: ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def pos(self) -> Float[torch.Tensor, "num_nodes 3"]: ...
    @property
    def x(self) -> Float[torch.Tensor, "num_nodes num_node_features"]: ...
    @property
    def y(self) -> Float[torch.Tensor, "batch_size num_targets"]: ...


DIPOLE_INDEX = 0


def load_dataset(path: str = "./data/QM9") -> Dataset:
    return QM9(path)


def split_dataset(dataset: Dataset) -> tuple[DataLoader, DataLoader, DataLoader]:
    batch_size: int = wandb.config["batch_size"]

    train_dataset: Dataset = dataset[:10000]  # pyright: ignore [reportAssignmentType]
    val_dataset: Dataset = dataset[10000:11000]  # pyright: ignore [reportAssignmentType]
    test_dataset: Dataset = dataset[11000:12000]  # pyright: ignore [reportAssignmentType]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


class GIN(nn.Module):
    def __init__(self, num_node_features: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.gin = tgnn.GIN(
            in_channels=num_node_features + 3, hidden_channels=hidden_dim, num_layers=7
        )
        self.lin = nn.Linear(hidden_dim, 1)

    @jaxtyped(typechecker=beartype.beartype)
    def forward(self, data: BatchData) -> Float[torch.Tensor, "batch_size 1"]:
        x: Float[torch.Tensor, "num_nodes in_channels=num_node_features+3"] = (
            torch.hstack([data.x, data.pos])
        )
        edge_index: Float[torch.Tensor, "2 num_edges"] = data.edge_index
        batch: Float[torch.Tensor, " num_nodes"] = data.batch
        x: Float[torch.Tensor, "num_nodes hidden_dim"] = self.gin(x, edge_index)
        x: Float[torch.Tensor, "batch_size hidden_dim"] = tgnn.global_mean_pool(
            x, batch
        )  # pyright: ignore [reportCallIssue]
        x: Float[torch.Tensor, "batch_size 1"] = self.lin(x)
        return x


def train(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer) -> float:
    device: DeviceLikeType = wandb.config["device"]
    model.train()
    total_loss = 0.0
    for data_ in loader:
        data: BatchData = data_.to(device)
        optimizer.zero_grad()
        y_pred: Float[torch.Tensor, "batch_size 1"] = model(data)
        y_true: Float[torch.Tensor, "batch_size 1"] = data.y[:, DIPOLE_INDEX].view(
            data.batch_size, 1
        )
        loss: Float[torch.Tensor, ""] = F.mse_loss(y_pred, y_true)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)  # pyright: ignore [reportArgumentType]


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    device: DeviceLikeType = wandb.config["device"]
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data_ in loader:
            data: BatchData = data_.to(device)
            y_pred: Float[torch.Tensor, "batch_size 1"] = model(data)
            y_true: Float[torch.Tensor, "batch_size 1"] = data.y[:, DIPOLE_INDEX].view(
                data.batch_size, 1
            )
            loss: Float[torch.Tensor, ""] = F.mse_loss(y_pred, y_true)
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)  # pyright: ignore [reportArgumentType]


def compute_r2(model: nn.Module, loader: DataLoader) -> float:
    device: DeviceLikeType = wandb.config["device"]
    model.eval()
    y_pred_list: list[Float[torch.Tensor, "batch_size 1"]] = []
    y_true_list: list[Float[torch.Tensor, "batch_size 1"]] = []
    with torch.no_grad():
        for data_ in loader:
            data: BatchData = data_.to(device)
            y_pred: Float[torch.Tensor, "batch_size 1"] = model(data)
            y_true: Float[torch.Tensor, "batch_size 1"] = data.y[:, DIPOLE_INDEX].view(
                data.batch_size, 1
            )
            y_pred_list.append(y_pred)
            y_true_list.append(y_true)
    y_pred: Float[torch.Tensor, "num_samples 1"] = torch.vstack(y_pred_list)
    y_true: Float[torch.Tensor, "num_samples 1"] = torch.vstack(y_true_list)
    r2_score: Float[torch.Tensor, ""] = tm.functional.r2_score(y_pred, y_true)
    return r2_score.item()


def init() -> None:
    tk.logging.init()
    wandb.init(
        config={
            "batch_size": 32,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "hidden_dim": 128,
            "lr": 0.001,
            "n_epochs": 50,
        },
        group="GIN",
    )


def main() -> None:
    init()
    dataset: Dataset = load_dataset()
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_loader, val_loader, test_loader = split_dataset(dataset)
    model: nn.Module = GIN(
        num_node_features=dataset.num_node_features,
        hidden_dim=wandb.config["hidden_dim"],
    ).to(wandb.config["device"])
    optimizer: optim.Optimizer = optim.Adam(model.parameters(), lr=wandb.config["lr"])
    n_epochs: int = wandb.config["n_epochs"]
    for epoch in tqdm(range(n_epochs)):
        train_loss: float = train(model=model, loader=train_loader, optimizer=optimizer)
        val_loss: float = evaluate(model=model, loader=val_loader)
        wandb.log({"train/loss": train_loss, "val/loss": val_loss})
        logger.info(
            f"Epoch: {epoch:>2} / {n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )
    test_loss: float = evaluate(model=model, loader=test_loader)
    wandb.summary["test/loss"] = test_loss
    r2_score: float = compute_r2(model=model, loader=test_loader)
    wandb.summary["test/r2_score"] = r2_score
    logger.info(f"Test Loss: {test_loss:.4f}, R2 Score: {r2_score:.4f}")


if __name__ == "__main__":
    main()
