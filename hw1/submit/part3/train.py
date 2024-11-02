from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Literal

import medmnist
import torch
import wandb
from torch import nn
from torchvision import models
from torchvision.transforms import v2
from tqdm import tqdm

if TYPE_CHECKING:
    import numpy.typing as npt


@functools.cache
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_transform(augmentation: str = "") -> v2.Transform:
    match augmentation:
        case "auto":
            return v2.Compose(
                [v2.AutoAugment(), v2.ToTensor(), v2.Normalize(mean=[0.5], std=[0.5])]
            )
        case _:
            return v2.Compose([v2.ToTensor(), v2.Normalize(mean=[0.5], std=[0.5])])


@functools.cache
def load_dataset(split: Literal["train", "val", "test"]) -> medmnist.PathMNIST:
    config = wandb.config
    dataset: medmnist.PathMNIST = medmnist.PathMNIST(
        split=split,
        transform=load_transform(config["augmentation"]),
        download=True,
        size=64,
    )
    return dataset


@functools.cache
def load_evaluator(split: Literal["val", "test"]) -> medmnist.Evaluator:
    evaluator = medmnist.Evaluator(flag="pathmnist", split=split, size=64)
    return evaluator


def init_model(n_labels: int) -> nn.Module:
    config = wandb.config
    model: nn.Module = models.get_model(config["model_name"], num_classes=n_labels)
    model = model.to(device=get_device())
    return model


def train(
    model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module
) -> float:
    config = wandb.config
    train_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        load_dataset("train"),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=10,
    )

    model.train()
    total_loss: float = 0.0
    bar = tqdm(train_loader, desc="train")
    for inputs_, labels_ in bar:
        inputs: torch.Tensor = torch.as_tensor(inputs_, device=get_device())
        labels: torch.Tensor = torch.as_tensor(labels_, device=get_device())
        labels = labels.squeeze()
        optimizer.zero_grad()
        outputs: torch.Tensor = model(inputs)
        loss: torch.Tensor = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.shape[0]
        bar.set_postfix(loss=loss.item())
    epoch_loss: float = total_loss / len(train_loader.dataset)
    return epoch_loss


def valid_or_test(
    model: nn.Module, split: Literal["val", "test"]
) -> tuple[float, float]:
    config = wandb.config
    data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        load_dataset(split),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=10,
    )

    model.eval()
    y_true: torch.Tensor = torch.tensor([])
    y_score: torch.Tensor = torch.tensor([])
    for inputs_, labels_ in tqdm(data_loader, desc=split):
        inputs: torch.Tensor = torch.as_tensor(inputs_, device=get_device())
        labels: torch.Tensor = torch.as_tensor(
            labels_, dtype=torch.float, device=get_device()
        )
        labels = labels.reshape(-1, 1)
        outputs: torch.Tensor = model(inputs)
        outputs = outputs.softmax(dim=-1)
        y_true = torch.cat((y_true, labels.to(device=y_true.device)))
        y_score = torch.cat((y_score, outputs.to(device=y_score.device)))
    y_true_np: npt.NDArray = y_true.numpy(force=True)
    y_score_np: npt.NDArray = y_score.numpy(force=True)
    evaluator: medmnist.Evaluator = load_evaluator(split)
    auc: float
    acc: float
    auc, acc = evaluator.evaluate(y_score_np)
    return auc, acc


def run() -> None:
    config = wandb.config
    model: nn.Module = init_model(len(load_dataset("train").info["label"]))
    criterion: nn.Module = nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        amsgrad=config["amsgrad"],
    )
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["n_epochs"])
    )
    best_acc: float = 0.0
    for epoch in range(config["n_epochs"]):
        train_loss: float = train(model=model, optimizer=optimizer, criterion=criterion)
        wandb.log({"train/loss": train_loss}, step=epoch)
        with torch.no_grad():
            val_auc: float
            val_acc: float
            val_auc, val_acc = valid_or_test(model=model, split="val")
        wandb.log({"validate/auc": val_auc, "validate/acc": val_acc}, step=epoch)
        lr_scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "./checkpoints/best.pt")
    with torch.no_grad():
        test_auc: float
        test_acc: float
        test_auc, test_acc = valid_or_test(model=model, split="test")
    wandb.log({"test/auc": test_auc, "test/acc": test_acc})


def main() -> None:
    wandb.init(
        project="hw1",
        config={
            "amsgrad": False,
            "augmentation": "auto",
            "batch_size": 64,
            "lr": 0.001,
            "model_name": "mobilenet_v3_large",
            "n_epochs": 20,
            "weight_decay": 0,
        },
    )
    run()


if __name__ == "__main__":
    main()
