from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Literal

import medmnist
import torch.utils.data
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


def load_transform() -> v2.Transform:
    config = wandb.config
    weights: models.WeightsEnum = models.get_weight(config["weights_name"])
    return weights.transforms()


@functools.cache
def load_dataset(split: Literal["train", "val", "test"]) -> medmnist.PathMNIST:
    dataset: medmnist.PathMNIST = medmnist.PathMNIST(
        split=split, transform=load_transform(), download=True, size=64
    )
    return dataset


@functools.cache
def load_evaluator(split: Literal["val", "test"]) -> medmnist.Evaluator:
    evaluator = medmnist.Evaluator(flag="pathmnist", split=split, size=64)
    return evaluator


def init_model(n_labels: int) -> nn.Module:
    config = wandb.config
    model: nn.Module
    match weights := config["weights_name"]:
        case "none":
            model = models.get_model(config["model_name"], num_classes=n_labels)
        case _:
            model = models.get_model(config["model_name"], weights=weights)
            for param in model.parameters():
                param.requires_grad = False
            final_layer: nn.Module = model.classifier[-1]
            model.classifier[-1] = nn.Linear(
                final_layer.in_features, n_labels, bias=True
            )
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


def init_lr_scheduler(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler:
    config = wandb.config
    main_lr_scheduler: torch.optim.lr_scheduler.LRScheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["n_epochs"] - config["lr_warmup_epochs"],
            eta_min=config["lr_min"],
        )
    )
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler
    if config["lr_warmup_epochs"] > 0:
        warmup_lr_scheduler: torch.optim.lr_scheduler.LRScheduler = (
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=config["lr_warmup_decay"],
                total_iters=config["lr_warmup_epochs"],
            )
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [warmup_lr_scheduler, main_lr_scheduler],
            milestones=[config["lr_warmup_epochs"]],
        )
    else:
        lr_scheduler = main_lr_scheduler
    return lr_scheduler


class EarlyStopping:
    best_score: float = -torch.inf
    best_state: dict[str, Any]
    counter: int = 0
    delta: float = 0.0
    early_stop: bool = False
    patience: int = 3

    def __init__(self, patience: int = 3, *, delta: float = 0.0) -> None:
        self.delta = delta
        self.patience = patience

    def __call__(self, score: float, model: nn.Module) -> None:
        if score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_state = model.state_dict()
            self.counter = 0
            self.save()

    def save(self, path: str = "./checkpoints/best.pt") -> None:
        torch.save(self.best_state, path)


def run() -> None:
    config = wandb.config
    wandb.define_metric("validate/acc", summary="max")
    model: nn.Module = init_model(len(load_dataset("train").info["label"]))
    criterion: nn.Module = nn.CrossEntropyLoss(
        label_smoothing=config["label_smoothing"]
    )
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        amsgrad=config["amsgrad"],
    )
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler = init_lr_scheduler(optimizer)
    early_stopping: EarlyStopping = EarlyStopping(
        patience=config["early_stopping_patience"], delta=config["early_stopping_delta"]
    )
    for epoch in range(config["n_epochs"]):
        train_loss: float = train(model=model, optimizer=optimizer, criterion=criterion)
        wandb.log({"train/loss": train_loss}, step=epoch)
        with torch.no_grad():
            val_auc: float
            val_acc: float
            val_auc, val_acc = valid_or_test(model=model, split="val")
        wandb.log({"validate/auc": val_auc, "validate/acc": val_acc}, step=epoch)
        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            break
        lr_scheduler.step()
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
            "augmentation": "pre-trained",
            "batch_size": 512,
            "early_stopping_delta": 0.01,
            "early_stopping_patience": 3,
            "label_smoothing": 0.1,
            "lr_min": 0.0,
            "lr_warmup_decay": 0.1,
            "lr_warmup_epochs": 2,
            "lr": 0.0019976195957370426,
            "model_name": "MobileNet_V3_Large",
            "n_epochs": 20,
            "weight_decay": 0.0000010640579453275,
            "weights_name": "MobileNet_V3_Large_Weights.IMAGENET1K_V2",
        },
    )
    run()


if __name__ == "__main__":
    main()
