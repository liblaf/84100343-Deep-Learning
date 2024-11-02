import functools
from typing import Literal

import captum
import matplotlib as mpl
import matplotlib.pyplot as plt
import medmnist
import numpy as np
import torch
from icecream import ic
from torchvision import models, transforms

DEVICE: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"


@functools.cache
def load_dataset(split: Literal["train", "val", "test"]) -> medmnist.PathMNIST:
    dataset: medmnist.PathMNIST = medmnist.PathMNIST(
        split=split,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
        ),
        download=True,
        size=64,
    )
    return dataset


@functools.cache
def load_evaluator(split: Literal["val", "test"]) -> medmnist.Evaluator:
    evaluator = medmnist.Evaluator(flag="pathmnist", split=split, size=64)
    return evaluator


def init_model(n_labels: int) -> models.ResNet:
    model: models.ResNet = models.resnet18(num_classes=n_labels)
    model = model.to(device=DEVICE)
    ic(model)
    return model


def normalize(img: torch.Tensor) -> torch.Tensor:
    return (img - img.min()) / (img.max() - img.min())


def main() -> None:
    mpl.rcParams.update({"figure.dpi": 300})
    dataset: medmnist.PathMNIST = load_dataset("test")
    model: models.ResNet = init_model(len(dataset.info["label"]))
    model.load_state_dict(torch.load("./checkpoints/best.pt"))
    saliency = captum.attr.Saliency(model)

    random_generator: np.random.Generator = np.random.default_rng()
    for input_idx in random_generator.choice(len(dataset), 10, replace=False):
        inputs_, _labels = dataset[input_idx]
        inputs: torch.Tensor = torch.as_tensor(inputs_, device=DEVICE).unsqueeze(0)
        labels: torch.Tensor = torch.as_tensor(_labels, device=DEVICE).unsqueeze(0)
        label: int = labels.item()
        input_fname: str = f"./fig/task2/{input_idx:05d}-input.png"
        plt.imsave(
            input_fname, normalize(inputs.squeeze().permute(1, 2, 0)).numpy(force=True)
        )
        attributions: torch.Tensor = saliency.attribute(inputs, target=label)
        attributions = attributions.squeeze()
        saliency_fname: str = f"./fig/task2/{input_idx:05d}-saliency.png"
        plt.imsave(
            saliency_fname, normalize(attributions.permute(1, 2, 0)).numpy(force=True)
        )
        print(
            f"| {input_idx:05d} | {label} | ![]({input_fname}) | ![]({saliency_fname}) |"
        )


if __name__ == "__main__":
    main()
