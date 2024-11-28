import os
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
import pooch
import seaborn as sns
import toolkit as tk
import torch
import wandb
from jaxtyping import Float
from torch import nn

type Action = Float[torch.Tensor, "action_size=3"]
type BAction = Float[torch.Tensor, "batch_size action_size=3"]
type BH = Float[torch.Tensor, "batch_size hidden_size"]
type BI = Float[torch.Tensor, "batch_size input_size=3*32*32+3"]
type BO = Float[torch.Tensor, "batch_size output_size=3*32*32"]
type BSAction = Float[torch.Tensor, "batch_size seq_len action_size=3"]
type BSState = Float[
    torch.Tensor, "batch_size seq_len img_channel=3 img_width=32 img_height=32"
]
type BState = Float[torch.Tensor, "batch_size img_channel=3 img_width=32 img_height=32"]
type F1BH = Float[torch.Tensor, "1 batch_size hidden_size"]
type FloatScalar = Float[torch.Tensor, ""]
type SAction = Float[torch.Tensor, "seq_len action_size=3"]
type SState = Float[torch.Tensor, "seq_len img_channel=3 img_width=32 img_height=32"]
type State = Float[torch.Tensor, "img_channel=3 img_width=32 img_height=32"]


type StateAction = tuple[
    Float[npt.NDArray, "img_channel=3 img_width=32 img_height=32"],
    Float[npt.NDArray, "action_size=3"],
    float,
]
type DeviceLikeType = str | torch.device | int


class WorldModelDataLoader:
    batch_size: int
    device: torch.device
    seq_len: int
    current_data: list[list[StateAction]]
    data: list[list[StateAction]]
    test_data: list[list[StateAction]]
    train_data: list[list[StateAction]]
    valid_data: list[list[StateAction]]

    def __init__(self, batch_size: int, seq_len: int, device: DeviceLikeType) -> None:
        data_fpath = Path(
            pooch.retrieve(
                "https://cloud.tsinghua.edu.cn/f/18096a7cee674e76922c/?dl=1",
                known_hash="sha256:2060fbd5be6ce8c8d1d8d3f976c28eedcd293d8d8cc48f71c5c98f98c00a1f2d",
                fname="car_racing_data_32x32_120.pkl",
                progressbar=True,
            )
        )
        with data_fpath.open("rb") as fp:
            self.data = pickle.load(fp)  # noqa: S301

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = torch.device(device)

        # 拆分数据为 train, valid, test 集合
        split_train = int(0.8 * len(self.data))
        split_valid = int(0.1 * len(self.data))
        self.train_data = self.data[:split_train]
        self.valid_data = self.data[split_train : split_train + split_valid]
        self.test_data = self.data[split_train + split_valid :]

    def set_train(self) -> None:
        self.current_data = self.train_data
        self.index = 0
        self.sub_index = 0  # 子序列的起始索引

    def set_valid(self) -> None:
        self.current_data = self.valid_data
        self.index = 0
        self.sub_index = 0

    def set_test(self) -> None:
        self.current_data = self.test_data
        self.index = 0
        self.sub_index = 0

    def get_batch(
        self,
    ) -> tuple[BSState, BSAction, bool]:
        states_list: list[SState] = []
        actions_list: list[SAction] = []
        batch_data: list[list[StateAction]] = self.current_data[
            self.index : self.index + self.batch_size
        ]

        for sequence in batch_data:
            state_seq: list[State] = [
                torch.tensor(step[0])
                for step in sequence[self.sub_index : self.sub_index + self.seq_len]
            ]
            action_seq: list[Action] = [
                torch.tensor(step[1])
                for step in sequence[self.sub_index : self.sub_index + self.seq_len]
            ]
            if len(state_seq) < self.seq_len:
                pad_len: int = self.seq_len - len(state_seq)
                state_seq += [torch.zeros_like(state_seq[0])] * pad_len
                action_seq += [torch.zeros_like(action_seq[0])] * pad_len

            states_list.append(torch.stack(state_seq))
            actions_list.append(torch.stack(action_seq))

        self.sub_index += self.seq_len
        if self.sub_index >= len(self.current_data[self.index]):
            self.index += self.batch_size
            self.sub_index = 0
        states: BSState = torch.stack(states_list).to(self.device)
        actions: BSAction = torch.stack(actions_list).to(self.device)

        end_flag: bool = self.index >= len(self.current_data)

        return states, actions, end_flag

    @property
    def state_size(self) -> int:
        return self.data[0][0][0].size

    @property
    def action_size(self) -> int:
        return self.data[0][0][1].size


class WorldModel(nn.Module):
    lstm: nn.LSTM
    decoder: nn.Linear

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(
        self, state: BState, action: BAction, hidden: tuple[F1BH, F1BH]
    ) -> tuple[BO, tuple[F1BH, F1BH]]:
        """Forward pass for the WorldModel.

        Args:
            state: Tensor of shape [batch_size, 3, 32, 32] (current RGB image at this time step).
            action: Tensor of shape [batch_size, action_size] (3-dimensional action vector).
            hidden: Tuple of hidden states for LSTM, (h_t, c_t), each of shape [1, batch_size, hidden_size].

        Returns:
            next_state_pred: Tensor of shape [batch_size, output_size] (flattened next state prediction, 3*32*32).
            hidden: Updated hidden state tuple (h_t, c_t) for the LSTM.
        """
        # DONE: Implement your model here. You should implement two models, one using built-in lstm layers and one implemented by yourself.
        batch_size: int = state.shape[0]
        state_size: int = state.numel() // batch_size
        action_size: int = action.numel() // batch_size
        input_size: int = state_size + action_size
        input_tensor: BI = torch.hstack(
            [state.view(batch_size, state_size), action.view(batch_size, action_size)]
        )
        features: F1BH
        features, hidden = self.lstm(
            input_tensor.view(1, batch_size, input_size), hidden
        )
        hidden_size: int = hidden[0].shape[-1]
        features: BH = features.view(batch_size, hidden_size)
        next_state_pred: BO = self.decoder(features)
        return next_state_pred, hidden


def save_img(fpath: str | os.PathLike[str], img: BState) -> None:
    fpath = Path(fpath)
    img: Float[npt.NDArray, "img_width=32 img_height=32 img_channel=3"] = (
        img.view(3, 32, 32).permute(1, 2, 0).clamp(0, 1).numpy(force=True)
    )
    fpath.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(fpath, img)


def save_states(fpath: str | os.PathLike[str], states: BSState) -> None:
    fpath = Path(fpath)
    seq_len: int = states.shape[1]
    for t in range(seq_len):
        save_img(fpath / f"{t}.png", states[:, t])


def loss_over_step(
    fpath: str | os.PathLike[str],
    ground_truth: BSState,
    prediction: BSState,
    criterion: nn.Module,
) -> Float[npt.NDArray, " seq_len"]:
    fpath = Path(fpath)
    seq_len: int = ground_truth.shape[1]
    loss_over_step: list[float] = [
        criterion(prediction[:, i], ground_truth[:, i]).item() for i in range(seq_len)
    ]
    return np.asarray(loss_over_step)


def main() -> None:
    tk.logging.init()
    wandb.init(
        config={
            "batch_size": 1,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "hidden_size": 128,
            "seq_len": 150,
        }
    )
    mpl.rcParams["figure.dpi"] = 300
    batch_size: int = wandb.config["batch_size"]
    device: DeviceLikeType = wandb.config["device"]
    hidden_size: int = wandb.config["hidden_size"]
    seq_len: int = wandb.config["seq_len"]
    data_loader: WorldModelDataLoader = WorldModelDataLoader(
        batch_size=wandb.config["batch_size"],
        seq_len=wandb.config["seq_len"],
        device=wandb.config["device"],
    )
    model: WorldModel = WorldModel(
        input_size=data_loader.state_size + data_loader.action_size,
        hidden_size=wandb.config["hidden_size"],
        output_size=data_loader.state_size,
    ).to(wandb.config["device"])
    criterion: nn.MSELoss = nn.MSELoss()
    model.load_state_dict(torch.load("world_model_best.pth", weights_only=True))
    model.eval()
    data_loader.set_test()
    states: BSState
    actions: BSAction
    states, actions, _end_flag = data_loader.get_batch()
    # Ground Truth
    save_states("fig/task3/ground-truth", states)
    # Teacher Forcing
    teacher_forcing_states: BSState = states.clone()
    hidden: tuple[F1BH, F1BH] = (
        torch.zeros(1, batch_size, hidden_size).to(device),
        torch.zeros(1, batch_size, hidden_size).to(device),
    )
    for t in range(seq_len - 1):
        current_state: BState = states[:, t]
        action: BAction = actions[:, t]
        next_state_pred: BO
        next_state_pred, hidden = model(current_state, action, hidden)
        next_state_pred: BState = next_state_pred.view(batch_size, 3, 32, 32)
        teacher_forcing_states[:, t + 1] = next_state_pred
    save_states("fig/task3/teacher-forcing", teacher_forcing_states)
    # Autoregressive Rollout
    autoregressive_rollout_states: BSState = states.clone()
    hidden: tuple[F1BH, F1BH] = (
        torch.zeros(1, batch_size, hidden_size).to(device),
        torch.zeros(1, batch_size, hidden_size).to(device),
    )
    next_state_pred: BState = states[:, 0]
    for t in range(seq_len - 1):
        current_state: BState = next_state_pred
        action: BAction = actions[:, t]
        next_state_pred: BO
        next_state_pred, hidden = model(current_state, action, hidden)
        next_state_pred: BState = next_state_pred.view(batch_size, 3, 32, 32)
        autoregressive_rollout_states[:, t + 1] = next_state_pred
    save_states("fig/task3/autoregressive-rollout", autoregressive_rollout_states)
    # Loss
    teacher_forcing_loss: Float[npt.NDArray, " seq_len"] = loss_over_step(
        "fig/task3/teacher-forcing-loss",
        states,
        teacher_forcing_states,
        criterion,
    )
    autoregressive_rollout_loss: Float[npt.NDArray, " seq_len"] = loss_over_step(
        "fig/task3/autoregressive-rollout-loss",
        states,
        autoregressive_rollout_states,
        criterion,
    )
    loss_dataframe: pl.DataFrame = pl.concat(
        [
            pl.DataFrame(
                {
                    "step": range(seq_len),
                    "loss": teacher_forcing_loss,
                    "strategy": ["Teacher Forcing"] * seq_len,
                }
            ),
            pl.DataFrame(
                {
                    "step": range(seq_len),
                    "loss": autoregressive_rollout_loss,
                    "strategy": ["Autoregressive Rollout"] * seq_len,
                }
            ),
        ]
    )
    plt.figure()
    sns.lineplot(loss_dataframe, x="step", y="loss", hue="strategy")
    plt.tight_layout()
    plt.savefig("fig/task3/loss.png")


if __name__ == "__main__":
    main()
