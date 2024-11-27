import pickle
from pathlib import Path

import numpy.typing as npt
import pooch
import toolkit as tk
import torch
import wandb
from jaxtyping import Float
from loguru import logger
from torch import nn, optim

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
        self.decoder = nn.Linear(self.lstm.hidden_size, output_size)

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


def train(
    data_loader: WorldModelDataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    *,
    n_epochs: int = 50,
) -> None:
    device: DeviceLikeType = wandb.config["device"]
    hidden_size: int = wandb.config["hidden_size"]
    seq_len: int = wandb.config["seq_len"]

    best_val_loss: float = torch.inf

    for epoch in range(n_epochs):
        model.train()
        data_loader.set_train()
        total_train_loss: float = 0.0
        total_train_samples: int = 0

        while True:
            states: BSState
            actions: BSAction
            end_flag: bool
            states, actions, end_flag = data_loader.get_batch()
            batch_size: int = states.size(0)
            # Initialize the hidden state (h_0, c_0) for the LSTM, resetting it for each new batch
            hidden: tuple[F1BH, F1BH] = (
                torch.zeros(1, batch_size, hidden_size).to(device),
                torch.zeros(1, batch_size, hidden_size).to(device),
            )
            # Loop through each time step in the sequence
            for t in range(seq_len - 1):
                current_state: BState = states[:, t]
                action: BAction = actions[:, t]
                next_state: BO = states[:, t + 1].view(batch_size, -1)

                next_state_pred: BO
                next_state_pred, hidden = model(current_state, action, hidden)
                loss: FloatScalar = criterion(next_state_pred, next_state)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                hidden = tuple(h.detach() for h in hidden)  # pyright: ignore [reportAssignmentType]

                total_train_loss += loss.item()
                total_train_samples += 1

            if end_flag:
                break

        avg_train_loss: float = total_train_loss / total_train_samples
        val_loss: float = evaluate(
            data_loader=data_loader, model=model, criterion=criterion
        )

        wandb.log({"train/loss": avg_train_loss, "val/loss": val_loss})
        logger.info(
            f"Epoch {epoch + 1:>{len(str(n_epochs))}}/{n_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "world_model_best.pth")
            logger.info("Best model saved.")


def evaluate(
    data_loader: WorldModelDataLoader, model: nn.Module, criterion: nn.Module
) -> float:
    device: DeviceLikeType = wandb.config["device"]
    hidden_size: int = wandb.config["hidden_size"]
    seq_len: int = wandb.config["seq_len"]

    model.eval()
    data_loader.set_valid()
    total_val_loss: float = 0.0
    total_val_samples: int = 0

    with torch.no_grad():
        while True:
            states: BSState
            actions: BSAction
            end_flag: bool
            states, actions, end_flag = data_loader.get_batch()
            batch_size_actual: int = states.size(0)

            hidden: tuple[F1BH, F1BH] = (
                torch.zeros(1, batch_size_actual, hidden_size).to(device),
                torch.zeros(1, batch_size_actual, hidden_size).to(device),
            )

            for t in range(seq_len - 1):
                current_state: BState = states[:, t]
                action: BAction = actions[:, t]
                next_state: BState = states[:, t + 1].view(batch_size_actual, -1)

                next_state_pred: BState
                next_state_pred, hidden = model(current_state, action, hidden)
                loss: FloatScalar = criterion(next_state_pred, next_state)

                total_val_loss += loss.item()
                total_val_samples += 1

            if end_flag:
                break

    avg_val_loss: float = total_val_loss / total_val_samples
    return avg_val_loss


def test(
    data_loader: WorldModelDataLoader, model: nn.Module, criterion: nn.Module
) -> float:
    device: DeviceLikeType = wandb.config["device"]
    hidden_size: int = wandb.config["hidden_size"]
    seq_len: int = wandb.config["seq_len"]

    model.eval()
    data_loader.set_test()
    total_test_loss = 0
    total_test_samples = 0

    with torch.no_grad():
        while True:
            states: BSState
            actions: BSAction
            end_flag: bool
            states, actions, end_flag = data_loader.get_batch()
            batch_size_actual: int = states.size(0)

            hidden: tuple[F1BH, F1BH] = (
                torch.zeros(1, batch_size_actual, hidden_size).to(device),
                torch.zeros(1, batch_size_actual, hidden_size).to(device),
            )

            for t in range(seq_len - 1):
                current_state: BState = states[:, t]
                action: BAction = actions[:, t]
                next_state: BState = states[:, t + 1].view(batch_size_actual, -1)
                next_state_pred: BState
                next_state_pred, hidden = model(current_state, action, hidden)
                loss: FloatScalar = criterion(next_state_pred, next_state)

                total_test_loss += loss.item()
                total_test_samples += 1

            if end_flag:
                break

    avg_test_loss: float = total_test_loss / total_test_samples
    logger.info(f"Test Loss: {avg_test_loss:.4f}")
    return avg_test_loss


def main() -> None:
    tk.logging.init()
    wandb.init(
        config={
            "batch_size": 16,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "hidden_size": 128,
            "lr": 3e-4,
            "n_epochs": 50,
            "seq_len": 10,
        },
    )
    wandb.define_metric("val/loss", summary="min")
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
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=wandb.config["lr"])
    train(
        data_loader=data_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        n_epochs=wandb.config["n_epochs"],
    )
    evaluate(data_loader=data_loader, model=model, criterion=criterion)
    test_loss: float = test(data_loader=data_loader, model=model, criterion=criterion)
    wandb.summary["test/loss"] = test_loss


if __name__ == "__main__":
    main()
