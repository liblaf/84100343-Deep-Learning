import numpy as np
import numpy.typing as npt
import sklearn.metrics
import torch
import torch.nn.functional as F
import wandb
from jaxtyping import Float
from torch import optim
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

# 定义数据集和特征索引
path = "./data/QM9"
dataset = QM9(path)
DIPOLE_INDEX = 0  # 偶极矩在 y 中的位置

# 划分数据集
train_dataset = dataset[:10000]
val_dataset = dataset[10000:11000]
test_dataset = dataset[11000:12000]

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.conv1 = SAGEConv(dataset.num_features + 3, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, 1)  # 输出偶极矩

    def forward(self, data):
        x = torch.cat([data.x, data.pos], dim=1)  # 将节点特征和坐标拼接
        edge_index = data.edge_index
        batch = data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GraphSAGE(hidden_dim=128).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.y[:, DIPOLE_INDEX].unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)


def evaluate(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = F.mse_loss(out, data.y[:, DIPOLE_INDEX].unsqueeze(1))
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


wandb.init(group="GrahpSAGE")
train_losses, val_losses = [], []
for epoch in range(1, 51):
    train_loss = train()
    val_loss = evaluate(val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    wandb.log({"train/loss": train_loss, "val/loss": val_loss})
    print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

test_loss = evaluate(test_loader)
wandb.summary["test/loss"] = test_loss
print(f"Test Loss: {test_loss:.4f}")


def calculate_r2(loader: DataLoader) -> float:
    model.eval()
    y_true_list: list[Float[npt.NDArray, "batch_size 1"]] = []
    y_pred_list: list[Float[npt.NDArray, "batch_size 1"]] = []
    with torch.no_grad():
        for data_ in loader:
            data: Data = data_.to(device)
            out: Float[torch.Tensor, "batch_size 1"] = model(data)
            y_true_list.append(data.y[:, DIPOLE_INDEX].unsqueeze(1).numpy(force=True))
            y_pred_list.append(out.numpy(force=True))
    y_true: Float[npt.NDArray, "N 1"] = np.vstack(y_true_list)
    y_pred: Float[npt.NDArray, "N 1"] = np.vstack(y_pred_list)
    return sklearn.metrics.r2_score(y_true, y_pred)  # pyright: ignore [reportReturnType]


r2: float = calculate_r2(test_loader)
wandb.summary["test/r2"] = r2
print(f"Test R2 Score: {r2:.4f}")
