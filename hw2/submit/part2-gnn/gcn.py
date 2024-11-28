import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch_geometric.data import DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.nn import GCNConv, global_mean_pool

path = "./data/QM9"
dataset = QM9(path)
DIPOLE_INDEX = 0  # 偶极矩在 y 中的位置

train_dataset = dataset[:10000]
val_dataset = dataset[10000:11000]
test_dataset = dataset[11000:12000]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class GCN(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(
            dataset.num_features + 3, hidden_dim
        )  # 将节点特征和坐标拼接
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, 1)  # 输出偶极矩

    def forward(self, data):
        # 将节点特征和坐标拼接
        x = torch.cat([data.x, data.pos], dim=1)
        edge_index = data.edge_index
        batch = data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN(hidden_dim=128).to(device)
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


# 训练模型
wandb.init(group="GCN")
train_losses, val_losses = [], []
for epoch in range(1, 51):
    train_loss = train()
    val_loss = evaluate(val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    wandb.log({"train/loss": train_loss, "val/loss": val_loss})
    print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# 测试集评估
test_loss = evaluate(test_loader)
wandb.summary["test/loss"] = test_loss
print(f"Test Loss: {test_loss:.4f}")