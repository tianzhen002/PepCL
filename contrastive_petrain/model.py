#写入使用的模型

import torch
import torch.nn as nn


class VanillaNet(nn.Module):
    def __init__(self, hidden_dim, out_dim, device, dtype):
        super(VanillaNet, self).__init__()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.device = device
        self.dtype = dtype

        self.fc1 = nn.Linear(1280, hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# class LayerNormNet(nn.Module):
#     def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.1):
#         super(LayerNormNet, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.out_dim = out_dim
#         self.device = device
#         self.dtype = dtype
#         self.drop_out = drop_out
#
#         # 线性层：调整维度
#         self.fc1 = nn.Linear(1024, hidden_dim, dtype=dtype, device=device)
#         self.ln1 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
#
#         # Transformer 处理残基信息交互
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim,
#             nhead=8,
#             dim_feedforward=4 * hidden_dim,  # 前馈层扩大 4 倍
#             dropout=drop_out,
#             batch_first=True,
#             dtype=dtype,
#             device=device
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
#
#         # 线性层：输出到 out_dim
#         self.fc2 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
#         self.ln2 = nn.LayerNorm(out_dim, dtype=dtype, device=device)
#
#         self.dropout = nn.Dropout(p=drop_out)
#
#     def forward(self, x):
#         # x 形状: (N, theta, 1024)
#         N, theta, _ = x.shape
#
#         # 通过第一层线性变换和 LayerNorm
#         x = self.dropout(self.ln1(self.fc1(x)))  # (N, theta, hidden_dim)
#         x = torch.relu(x)
#
#         # Transformer 进行信息交互（残基级别）
#         x = self.transformer(x)  # (N, theta, hidden_dim)
#
#         # 输出层
#         x = self.dropout(self.ln2(self.fc2(x)))  # (N, theta, out_dim)
#         x = torch.relu(x)
#
#         return x

class LayerNormNet(nn.Module):
    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.5):  # <- dropout 提高到 0.3
        super(LayerNormNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.drop_out = drop_out
        self.device = device
        self.dtype = dtype

        self.fc1 = nn.Linear(1024, hidden_dim, dtype=dtype, device=device)
        self.ln1 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device)
        self.ln2 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)

        # 更高的 Dropout 概率
        self.dropout1 = nn.Dropout(p=drop_out)
        self.dropout2 = nn.Dropout(p=drop_out)

    def forward(self, x):
        N, theta, _ = x.shape
        x = x.view(N * theta, -1)
        x = self.dropout1(self.ln1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout2(self.ln2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        x = x.view(N, theta, -1)
        return x

# #
# class LayerNormNet(nn.Module):
#     """
#     用 1D-CNN 代替 MLP：
#     - 输入 x: (batch_size, theta, feat_dim)
#     - 输出 y: (batch_size, theta, out_dim)
#     """
#     def __init__(self, hidden_dim: int, out_dim: int,
#                  drop_out: float = 0.3, device=None, dtype=torch.float32):
#         super(LayerNormNet, self).__init__()
#         self.device = device
#         self.dtype = dtype
#
#         # 第一层卷积：1024 -> hidden_dim
#         self.conv1 = nn.Conv1d(
#             in_channels=1024, out_channels=hidden_dim,
#             kernel_size=3, padding=1
#             )
#         self.ln1 = nn.LayerNorm(hidden_dim)
#         self.dropout1 = nn.Dropout(p=drop_out)
#
#         # 第二层卷积：hidden_dim -> hidden_dim
#         self.conv2 = nn.Conv1d(
#             in_channels=hidden_dim, out_channels=hidden_dim,
#             kernel_size=3, padding=1
#
#         )
#         self.ln2 = nn.LayerNorm(hidden_dim)
#         self.dropout2 = nn.Dropout(p=drop_out)
#
#         # 第三层卷积：hidden_dim -> out_dim
#         self.conv3 = nn.Conv1d(
#             in_channels=hidden_dim, out_channels=out_dim,
#             kernel_size=1
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: (N, theta, feat_dim)
#         # 转为 Conv1d 需要的 (N, feat_dim, theta)
#         x = x.permute(0, 2, 1)
#
#         # conv1 + LayerNorm + ReLU + Dropout
#         x = self.conv1(x)                         # (N, hidden_dim, theta)
#         x = x.permute(0, 2, 1)                    # (N, theta, hidden_dim)
#         x = self.ln1(x)
#         x = torch.relu(x)
#         x = self.dropout1(x)
#         x = x.permute(0, 2, 1)                    # (N, hidden_dim, theta)
#
#         # conv2 + LayerNorm + ReLU + Dropout
#         x = self.conv2(x)                         # (N, hidden_dim, theta)
#         x = x.permute(0, 2, 1)                    # (N, theta, hidden_dim)
#         x = self.ln2(x)
#         x = torch.relu(x)
#         x = self.dropout2(x)
#         x = x.permute(0, 2, 1)                    # (N, hidden_dim, theta)
#
#         # conv3 输出 (N, out_dim, theta)
#         x = self.conv3(x)                         # (N, out_dim, theta)
#
#         # 恢复成 (N, theta, out_dim)
#         x = x.permute(0, 2, 1)
#         return x


# class ResBlock1D(nn.Module):
#     def __init__(self, channels, drop_out=0.3):
#         super().__init__()
#         self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm1d(channels)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=drop_out)
#         self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm1d(channels)
#
#     def forward(self, x):
#         identity = x
#         out = self.dropout(self.relu(self.bn1(self.conv1(x))))
#         out = self.bn2(self.conv2(out))
#         return self.relu(out + identity)
#
#
# class LayerNormNet(nn.Module):
#     def __init__(self, hidden_dim, out_dim, drop_out=0.3):
#         super().__init__()
#         self.conv_in = nn.Conv1d(in_channels=1024, out_channels=hidden_dim, kernel_size=3, padding=1)
#         self.bn_in = nn.BatchNorm1d(hidden_dim)
#         self.relu = nn.ReLU()
#         self.resblock1 = ResBlock1D(hidden_dim, drop_out)
#         self.resblock2 = ResBlock1D(hidden_dim, drop_out)
#         self.pool = nn.AdaptiveAvgPool1d(output_size=20)  # 控制长度统一
#         self.conv_out = nn.Conv1d(hidden_dim, out_dim, kernel_size=1)
#
#     def forward(self, x):
#         x = x.permute(0, 2, 1)  # (B, feat_dim, theta)
#         x = self.relu(self.bn_in(self.conv_in(x)))
#         x = self.resblock1(x)
#         x = self.resblock2(x)
#         x = self.pool(x)  # (B, hidden_dim, 20)
#         x = self.conv_out(x)  # (B, out_dim, 20)
#         x = x.permute(0, 2, 1)  # (B, 20, out_dim)
#         return x



# class LayerNormNet(nn.Module):
#     def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.1):
#         super(LayerNormNet, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.out_dim = out_dim
#         self.device = device
#         self.dtype = dtype
#         self.drop_out = drop_out
#
#         # 线性层：降维到 hidden_dim
#         self.fc1 = nn.Linear(1024, hidden_dim, dtype=dtype, device=device)
#         self.ln1 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
#
#         # **CNN 负责序列信息交互**
#         self.conv1 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1, dtype=dtype, device=device)
#         self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1, dtype=dtype, device=device)
#         self.conv3 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1, dtype=dtype, device=device)
#
#         self.bn1 = nn.BatchNorm1d(hidden_dim, dtype=dtype, device=device)
#         self.bn2 = nn.BatchNorm1d(hidden_dim, dtype=dtype, device=device)
#         self.bn3 = nn.BatchNorm1d(hidden_dim, dtype=dtype, device=device)
#
#         self.dropout = nn.Dropout(p=drop_out)
#
#         # 线性层：降维到 out_dim
#         self.fc2 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
#         self.ln2 = nn.LayerNorm(out_dim, dtype=dtype, device=device)
#
#     def forward(self, x):
#         # x 形状: (N, theta, 1024)
#
#         # 通过第一层线性变换和 LayerNorm
#         x = self.fc1(x)  # (N, theta, hidden_dim)
#         x = self.ln1(x)
#         x = torch.relu(x)
#         x = self.dropout(x)
#
#         # **调整维度以适应 1D 卷积**
#         x = x.permute(0, 2, 1)  # (N, hidden_dim, theta)
#
#         # **CNN 进行局部特征提取**
#         x = torch.relu(self.bn1(self.conv1(x)))
#         x = torch.relu(self.bn2(self.conv2(x)))
#         x = torch.relu(self.bn3(self.conv3(x)))
#
#         # **调整回原始维度**
#         x = x.permute(0, 2, 1)  # (N, theta, hidden_dim)
#
#         # 通过最终线性层降维
#         x = self.fc2(x)  # (N, theta, out_dim)
#         x = self.ln2(x)
#         x = torch.relu(x)
#         x = self.dropout(x)
#
#         return x



class BatchNormNet(nn.Module):
    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(BatchNormNet, self).__init__()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.drop_out = drop_out
        self.device = device
        self.dtype = dtype

        self.fc1 = nn.Linear(1280, hidden_dim, dtype=dtype, device=device)
        self.bn1 = nn.BatchNorm1d(hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim,
                             dtype=dtype, device=device)
        self.bn2 = nn.BatchNorm1d(hidden_dim, dtype=dtype, device=device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.dropout(self.bn1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.bn2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        return x


class InstanceNorm(nn.Module):
    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(InstanceNorm, self).__init__()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.drop_out = drop_out
        self.device = device
        self.dtype = dtype

        self.fc1 = nn.Linear(1280, hidden_dim, dtype=dtype, device=device)
        self.in1 = nn.InstanceNorm1d(hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim,
                             dtype=dtype, device=device)
        self.in2 = nn.InstanceNorm1d(hidden_dim, dtype=dtype, device=device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.dropout(self.in1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.in2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        return x
