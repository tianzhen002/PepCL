import math

import torch
from torch import nn



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('encoding', self._get_timing_signal(max_len, d_model))

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]

    def _get_timing_signal(self, length, channels):
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2) * -(math.log(10000.0) / channels))
        pe = torch.zeros(length, channels)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        d_model：表示整个特征维度，要求必须是偶数。
        max_len：最大序列长度。
        """
        super(RotaryPositionalEncoding, self).__init__()
        assert d_model % 2 == 0, "d_model must be even for rotary encoding"
        self.d_model = d_model
        # 预计算角度，形状为 [max_len, d_model//2]
        self.register_buffer('angles', self._precompute_angles(max_len, d_model))

    def _precompute_angles(self, max_len, d_model):
        half_dim = d_model // 2
        # 计算每个维度的倒数频率
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
        # 生成位置向量 [max_len, 1]
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        # 计算角度矩阵 [max_len, half_dim]
        angles = positions * inv_freq.unsqueeze(0)
        return angles

    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
        旋转编码的实现：将 x 分成两半，分别进行旋转后拼接回来。
        """
        batch, seq_len, d_model = x.shape
        half = d_model // 2
        # 获取对应位置的角度，扩展到 batch 维度，形状 [batch, seq_len, half]
        angles = self.angles[:seq_len, :].unsqueeze(0).expand(batch, -1, -1)
        x1 = x[:, :, :half]
        x2 = x[:, :, half:]
        # 应用旋转公式：新x1 = x1*cos(angle) - x2*sin(angle)，新x2 = x1*sin(angle) + x2*cos(angle)
        x1_rot = x1 * torch.cos(angles) - x2 * torch.sin(angles)
        x2_rot = x1 * torch.sin(angles) + x2 * torch.cos(angles)
        # 拼接回去
        x_rot = torch.cat([x1_rot, x2_rot], dim=-1)
        return x_rot

class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, max_len=5000):
        super(TransformerModel, self).__init__()
        # 使用旋转位置编码
        self.pos_encoder = RotaryPositionalEncoding(d_model, max_len=max_len)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers,
                                          dim_feedforward=dim_feedforward)

    def forward(self, src, tgt):
        # 注意：旋转位置编码是直接替换原来的加法位置编码
        src = self.pos_encoder(src)
        output = self.transformer(src, tgt)
        return output


# class TCN(nn.Module):
#     def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
#         super(TCN, self).__init__()
#         layers = []
#         num_levels = len(num_channels)
#
#         for i in range(num_levels):
#             dilation = 2 ** i
#             in_channels = input_size if i == 0 else num_channels[i - 1]
#             out_channels = num_channels[i]
#             layers += [nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=dilation * (kernel_size - 1) // 2),
#                        nn.ReLU(),
#                        nn.Dropout(dropout)]
#
#         self.network = nn.Sequential(*layers)
#         self.linear = nn.Linear(num_channels[-1], output_size)
#
#     def forward(self, x):
#         x = self.network(x.permute(0, 2, 1))
#         x = self.linear(x.permute(0, 2, 1))    # Reshape for linear layer
#         return x




class AIMP(torch.nn.Module):
    def __init__(self, pre_feas_dim, hidden, n_transformer, dropout):
        super(AIMP, self).__init__()

        self.pre_embedding = nn.Sequential(
            nn.Conv1d(pre_feas_dim, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )

        # self.dbemb=nn.Sequential(
        #     nn.Linear(128,hidden),
        #     nn.BatchNorm1d(hidden),
        #     nn.Linear(hidden,128),
        # )

        self.dbemb = nn.Sequential(
            nn.Linear(128, hidden),
            nn.LayerNorm(hidden),  # 使用 LayerNorm 更适合处理 [N, 40, hidden] 数据
            nn.ReLU(),
            nn.Linear(hidden, 128),
            nn.ReLU()  # 如果你希望在最后也有激活函数的话
        )

        self.embedding = nn.Sequential(
            nn.Conv1d(hidden +128, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),

            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )

        self.bn = nn.ModuleList([nn.BatchNorm1d(pre_feas_dim),
                                 # nn.BatchNorm1d(feas_dim),
                                 nn.BatchNorm1d(128)
                                 # nn.BatchNorm1d(seq_feas_dim),
                                 ])

        # self.n_bilstm = n_bilstm
        self.n_transformer = n_transformer
        # self.lstm = nn.LSTM(input_size=hidden, hidden_size=hidden, num_layers=self.n_bilstm, bidirectional=True,
        #                     dropout=dropout, batch_first=True)
        #
        # self.lstm_act = nn.Sequential(
        #     nn.BatchNorm1d(hidden * 2),
        #     nn.ELU(inplace=True),
        #     nn.Dropout(dropout),
        #     nn.Conv1d(hidden * 2, hidden, kernel_size=1),
        # )


        # self.tcn = TCN(feas_dim, hidden, [hidden // 2, hidden // 2, hidden // 2], 21, dropout)
        # self.tcn_res = nn.Sequential(
        #     nn.Conv1d(hidden + feas_dim, hidden, kernel_size=1),
        #     nn.BatchNorm1d(hidden),
        #     nn.ELU(inplace=True),
        #     nn.Dropout(dropout),
        #     nn.Conv1d(hidden, hidden, kernel_size=1),
        # )

        # self.transformer = nn.Transformer(d_model=hidden, nhead=2, num_encoder_layers=self.n_transformer, batch_first=True)
        self.transformer = TransformerModel(d_model=hidden, nhead=4, num_layers=self.n_transformer,
                                            dim_feedforward=2048)
        self.transformer_act = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )
        self.transformer_res = nn.Sequential(
            nn.Conv1d(hidden + hidden, hidden, kernel_size=1),
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )
        self.transformer_pool = nn.AdaptiveAvgPool2d((1, None))
        self.clf = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in [self.pre_embedding, self.embedding, self.clf]:
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                # nn.init.zeros_(layer.bias)
        for layer in [self.transformer_act, self.transformer_res]:
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                # nn.init.zeros_(layer.bias)

    def forward(self, pre_feas, duibi):
        # batch_size = pre_feas.shape[0]
        pre_feas = self.bn[0](pre_feas.permute(0, 2, 1)).permute(0, 2, 1)
        # feas = self.bn[1](feas.permute(0, 2, 1)).permute(0, 2, 1)
        duibi = self.bn[1](duibi.permute(0, 2, 1)).permute(0, 2, 1)
        N = duibi.shape[0]
        duibi=self.dbemb(duibi.reshape(-1,128))

        duibi=duibi.reshape(N, 40, 128)



        pre_feas = self.pre_embedding(pre_feas.permute(0, 2, 1)).permute(0, 2, 1)

        # tcn_out = self.tcn(feas)
        # tcn_out = self.tcn_res(torch.cat([tcn_out, feas], dim=-1).permute(0, 2, 1)).permute(0, 2, 1)

        feas_em = self.embedding(torch.cat([pre_feas, duibi], dim=-1).permute(0, 2, 1)).permute(0, 2, 1)

        transformer_out = self.transformer(feas_em, feas_em)
        transformer_out = self.transformer_act(transformer_out.permute(0, 2, 1)).permute(0, 2, 1)
        transformer_out = self.transformer_res(torch.cat([transformer_out, feas_em], dim=-1).permute(0, 2, 1)).permute(
            0, 2, 1)
        transformer_out = self.transformer_pool(transformer_out).squeeze(1)

        out = self.clf(transformer_out)
        out = torch.nn.functional.softmax(out, -1)
        return out[:, -1]
