import torch
import torch.nn as nn

 
# Conv1D + Transformer
class TransformerModel(nn.Module):
    def __init__(self, input_features, seq_len, d_model, num_heads, ff_dim, 
                 num_encoder_blocks, kernel_size, dropout, head_dropout):
        """
        基于 Conv1D 和 Transformer 编码器的 RUL 预测模型
        
        参数:
        - input_features (int): 输入特征的数量 (例如 14)
        - seq_len (int): 序列长度 (例如 30)
        - d_model (int): Transformer 的内部维度 (必须能被 num_heads 整除)
        - num_heads (int): 多头注意力的头数
        - ff_dim (int): Transformer 内部前馈网络的维度
        - num_encoder_blocks (int): Transformer 编码器层数
        - kernel_size (int): 1D 卷积核大小
        - dropout (float): Transformer 内部的 dropout 率
        - head_dropout (float): 回归头部的 dropout 率
        """
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        
        if self.d_model  % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除。")

        self.kernel_size = kernel_size
        # 自动计算 "same" padding，以保持序列长度不变
        self.padding = (self.kernel_size - 1) // 2
        
        # --- 1. 1D 卷积层 (特征提取器) ---
        # 期望输入: [B, F, T]
        self.conv_layer = nn.Conv1d(
            in_channels=input_features,
            out_channels=self.d_model,
            kernel_size=self.kernel_size,
            padding=self.padding
        )
        # BatchNorm1d 作用在 'channels' 维度 (即 d_model 维度)
        self.conv_batch_norm = nn.BatchNorm1d(self.d_model)
        self.conv_activation = nn.ReLU()
        
        # --- 2. 位置编码 ---
        # 使用可学习的位置嵌入 (Learned Positional Embedding)
        # 我们需要 seq_len 个位置, 每个位置 d_model 维
        self.pos_embedding = nn.Embedding(seq_len, self.d_model)
        # 注册一个 buffer (不作为模型参数)，用于索引
        self.register_buffer('positions', torch.arange(0, seq_len))
        self.embed_dropout = nn.Dropout(dropout)

        # --- 3. Transformer 编码器 ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True        # 输入形状: [Batch, SeqLen, d_model]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_blocks
        )
    
        # --- 4. 回归头 (Regression Head) ---
        # 假设输入是 [B, d_model]
        self.regression_head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        前向传播
        x 形状: [B, T, F] (Batch, Time, Features)
        """
        
        # 1. 调整维度以适应 Conv1d
        # [B, T, F] -> [B, F, T]
        x = x.permute(0, 2, 1) 
        
        # 2. 通过 1D 卷积层
        # [B, F, T] -> [B, d_model, T]
        x_conv = self.conv_layer(x)
        x_conv = self.conv_batch_norm(x_conv)
        x_conv = self.conv_activation(x_conv)
        
        # 3. 调整维度以适应 TransformerEncoder (batch_first=True)
        # [B, d_model, T] -> [B, T, d_model]
        x_trans = x_conv.permute(0, 2, 1)
        
        # 4. 添加位置编码
        # self.positions -> [T]
        # pos_enc -> [T, d_model]
        # [B, T, d_model] + [T, d_model] (自动广播)
        pos_enc = self.pos_embedding(self.positions)
        x = x_trans + pos_enc
        x = self.embed_dropout(x)
        
        # 5. 通过 Transformer 编码器
        # [B, T, d_model] -> [B, T, d_model]
        x = self.transformer_encoder(x)
        
        # 6. 池化/选择 (!!! 关键步骤 !!!)
        # (这里使用全局平均池化, x.mean(dim=1))
        # [B, T, d_model] -> [B, d_model]
        x_pooled = x.mean(dim=1)
        
        # 7. 通过回归头
        # [B, d_model] -> [B, 1]
        out = self.regression_head(x_pooled)
        
        return out