import torch

# --- 1. 全局配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data/CMaps"
MODEL_SAVE_DIR = "./saved_models"
RUL_CAP = 120
SEQUENCE_LENGTH = 30

# --- 2. 训练配置 ---
BATCH_SIZE = 64
EPOCHS = 50                         # 训练的最大轮数
EARLY_STOP_PATIENCE = 10

# --- 3. 模型超参数 (使用 Conv1D + Transformer) ---
# FD001 的超参数
MODEL_PARAMS_FD001 = {
    "num_heads": 8,                  # 注意力头数
    "num_encoder_blocks": 4,         # 编码器块数量
    "kernel_size" : 5,
    "dropout" : 0.1,             
    "head_dropout" : 0.1,            # 回归头的 Dropout
    "d_model": 256,                  # 模型的主维度
    "ff_dim":  1024                  # 前馈网络维度(4 * "d_model")
}
# FD002 的超参数
MODEL_PARAMS_FD002 = {
    "num_heads": 8,                  # 注意力头数
    "num_encoder_blocks": 4,         # 编码器块数量
    "kernel_size" : 5,
    "dropout" : 0.1,             
    "head_dropout" : 0.1,            # 回归头的 Dropout
    "d_model": 256,                  # 模型的主维度
    "ff_dim":  1024                  # 前馈网络维度(4 * "d_model")
}
# FD003 的超参数
MODEL_PARAMS_FD003 = {
    "num_heads": 8,                  # 注意力头数
    "num_encoder_blocks": 4,         # 编码器块数量
    "kernel_size" : 5,
    "dropout" : 0.1,             
    "head_dropout" : 0.1,            # 回归头的 Dropout
    "d_model": 256,                  # 模型的主维度
    "ff_dim":  1024                  # 前馈网络维度(4 * "d_model")
}
# FD004 的超参数
MODEL_PARAMS_FD004 = {
    "num_heads": 8,                  # 注意力头数
    "num_encoder_blocks": 4,         # 编码器块数量
    "kernel_size" : 11,
    "dropout" : 0.1,             
    "head_dropout" : 0.1,            # 回归头的 Dropout
    "d_model": 256,                  # 模型的主维度
    "ff_dim":  1024                  # 前馈网络维度(4 * "d_model")
}

ALL_MODEL_PARAMS = {
    "FD001": MODEL_PARAMS_FD001,
    "FD002": MODEL_PARAMS_FD002,
    "FD003": MODEL_PARAMS_FD003,
    "FD004": MODEL_PARAMS_FD004,
}

# --- 4. 优化器配置 ---
OPTIMIZER_PARAMS = {
    "lr": 0.0001,                    # 学习率
    # "weight_decay": 5e-3           # AdamW 的权重衰减
}