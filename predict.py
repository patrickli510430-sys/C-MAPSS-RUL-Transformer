import torch
import pickle
import numpy as np
import argparse

# 导入我们的模块
from src.model import TransformerModel
from src.config import DEVICE, MODEL_SAVE_DIR, MODEL_PARAMS_FD001, SEQUENCE_LENGTH

def load_prediction_assets(dataset_name, params, input_features):
    """加载模型和归一化器"""
    
    # 1. 实例化模型 (与训练时完全相同的参数)
    model = TransformerModel(
        input_features=input_features,
        seq_len=SEQUENCE_LENGTH,
        d_model=params["d_model"],
        num_heads=params["num_heads"],
        ff_dim=params["ff_dim"],
        num_encoder_blocks=params["num_encoder_blocks"],
        dropout=params["dropout"]
    ).to(DEVICE)
    
    # 2. 加载模型权重
    model_path = os.path.join(MODEL_SAVE_DIR, f"{dataset_name}_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval() # !! 设置为评估模式 !!
    
    # 3. 加载归一化器 (!! 至关重要 !!)
    scaler_path = os.path.join(MODEL_SAVE_DIR, f"{dataset_name}_scaler.pkl")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    print(f"模型和归一化器 {dataset_name} 加载成功。")
    return model, scaler

def predict_rul(model, scaler, input_data_np):
    """
    对一个 NumPy 序列进行 RUL 预测
    
    Args:
        model (TransformerModel): 已加载的模型
        scaler (MinMaxScaler): 已加载的归一化器
        input_data_np (np.array): 形状为 (30, 18) 的 NumPy 数组 (30=SeqLen, 18=Features)
                                  注意：特征数量必须匹配！
    """
    
    if input_data_np.shape[0] != SEQUENCE_LENGTH:
        raise ValueError(f"输入序列长度必须为 {SEQUENCE_LENGTH}，但得到 {input_data_np.shape[0]}")
    
    # 1. 归一化
    data_scaled = scaler.transform(input_data_np)
    
    # 2. 转换为 Tensor 并增加 Batch 维度 [1, 30, 18]
    data_tensor = torch.tensor(data_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    # 3. 预测
    with torch.no_grad():
        predicted_rul = model(data_tensor)
        
    return predicted_rul.item()

if __name__ == "__main__":
    # 这是一个示例
    
    # 1. 加载模型
    # (注意: 这里的 18 必须是 FD001 训练时的实际特征数)
    FD001_FEATURES = 18 
    model_fd001, scaler_fd001 = load_prediction_assets(
        "FD001", MODEL_PARAMS_FD001, FD001_FEATURES
    )

    # 2. 模拟一个新数据样本 (30个时间步, 18个特征)
    # 在实际应用中，你将从你的测试集或实时数据中获取这个
    sample_data = np.random.rand(SEQUENCE_LENGTH, FD001_FEATURES) 
    
    # 3. 进行预测
    rul = predict_rul(model_fd001, scaler_fd001, sample_data)
    
    print(f"\n--- 预测结果 ---")
    print(f"预测的剩余使用寿命 (RUL) 为: {rul:.2f}")