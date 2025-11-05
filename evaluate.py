import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- 从 src 导入我们的模块 ---
from src.config import (DEVICE, DATA_DIR, MODEL_SAVE_DIR, 
                        MODEL_PARAMS_FD001, MODEL_PARAMS_FD002, 
                        MODEL_PARAMS_FD003, MODEL_PARAMS_FD004,
                        SEQUENCE_LENGTH, RUL_CAP) # 需要 RUL_CAP 来绘图
from src.model import TransformerModel
from src.data_loader import get_dataloaders # 我们用它来获取 test_loader
from src.utils import load_model
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def calculate_cmaps_score(y_true, y_pred):
    """
    计算 C-MAPSS 竞赛的非对称评分函数。
    晚预测（d > 0，危险）的惩罚高于早预测（d < 0，保守）。
    """
    diff = y_pred - y_true
    score = 0
    for d in diff:
        if d < 0:
            score += np.exp(-d / 13) - 1
        else:
            score += np.exp(d / 10) - 1
    return score

def evaluate_model(model, dataloader, device):
    """
    在给定的 dataloader (测试集) 上运行完整的评估。
    
    返回:
        y_true (np.array): 所有的真实标签
        y_pred (np.array): 所有的模型预测值
    """
    model.eval() # 确保模型处于评估模式 (关闭 dropout 等)
    y_pred_list = []
    y_true_list = []
    
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            
            # 模型前向传播
            outputs = model(batch_X)
            
            # 收集结果
            y_pred_list.append(outputs.cpu().numpy())
            y_true_list.append(batch_y.numpy())

    # 将所有批次的结果合并为单个 NumPy 数组
    y_pred = np.concatenate(y_pred_list).flatten()
    y_true = np.concatenate(y_true_list).flatten()
    
    return y_true, y_pred

def main(dataset_name):
    """
    主评估函数。
    加载测试数据，加载已保存的最佳模型，并计算最终指标。
    """
    
    # --- 1. 加载测试数据 ---
    # 我们调用 get_dataloaders，它会正确地加载和预处理(!!) test_df 和 RUL_df
    # 并使用在训练时保存的 scaler 来归一化 test_df。
    # 我们只需要返回的 test_loader 和 input_features。
    print(f"正在加载 {dataset_name} 的测试数据和归一化器...")
    try:
        _, _, test_loader, input_features = get_dataloaders(
            dataset_name=dataset_name,
            data_dir=DATA_DIR,
            scaler_save_dir=MODEL_SAVE_DIR
        )
    except FileNotFoundError as e:
        print(f"错误: 找不到归一化器文件。 {e}")
        print(f"请先运行: python train.py --dataset {dataset_name}")
        return
        
    print(f"测试数据加载完毕。输入特征数: {input_features}")

    # --- 2. 实例化并加载训练好的模型 ---
    # (这里可以添加逻辑来为 FD002, FD003, FD004 选择不同的 params)
    if dataset_name == 'FD001':
        params = MODEL_PARAMS_FD001
    elif dataset_name == 'FD002':
        params = MODEL_PARAMS_FD002
    elif dataset_name == 'FD003':
        params = MODEL_PARAMS_FD003
    elif dataset_name == 'FD004':
        params = MODEL_PARAMS_FD004
    else:
        print(f"错误: 未知数据集 {dataset_name}。")
        return

    model = TransformerModel(
        input_features=input_features,
        seq_len=SEQUENCE_LENGTH,
        d_model=params["d_model"],
        num_heads=params["num_heads"],
        ff_dim=params["ff_dim"],
        num_encoder_blocks=params["num_encoder_blocks"],
        dropout=params["dropout"],
        head_dropout=params["head_dropout"],
        kernel_size=params["kernel_size"],
    ).to(DEVICE)
    
    model_path = os.path.join(MODEL_SAVE_DIR, f"{dataset_name}_model.pth")
    
    try:
        model = load_model(model, model_path, DEVICE) # 使用 utils.py 中的函数
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 {model_path}。")
        print(f"请先运行: python train.py --dataset {dataset_name}")
        return
        
    # --- 3. 执行评估 ---
    print("模型加载成功。正在评估测试集...")
    y_true, y_pred = evaluate_model(model, test_loader, DEVICE)
    
    # --- 4. 计算并报告指标 ---
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    cmaps_score = calculate_cmaps_score(y_true, y_pred)
    
    print("\n" + "="*30)
    print(f"  评估结果 (Test Set: {dataset_name})")
    print("="*30)
    print(f"RMSE (均方根误差):   {rmse:.4f}")
    print(f"MAE (平均绝对误差):  {mae:.4f}")
    print(f"C-MAPSS 评分:      {cmaps_score:.4f} (越低越好)")
    print("="*30)
    
    # --- 5. 生成并保存评估图 ---
    plot_path = os.path.join(MODEL_SAVE_DIR, f"{dataset_name}_evaluation_plot.png")
    plt.figure(figsize=(10, 10))
    plt.scatter(y_true, y_pred, alpha=0.5, label='预测值')
    plt.plot([0, RUL_CAP], [0, RUL_CAP], 'r--', label="理想线 (y=x)")
    plt.title(f"真实 RUL vs 预测 RUL ({dataset_name}) - 最终评估")
    plt.xlabel("真实 RUL (True RUL)")
    plt.ylabel("预测 RUL (Predicted RUL)")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, RUL_CAP + 10)
    plt.ylim(0, RUL_CAP + 10)
    plt.savefig(plot_path)
    print(f"评估散点图已保存到: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估 C-MAPSS RUL 预测模型")
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['FD001', 'FD002', 'FD003', 'FD004'],
                        help='要评估的数据集名称 (例如: FD001)')
    args = parser.parse_args()
    
    main(dataset_name=args.dataset)