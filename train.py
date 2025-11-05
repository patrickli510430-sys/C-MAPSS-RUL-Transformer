import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import copy
from tqdm import tqdm
import argparse # 用于命令行参数

# --- 从 src 导入我们的模块 ---
from src.config import (DEVICE, DATA_DIR, MODEL_SAVE_DIR, 
                        EPOCHS, EARLY_STOP_PATIENCE,
                        MODEL_PARAMS_FD001,MODEL_PARAMS_FD002,MODEL_PARAMS_FD003,MODEL_PARAMS_FD004,
                        OPTIMIZER_PARAMS,
                        SEQUENCE_LENGTH,ALL_MODEL_PARAMS)
from src.model import TransformerModel
from src.data_loader import get_dataloaders
from src.utils import save_model, plot_training_history

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """执行一个 Epoch 的训练"""
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        # 将所有参数的梯度范数裁剪到最大值 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
    return running_loss / len(dataloader)

def validate_one_epoch(model, dataloader, criterion, device):
    """执行一个 Epoch 的验证"""
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            running_loss += loss.item()
    return running_loss / len(dataloader)

def main(dataset_name):
    """主训练函数"""
    
    # --- 1. 准备数据 ---
    # get_dataloaders 将处理所有事情：加载、预处理、保存 scaler、创建 loaders
    train_loader, val_loader, test_loader, input_features = get_dataloaders(
        dataset_name=dataset_name,
        data_dir=DATA_DIR,
        scaler_save_dir=MODEL_SAVE_DIR
    )
    
    # --- 2. 实例化模型 ---
    # 首先，我们需要获取数据集的参数，以便为模型选择正确的参数。
    # 如果找不到，则使用 FD001 的参数。 
    if dataset_name in ALL_MODEL_PARAMS:
        params = ALL_MODEL_PARAMS[dataset_name]
        print(f"INFO: 正在为数据集 {dataset_name} 选择参数: {params}")
    else:
        # 如果找不到，则默认使用 FD001 的参数
        print(f"WARNING: 未找到数据集 {dataset_name} 的参数，使用 FD001 默认参数。")
        params = ALL_MODEL_PARAMS["FD001"]
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
    
    print(f"模型已实例化并移动到: {DEVICE}")
    print(f"总参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- 3. 定义优化器和损失函数 ---
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), **OPTIMIZER_PARAMS)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, min_lr=1e-7, verbose=True)

    # --- 4. 训练循环 ---
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    history = {'loss': [], 'val_loss': []}

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = validate_one_epoch(model, val_loader, criterion, DEVICE)
        
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  (新低: 验证损失 {best_val_loss:.4f}。正在保存...)")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"早停触发 (Patience={EARLY_STOP_PATIENCE})。")
                break
                
    # --- 5. 保存结果 ---
    print("训练结束。")
    if best_model_state:
        # **【已修复】** 在保存之前，将模型的状态恢复到最佳状态
        model.load_state_dict(best_model_state)
        print("已加载最佳模型权重。")
        
        model_save_path = os.path.join(MODEL_SAVE_DIR, f"{dataset_name}_model.pth")
        save_model(model, model_save_path) # 使用 utils 函数
    else:
        print("警告：未找到 'best_model_state'。模型未保存。")
    
    # 保存训练历史图
    plot_save_path = os.path.join(MODEL_SAVE_DIR, f"{dataset_name}_training_history.png")
    plot_training_history(history, plot_save_path)
    print(f"训练历史图已保存到: {plot_save_path}")

    print(f"--- 训练 {dataset_name} 完成 ---")

if __name__ == "__main__":
    # 允许从命令行选择数据集
    parser = argparse.ArgumentParser(description="训练 C-MAPSS RUL 预测模型")
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['FD001', 'FD002', 'FD003', 'FD004'],
                        help='要训练的数据集名称')
    args = parser.parse_args()
    
    main(dataset_name=args.dataset)