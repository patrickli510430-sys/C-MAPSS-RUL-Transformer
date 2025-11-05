import torch
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def save_model(model, path):
    """保存模型的状态字典"""
    torch.save(model.state_dict(), path)
    print(f"模型已保存到: {path}")

def load_model(model, path, device):
    """加载模型的状态字典"""
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"模型已从 {path} 加载。")
    return model

def plot_training_history(history, save_path):
    """绘制并保存训练历史图"""
    plt.figure(figsize=(12, 6))
    plt.plot(history["loss"], label="训练损失 (Loss)")
    plt.plot(history["val_loss"], label="验证损失 (Val Loss)")
    plt.title("模型训练历史")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"训练历史图已保存到: {save_path}")
    
def plot_predictions(y_true, y_pred, save_path):
    """绘制并保存预测结果图"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="真实值 (True)")
    plt.plot(y_pred, label="预测值 (Pred)")
    plt.title("模型预测结果")
    plt.xlabel("样本")
    plt.ylabel("RUL")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"预测结果图已保存到: {save_path}")
    
def plot_predictions_with_error(y_true, y_pred, y_true_std, y_pred_std, save_path):
    """绘制并保存预测结果图"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="真实值 (True)")
    plt.plot(y_pred, label="预测值 (Pred)")
    plt.fill_between(range(len(y_true)), 
                     y_true - y_true_std, 
                     y_true + y_true_std, 
                     color='blue', alpha=0.2, label="真实值 ± 1σ")
    plt.fill_between(range(len(y_pred)), 
                     y_pred - y_pred_std, 
                     y_pred + y_pred_std, 
                     color='orange', alpha=0.2, label="预测值 ± 1σ")
    plt.title("模型预测结果")
    plt.xlabel("样本")
    plt.ylabel("RUL")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"预测结果图已保存到: {save_path}")