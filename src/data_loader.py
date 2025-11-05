import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pickle
import torch

# 从配置中导入常量
from src.config import RUL_CAP, SEQUENCE_LENGTH, BATCH_SIZE

def load_data(data_dir, dataset_name):
    """加载原始的 train, test, RUL 文件"""
    train_path = os.path.join(data_dir, f'train_{dataset_name}.txt')
    test_path = os.path.join(data_dir, f'test_{dataset_name}.txt')
    rul_path = os.path.join(data_dir, f'RUL_{dataset_name}.txt')
    # 基于读取数据最后是否有空列而进行判断是否使用(.iloc[:, :~])
    train_df = pd.read_csv(train_path, sep=' ', header=None).iloc[:, :-2]
    test_df = pd.read_csv(test_path, sep=' ', header=None).iloc[:, :-2]
    rul_df = pd.read_csv(rul_path, sep=' ', header=None).iloc[:, :-1]
    
    columns = ['ID', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)]
    train_df.columns = columns
    test_df.columns = columns
    rul_df.columns = ['RUL']
    
    return train_df, test_df, rul_df

def calculate_rul(df):
    """为训练集计算 RUL 并封顶"""
    max_cycles = df.groupby('ID')['cycle'].max().reset_index()
    max_cycles.columns = ['ID', 'max_cycle']
    df = df.merge(max_cycles, on='ID', how='left')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df.drop(columns=['max_cycle'], inplace=True)
    df['RUL'] = df['RUL'].clip(upper=RUL_CAP)
    return df

def perform_feature_selection(df_train, variance_threshold=0):
    """(你之前设计的那个健壮的特征选择函数)"""
    sensor_cols = [col for col in df_train.columns if col.startswith('sensor')]
    setting_cols = ['setting1', 'setting2', 'setting3']
    sensor_vars = df_train[sensor_cols].var()
    discarded_sensors = sensor_vars[sensor_vars == variance_threshold].index.tolist()
    sensors_to_use = [col for col in sensor_cols if col not in discarded_sensors]
    final_feature_cols = setting_cols + sensors_to_use
    print(f"丢弃了 {len(discarded_sensors)} 个传感器: {discarded_sensors}")
    print(f"使用 {len(final_feature_cols)} 个特征.")
    return final_feature_cols

def create_oc_scalers(train_data, feature_cols):
    """为每个工况创建一个 scaler"""
    # 找到所有独特的工况 (由3个 setting 定义)
    # 你可能需要对 setting 值进行取整来分组
    train_data_to_fit = train_data[feature_cols]
    train_data_to_fit['oc_key'] = (
        train_data_to_fit['setting1'].round(0).astype(str) + '_' +
        train_data_to_fit['setting2'].round(0).astype(str) + '_' +
        train_data_to_fit['setting3'].round(0).astype(str)
    )
    
    # 为每个工况创建一个 scaler
    oc_scalers = {}
    for oc in train_data_to_fit['oc_key'].unique():
        # 筛选出该工况的数据
        oc_data = train_data_to_fit[train_data_to_fit['oc_key'] == oc][feature_cols]
        
        # 为该工况创建一个新的 scaler
        scaler = MinMaxScaler()
        scaler.fit(oc_data)
        
        # 存储这个 scaler
        oc_scalers[oc] = scaler
    return oc_scalers


#    你需要对 train 和 test 数据集都执行此操作
def transform_per_oc(data, scalers, feature_cols):
    data_copy = data.copy()
    
    # 同样的方法生成 oc_key
    data_copy['oc_key'] = (
        data_copy['setting1'].round(0).astype(str) + '_' +
        data_copy['setting2'].round(0).astype(str) + '_' +
        data_copy['setting3'].round(0).astype(str)
    )

    # 逐行应用
    for oc, scaler in scalers.items():
        indices = data_copy[data_copy['oc_key'] == oc].index
        if not indices.empty:
            data_copy.loc[indices, feature_cols] = scaler.transform(
                data_copy.loc[indices, feature_cols]
            )
    return data_copy

def create_sequences(df, seq_length, features_cols, label_col):
    data_list = []
    label_list = []
    for unit_id in df['ID'].unique():
        unit_df = df[df['ID'] == unit_id]
        features = unit_df[features_cols].values
        labels = unit_df[label_col].values
        for i in range(len(features) - seq_length + 1):
            data_list.append(features[i : i + seq_length])
            label_list.append(labels[i + seq_length - 1])
    return np.array(data_list), np.array(label_list)

def create_test_sequences(df, seq_length, features_cols):
    data_list = []
    for unit_id in df['ID'].unique():
        unit_df = df[df['ID'] == unit_id]
        features = unit_df[features_cols].values
        if len(features) >= seq_length:
            data_list.append(features[-seq_length:])
        else:
            padding = np.zeros((seq_length - len(features), len(features_cols)))
            data_list.append(np.concatenate((padding, features)))
    return np.array(data_list)

def get_dataloaders(dataset_name, data_dir, scaler_save_dir):
    """
    一个主函数，完成从加载到生成 DataLoader 的所有步骤
    """
    print(f"--- 正在处理数据集: {dataset_name} ---")
    train_df, test_df, rul_df = load_data(data_dir, dataset_name)
    
    # 1. RUL 计算
    train_df = calculate_rul(train_df)
    rul_df['RUL'] = rul_df['RUL'].clip(upper=RUL_CAP)
    
    # 2. 特征选择
    feature_cols = perform_feature_selection(train_df)
    
    # 3. 归一化 (!! 关键 !!)
    if dataset_name in ['FD002', 'FD004']:
        # ... (执行分工况归一化逻辑) ...
        oc_scalers = create_oc_scalers(train_df, feature_cols)
        train_df = transform_per_oc(train_df, oc_scalers, feature_cols)
        test_df = transform_per_oc(test_df, oc_scalers, feature_cols)
        
        # 保存 scalers
        scaler_path = os.path.join(scaler_save_dir, f"{dataset_name}_oc_scalers.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(oc_scalers, f)
        print(f"为 {dataset_name} 创建了 {len(oc_scalers)} 个不同的归一化器。")
        print(f"归一化器已保存到: {scaler_path}")
    else:
        # ... (执行单一归一化逻辑) ...
        scaler = MinMaxScaler()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])
        # !! 保存 Scaler !!
        scaler_path = os.path.join(scaler_save_dir, f"{dataset_name}_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"单一归一化器已保存到: {scaler_path}")
        
    # 4. 序列生成
    print(f"\n--- 4.1. 生成序列 (窗口长度={SEQUENCE_LENGTH})... ---")
    X_train_np, y_train_np = create_sequences(train_df, SEQUENCE_LENGTH, feature_cols, 'RUL')
    X_test_np = create_test_sequences(test_df, SEQUENCE_LENGTH, feature_cols)
    y_test_np = rul_df['RUL'].values
    print(f"X_train 形状: {X_train_np.shape}")
    print(f"y_train 形状: {y_train_np.shape}")
    print(f"X_test 形状: {X_test_np.shape}")
    print(f"y_test 形状: {y_test_np.shape}")
    
    # 4.2. 生成序列 (窗口长度=1)
    # 你需要在这里生成 X_train_np, y_train_np, X_test_np, y_test_np
    # (此处省略以保持简洁, 假设我们拿到了 NumPy 数组)
    # X_train_np, y_train_np = np.random.rand(1000, 30, len(feature_cols)), np.random.rand(1000)
    # X_test_np, y_test_np = np.random.rand(100, 30, len(feature_cols)), np.random.rand(100)
    
    # 5. 划分训练/验证集
    X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(X_train_np, y_train_np, test_size=0.2, random_state=42)
    
    # 6. 创建 Tensors
    X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val_np, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_np, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test_np, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_np, dtype=torch.float32).view(-1, 1)

    # 7. 创建 DataLoaders
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"训练批次: {len(train_loader)}")
    print(f"验证批次: {len(val_loader)}")
    print(f"测试批次: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, len(feature_cols)