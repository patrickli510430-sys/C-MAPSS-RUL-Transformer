RUL-Transformer: 基于 PyTorch 和 Transformer 的涡轮风扇发动机剩余使用寿命预测 (C-MAPSS 数据集)

1. 项目简介 (Abstract)

本项目致力于解决预测性维护 (PdM) 领域的核心问题：剩余使用寿命 (RUL) 预测。我们使用 NASA 著名的 C-MAPSS (涡轮风扇发动机退化模拟) 数据集，实现了一个先进的深度学习模型。

与传统的 RNN/LSTM 方法不同，本项目采用了一种混合模型架构 (Conv1D + Transformer)，旨在同时捕捉时间序列数据中的局部特征（通过1D卷积层平滑噪音、提取模式）和全局依赖关系（通过 Transformer 编码器）。

该模型在模块化的 Python 脚本中实现，易于训练、评估和部署，并为 C-MAPSS 数据集中复杂的多工况 (Multi-Operating Condition) 和多故障 (Multi-Fault) 场景提供了强大的解决方案。

2. 背景 (Background)

在航空、制造和能源等重资产行业，预测性维护 (PdM) 是降低运营成本、提高安全性的关键技术。其核心是准确预测部件的 RUL (Remaining Useful Life)，即从当前时间点到发生功能故障的剩余时间。

C-MAPSS 数据集 是 RUL 预测领域最权威的基准测试之一。它由 NASA 提供，模拟了不同工况和故障模式下的涡轮风扇发动机传感器数据。该数据集分为四个子集 (FD001, FD002, FD003, FD004)，其复杂性递增：

FD001: 单一工况，单一故障 (HPC 退化)

FD002: 六种工况，单一故障

FD003: 单一工况，两种故障 (HPC 和风扇退化)

FD004: 六种工况，两种故障 (难度最高)

本项目旨在为所有四个子集建立精确的 RUL 预测模型。

3. 方法 (Methodology)

我们的方法论分为两个核心部分：健壮的数据预处理和先进的模型架构。

3.1 数据预处理

C-MAPSS 的数据预处理至关重要，我们的流程 (src/data_loader.py) 包含以下关键步骤：

RUL 标签生成: 对训练集，RUL 被计算为 RUL = max_cycle - current_cycle。为了防止模型在早期健康阶段过拟合，RUL 被设置为一个上限值（Piecewise Linear Degradation Model），本项目中 RUL_CAP = 125。

特征选择: 并非所有传感器都有用。我们通过计算每个传感器在训练集上的方差来自动移除常量或近常量（variance < 1e-5）的传感器，同时始终保留 setting1, setting2, setting3 作为工况特征。

分工况归一化 (Multi-OC Normalization):

对于 FD001 和 FD003（单工况），我们使用一个标准的 MinMaxScaler 对所有数据进行归一化。

对于 FD002 和 FD004（多工况），使用单一归一化器会扭曲数据。我们为每一种操作工况（由3个 setting 列定义）拟合一个单独的 MinMaxScaler。在转换数据时，每一行都根据其所属的工况应用对应的归一化器。

序列构造 (Windowing): 我们使用一个大小为 SEQUENCE_LENGTH（例如 30）的滑动窗口，将时间序列数据转换为 (样本数, 序列长度, 特征数) 的监督学习样本，标签为窗口末端的 RUL 值。

3.2 模型架构 (Conv1D + Transformer)

我们的模型 (src/model.py) 采用了一个混合架构，以充分利用两种模型的优势：

1D 卷积层 (Conv1D): 作为模型的第一层，输入序列 (B, T, F) 首先通过一个 1D 卷积层。

作用: 它充当一个可学习的特征提取器和降噪器，通过一个小型感受野（例如 kernel_size=3）平滑传感器噪音，并提取局部的、短期的模式。

投影: 它还将原始的低维特征（例如 18 维）投影到模型的高维工作空间（例如 128 维）。

Transformer 编码器 (Transformer Encoder): 卷积层的输出被送入一个标准的 Transformer 编码器（仅使用 Encoder，无 Decoder）。

作用: Transformer 的自注意力机制使其能够捕捉序列内的长期和全局依赖关系。例如，它能理解第3个周期和第27个周期的传感器读数如何共同影响 RUL，这是 LSTM 难以实现的。

回归头 (Regression Head): Transformer 的输出（在时间维度上取平均值 mean pooling）被送入一个简单的多层感知机（MLP），最终输出一个单一的 RUL 预测值。

这个架构使得模型既能抵抗局部噪音，又能理解全局退化趋势。

4. 结果 (Results)

我们对每个子数据集都独立地训练和评估了模型。

可视化结果 (FD001)

模型在 FD001 上的训练曲线和最终评估结果如下所示。

训练历史 (Loss vs Val Loss)

真实 RUL vs 预测 RUL (评估)

<img src="saved_models/FD001_training_history.png" alt="训练历史图" width="400">

<img src="saved_models/FD001_evaluation_plot.png" alt="评估结果图" width="400">

从左图可以看出，训练和验证损失稳定下降。从右图可以看出，预测值（蓝点）紧密地聚集在理想线（红线）周围，特别是在 RUL < 60 的关键预警区域，证明了模型的高精度。



5. 如何运行 (Installation & Usage)

5.1 项目结构

C-MAPSS_RUL_Project/
|-- data/
|   |-- CMaps/                 # (存放 .txt 数据文件)
|-- saved_models/              # (存放训练好的 .pth 和 .pkl 文件)
|-- src/
|   |-- config.py              # (所有配置和超参数)
|   |-- model.py               # (TransformerModel 类定义)
|   |-- data_loader.py         # (数据预处理和加载)
|   |-- utils.py               # (辅助函数)
|-- train.py                   # (主训练脚本)
|-- evaluate.py                # (主评估脚本)
|-- predict.py                 # (主预测脚本)
|-- requirements.txt           # (项目依赖)
|-- README.md                  # (本项目文档)


5.2 安装

克隆仓库

git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
cd C-MAPSS_RUL_Project


安装依赖

pip install -r requirements.txt


5.3 数据准备

从 NASA 官网下载 C-MAPSS 数据集。

将所有 .txt 文件 (train_FD00x.txt, test_FD00x.txt, RUL_FD00x.txt) 放入 data/CMaps/ 目录下。

5.4 训练

使用 train.py 脚本进行训练。您必须通过 --dataset 参数指定要训练的数据集。

# 训练 FD001 数据集
python train.py --dataset FD001

# 训练 FD004 数据集 (将使用不同的归一化逻辑和模型参数)
python train.py --dataset FD004


训练脚本将自动：

加载并预处理指定的数据集。

保存 scaler.pkl（或 oc_scalers.pkl）到 saved_models/。

训练模型，并使用早停。

保存最佳的 _model.pth 和 _training_history.png 到 saved_models/。

5.5 评估

训练完成后，使用 evaluate.py 在完整的测试集上报告最终性能。

# 评估 FD001 模型的最终性能
python evaluate.py --dataset FD001


评估脚本将：

加载 FD001_model.pth 和 FD001_scaler.pkl。

加载 test_FD001.txt 和 RUL_FD001.txt。

计算并打印最终的 RMSE 和 C-MAPSS 评分。

保存 FD001_evaluation_plot.png 到 saved_models/。

5.6 预测

使用 predict.py 加载模型，对一个（模拟的）新数据样本进行预测。

python predict.py --dataset FD001
📖 项目简介 (Overview)本项目旨在利用深度学习技术解决工业领域的关键问题——预测设备的剩余使用寿命（RUL）。我们专注于 NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) 涡扇发动机退化仿真数据集。准确的 RUL 预测对于实现预测性维护（Predictive Maintenance）至关重要，它可以帮助减少意外停机时间、降低维护成本并提高航空安全性。目前，该项目采用了一种混合模型架构，结合了 卷积神经网络 (CNN) 和 长短期记忆网络 (LSTM)。CNN 用于从多传感器数据中自动提取空间特征，而 LSTM 则擅长捕捉时间序列数据中的长期依赖关系和退化趋势。⚙️ 系统流程图 (System Workflow)下图展示了从原始数据输入到最终 RUL 预测的完整处理流程。代码段graph TD
    %% 定义样式
    classDef data fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#01579b;
    classDef process fill:#fff3e0,stroke:#ff9800,stroke-width:2px,color:#e65100;
    classDef model fill:#e8f5e9,stroke:#43a047,stroke-width:2px,color:#1b5e20;
    classDef output fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#880e4f;

    %% 数据准备阶段
    subgraph Data Preparation
        A[原始 C-MAPSS 数据集\n(FD001-FD004)]:::data --> B(数据清洗与探索性分析);
        B --> C(特征选择与工程);
        C --> D(数据标准化/归一化);
        D --> E[滑动时间窗切片\n(Sliding Window)];
    end

    %% 模型构建阶段
    subgraph Hybrid Model Architecture
        E --> F(CNN 层\n特征提取):::model;
        F --> G(LSTM 层\n时序建模):::model;
        G --> H(全连接层/Dense Layer):::model;
    end

    %% 训练与评估阶段
    subgraph Training & Eval
        H --> I{训练/推理?};
        I -- Training --> J(计算 Loss & 优化器更新);
        I -- Inference --> K(加载预训练权重\nmodel_weights.pth);
        J --> H;
        K --> H;
    end

    %% 输出阶段
    H --> L[最终 RUL 预测值]:::output;
    L --> M(性能评估\nRMSE, Score):::output;

    %% 连接线样式
    linkStyle default stroke:#607d8b,stroke-width:1px;
📂 项目结构 (Project Structure)项目的核心文件组织结构如下：BashC-MAPSS_RUL_Project/
├── CMAPSSData/          # [必要] 存放 NASA C-MAPSS 原始数据集的文件夹
├── C-MAPSS.ipynb        # [核心] 主 Jupyter Notebook，包含数据处理、模型训练和评估代码
├── model_weights.pth    # [产出] 训练好的模型权重文件
├── README.md            # 项目说明文档
└── requirements.txt     # (建议添加) 项目依赖包列表
🛠️ 环境依赖 (Prerequisites)为了顺利运行本项目，请确保您的环境满足以下依赖库版本要求。PackageVersion Requirement用途python3.8+ (Recommended)编程语言numpy1.24.3数值计算与数组操作pandas2.0.3数据处理与分析matplotlib3.7.2数据可视化torch (PyTorch)2.0.1深度学习框架 (后端)scikit-learn1.3.0数据预处理与评估指标建议: 强烈建议使用 conda 或 venv 创建独立的虚拟环境来管理这些依赖。🚀 快速开始 (Getting Started)1. 克隆项目Bashgit clone https://github.com/yourusername/C-MAPSS_RUL_Project.git
cd C-MAPSS_RUL_Project
2. 准备数据确保您已下载 NASA C-MAPSS 数据集，并将相关 txt 文件解压到项目根目录下的 CMAPSSData/ 文件夹中。3. 安装依赖如果项目包含 requirements.txt (推荐创建)，运行：Bashpip install -r requirements.txt
否则，请手动安装上述列出的指定版本依赖。4. 运行代码使用 Jupyter Notebook 或 JupyterLab 打开主文件：Bashjupyter notebook C-MAPSS.ipynb
按照 Notebook 中的单元格顺序逐步执行，即可完成数据加载、预处理、模型训练及评估。5. 使用预训练模型如果您想跳过训练过程，直接使用已保存的权重进行推理，请确保 model_weights.pth 文件存在，并在 Notebook 中执行加载权重的相关代码段。🧠 模型与方法 (Model Methodology)当前实现采用 CNN-LSTM 混合架构：CNN (一维卷积): 沿传感器维度滑动，捕捉不同传感器读数之间的局部空间相关性，提取鲁棒的特征。LSTM: 处理 CNN 提取的特征序列，学习时间维度上的长期退化模式。🗺️ 路线图与未来展望 (Roadmap)项目的后续开发计划如下：[x] 完成基础数据预处理流程（滑动窗口、标准化）。[x] 实现并验证 CNN-LSTM 基准模型。[ ] 探索 Transformer 架构: 利用自注意力机制（Self-Attention）捕捉更长距离的依赖关系，尝试提升预测精度。[ ] 研究图神经网络 (GNN): 将传感器网络建模为图结构，探索传感器节点间的复杂拓扑关系对 RUL 的影响。[ ] 完善模型评估部分，增加更多可视化图表（如预测值 vs 真实值对比图）。🤝 贡献 (Contributing)欢迎任何形式的贡献！如果您有改进建议、发现了 Bug，或者想添加新的模型实现（如 Transformer/GNN），请随时提交 Pull Request 或创建 Issue。📄 许可证 (License)
