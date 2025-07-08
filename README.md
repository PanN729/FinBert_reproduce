# FinBERT复现与创新探索项目

## 项目简介

本项目基于ACL 2021年论文《FinBERT: A Pretrained Language Model for Financial Communications》进行了完整的代码复现。FinBERT是一种针对金融领域文本情感分析任务优化的预训练语言模型，旨在提升金融新闻、公告等文本的情感分类准确率。

在复现基础上，我们进一步设计了结合金融实体识别和事件特征融合的创新方案，对模型性能进行了有效提升。

## 目录结构

# FinBERT复现与创新探索项目

FinBERT-Reproduce/
├── data/ # 数据集存放目录
│ └── Sentences_50Agree.txt # Financial PhraseBank数据集文本
├── preprocess.py # 数据预处理脚本
├── finbert_sentiment.py # 模型训练主程序
├── train_utils.py # 训练与评估辅助函数
├── requirements.txt # 环境依赖列表
├── README.md # 项目说明文档（此文件）
└── results/ # 实验结果和日志文件


## 环境依赖

建议使用Python 3.10+，并安装如下依赖：

```bash
pip install -r requirements.txt
主要依赖包括：

transformers==4.40.0

torch==2.1.2

scikit-learn==1.3.2

pandas==2.1.4

tqdm==4.66.2

numpy==1.26.4
