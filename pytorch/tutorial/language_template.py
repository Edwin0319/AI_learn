# -*- coding: utf-8 -*-
"""
自然语言处理全流程模板（文本分类任务）
包含：数据预处理、模型定义、训练优化、测试评估
"""
import os
import re
import time
import jieba  # 中文分词
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


# --------------------- 配置参数 ---------------------
class Config:
    # 数据参数
    data_path = "./data/text_data.csv"  # CSV格式：text列，label列
    text_col = "text"
    label_col = "label"
    max_seq_len = 50  # 文本最大长度
    vocab_size = 10000  # 词汇表大小

    # 模型参数
    embedding_dim = 256
    hidden_dim = 128
    num_classes = 10  # 分类类别数
    dropout = 0.2

    # 训练参数
    batch_size = 64
    learning_rate = 1e-3
    num_epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = Config()


# --------------------- 数据预处理 ---------------------
class TextProcessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = jieba.lcut if self.is_chinese() else self.english_tokenize
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.label_encoder = LabelEncoder()

    def is_chinese(self):
        """检测是否为中文数据（简单实现）"""
        sample_text = pd.read_csv(config.data_path)[config.text_col][0]
        return re.search(r'[\u4e00-\u9fff]', sample_text) is not None

    def english_tokenize(self, text):
        """英文分词处理"""
        text = re.sub(r'[^\w\s]', '', text.lower().strip())
        return text.split()

    def build_vocab(self, texts):
        """构建词汇表"""
        word_counts = {}
        for text in texts:
            words = self.tokenizer(text)
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        # 取最高频词汇
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        top_words = [word for word, _ in sorted_words[:self.config.vocab_size - 2]]
        # 更新word2idx
        for idx, word in enumerate(top_words, start=2):
            self.word2idx[word] = idx

    def text_to_indices(self, text):
        """文本转索引序列"""
        words = self.tokenizer(text)
        indices = [self.word2idx.get(word, 1) for word in words]  # 未登录词用UNK
        # 截断或填充
        if len(indices) > self.config.max_seq_len:
            indices = indices[:self.config.max_seq_len]
        else:
            indices += [0] * (self.config.max_seq_len - len(indices))
        return indices

    def process_data(self):
        """完整数据处理流程"""
        # 读取原始数据
        df = pd.read_csv(self.config.data_path)
        texts = df[self.config.text_col].fillna("").tolist()
        labels = df[self.config.label_col].values

        # 构建词汇表
        self.build_vocab(texts)

        # 标签编码
        self.label_encoder.fit(labels)
        encoded_labels = self.label_encoder.transform(labels)

        # 文本转索引
        text_indices = [self.text_to_indices(text) for text in texts]

        return np.array(text_indices), encoded_labels


# --------------------- 数据集类 ---------------------
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.texts[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


# --------------------- 模型定义 ---------------------
class TextClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.rnn = nn.LSTM(config.embedding_dim, config.hidden_dim,
                           batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden_dim * 2, config.num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, emb_dim)
        rnn_out, _ = self.rnn(embedded)  # (batch, seq_len, hidden_dim*2)
        # 取最后时刻的输出
        last_out = rnn_out[:, -1, :]
        output = self.fc(self.dropout(last_out))
        return output


# --------------------- 训练流程 ---------------------
def train_model(model, train_loader, val_loader, config):
    model = model.to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_acc = 0.0
    for epoch in range(config.num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0.0
        start_time = time.time()
        for texts, labels in train_loader:
            texts, labels = texts.to(config.device), labels.to(config.device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证阶段
        val_acc = evaluate(model, val_loader, config)
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

        # 打印信息
        time_cost = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{config.num_epochs}] | "
              f"Train Loss: {total_loss / len(train_loader):.4f} | "
              f"Val Acc: {val_acc:.4f} | Time: {time_cost:.1f}s")


# --------------------- 评估函数 ---------------------
def evaluate(model, data_loader, config):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for texts, labels in data_loader:
            texts = texts.to(config.device)
            outputs = model(texts)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return accuracy_score(all_labels, all_preds)


# --------------------- 测试流程 ---------------------
def test_model(model, test_loader, config, label_encoder):
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for texts, labels in test_loader:
            texts = texts.to(config.device)
            outputs = model(texts)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # 解码标签
    true_labels = label_encoder.inverse_transform(all_labels)
    pred_labels = label_encoder.inverse_transform(all_preds)

    # 输出报告
    print("\nTest Report:")
    print(classification_report(true_labels, pred_labels))
    print(f"Final Accuracy: {accuracy_score(true_labels, pred_labels):.4f}")


# --------------------- 主函数 ---------------------
if __name__ == "__main__":
    # 数据预处理
    processor = TextProcessor(config)
    text_indices, labels = processor.process_data()

    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(
        text_indices, labels, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # 创建DataLoader
    train_loader = DataLoader(
        TextDataset(X_train, y_train),
        batch_size=config.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TextDataset(X_val, y_val),
        batch_size=config.batch_size
    )
    test_loader = DataLoader(
        TextDataset(X_test, y_test),
        batch_size=config.batch_size
    )

    # 初始化模型
    model = TextClassifier(config)

    # 训练模型
    train_model(model, train_loader, val_loader, config)

    # 测试评估
    test_model(model, test_loader, config, processor.label_encoder)

    # 样例预测
    test_samples = ["这家餐厅的菜品非常美味，服务也很周到！",
                    "产品质量很差，完全不值得购买。"]
    model.eval()
    with torch.no_grad():
        for text in test_samples:
            indices = processor.text_to_indices(text)
            tensor = torch.tensor(indices).unsqueeze(0).to(config.device)
            output = model(tensor)
            pred_label = processor.label_encoder.inverse_transform(
                [torch.argmax(output).item()]
            )
            print(f"\nText: {text} \nPredicted Label: {pred_label[0]}")