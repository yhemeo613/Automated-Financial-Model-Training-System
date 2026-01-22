import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.base import BaseModel
import os

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_prob=0.0):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 增加 Dropout 层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # 取最后一个时间步的输出
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class LSTMModel(BaseModel):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=2, lr=0.001, epochs=50, dropout=0.2, weight_decay=0.01):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMNet(input_dim, hidden_dim, output_dim, num_layers, dropout_prob=dropout).to(self.device)
        self.criterion = nn.MSELoss()
        # 引入 L2 正则化 (weight_decay)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.epochs = epochs
        self.input_dim = input_dim # 记录输入维度以便重置模型
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
    def train(self, X_train, y_train, X_val=None, y_val=None, progress_callback=None):
        self.model.train()
        
        # 转换为 Tensor
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)
        
        if X_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(self.device)

        print(f"开始在 {self.device} 上训练 LSTM...")
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            loss = self.criterion(outputs, y_train_tensor)
            loss.backward()
            self.optimizer.step()
            
            # 计算进度
            progress = int((epoch + 1) / self.epochs * 100)
            msg = f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}"
            
            if (epoch+1) % 10 == 0:
                val_loss_str = ""
                if X_val is not None:
                    self.model.eval()
                    with torch.no_grad():
                        val_outputs = self.model(X_val_tensor)
                        val_loss = self.criterion(val_outputs, y_val_tensor)
                        val_loss_str = f", Val Loss: {val_loss.item():.4f}"
                    self.model.train()
                print(msg + val_loss_str)
            
            # 调用回调函数
            if progress_callback:
                progress_callback(progress, msg)
                
    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return predictions.cpu().numpy()

    def predict_with_uncertainty(self, X, n_iter=50):
        """
        使用 MC Dropout 计算预测值的不确定性
        """
        self.model.train() # 保持 Dropout 开启
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_iter):
                pred = self.model(X_tensor)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions) # shape: (n_iter, samples, 1)
        
        # 计算均值和标准差
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"模型已保存至 {path}")

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        print(f"模型已从 {path} 加载")
