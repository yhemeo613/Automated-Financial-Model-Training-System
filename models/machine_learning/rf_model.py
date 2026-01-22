'''
Author: dzy dzyperson@163.com
Date: 2026-01-22 14:02:23
LastEditors: dzy dzyperson@163.com
LastEditTime: 2026-01-22 14:33:11
FilePath: \模型训练\models\machine_learning\rf_model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
from models.base import BaseModel
import numpy as np

class RandomForestModel(BaseModel):
    def __init__(self, task_type='regression', n_estimators=100, max_depth=None, random_state=42):
        self.task_type = task_type
        if task_type == 'regression':
            self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        else:
            self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
            
    def train(self, X_train, y_train, X_val=None, y_val=None, progress_callback=None):
        # 随机森林不接受3D输入 (samples, time_steps, features)
        # 如果输入是3D的，需要展平或取最后一个时间步
        if len(X_train.shape) == 3:
            # 展平策略：samples, time_steps * features
            nsamples, nx, ny = X_train.shape
            X_train = X_train.reshape((nsamples, nx*ny))
            if X_val is not None:
                nsamples_val, nx_val, ny_val = X_val.shape
                X_val = X_val.reshape((nsamples_val, nx_val*ny_val))
        
        if progress_callback:
            progress_callback(10, "数据预处理完成，开始训练...")
            
        print(f"开始训练随机森林 ({self.task_type})...")
        self.model.fit(X_train, y_train)
        
        if progress_callback:
            progress_callback(90, "模型拟合完成，正在评估...")
        
        train_pred = self.model.predict(X_train)
        if self.task_type == 'regression':
            loss = mean_squared_error(y_train, train_pred)
            print(f"Train MSE: {loss:.4f}")
        else:
            acc = accuracy_score(y_train, train_pred)
            print(f"Train Accuracy: {acc:.4f}")

        if X_val is not None:
            val_pred = self.model.predict(X_val)
            if self.task_type == 'regression':
                val_loss = mean_squared_error(y_val, val_pred)
                print(f"Val MSE: {val_loss:.4f}")
            else:
                val_acc = accuracy_score(y_val, val_pred)
                print(f"Val Accuracy: {val_acc:.4f}")
                
        if progress_callback:
            progress_callback(100, "训练流程结束")

    def predict(self, X):
        if len(X.shape) == 3:
            nsamples, nx, ny = X.shape
            X = X.reshape((nsamples, nx*ny))
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)
        print(f"模型已保存至 {path}")

    def load(self, path):
        self.model = joblib.load(path)
        print(f"模型已从 {path} 加载")
