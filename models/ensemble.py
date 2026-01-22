import numpy as np
import os
from models.base import BaseModel
from models.deep_learning.lstm_model import LSTMModel
from models.machine_learning.rf_model import RandomForestModel

class EnsembleModel(BaseModel):
    def __init__(self, input_dim, model_configs=None):
        """
        集成模型：融合 LSTM 和 RF
        """
        self.input_dim = input_dim
        self.models = []
        self.weights = []
        
        # 默认配置
        if model_configs is None:
            self.model_configs = [
                {'type': 'lstm', 'weight': 0.6, 'params': {'epochs': 50, 'dropout': 0.2}},
                {'type': 'rf', 'weight': 0.4, 'params': {'n_estimators': 100}}
            ]
        else:
            self.model_configs = model_configs
            
        self._init_models()
        
    def _init_models(self):
        self.models = []
        self.weights = []
        for config in self.model_configs:
            if config['type'] == 'lstm':
                model = LSTMModel(input_dim=self.input_dim, **config.get('params', {}))
                self.models.append(model)
            elif config['type'] == 'rf':
                model = RandomForestModel(task_type='regression', **config.get('params', {}))
                self.models.append(model)
            self.weights.append(config['weight'])
            
        # 归一化权重
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
    def train(self, X_train, y_train, X_val=None, y_val=None, progress_callback=None):
        total_models = len(self.models)
        for i, model in enumerate(self.models):
            model_name = self.model_configs[i]['type'].upper()
            print(f"正在训练集成子模型 {i+1}/{total_models}: {model_name}...")
            
            if progress_callback:
                progress_callback(int((i / total_models) * 100), f"正在训练子模型: {model_name}")
                
            # 简化：RF 不需要 val 集，但传入也无妨
            model.train(X_train, y_train, X_val, y_val)
            
    def predict(self, X):
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            # 确保 pred 是 (N, 1) 形状
            if len(pred.shape) == 1:
                pred = pred.reshape(-1, 1)
            predictions.append(pred)
            
        # 加权平均
        final_pred = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            final_pred += pred * self.weights[i]
            
        return final_pred
        
    def predict_with_uncertainty(self, X):
        """
        通过集成模型的方差来估计不确定性
        """
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            if len(pred.shape) == 1:
                pred = pred.reshape(-1, 1)
            predictions.append(pred)
            
        predictions = np.array(predictions) # shape: (n_models, samples, 1)
        
        mean_pred = np.average(predictions, axis=0, weights=self.weights)
        # 计算加权标准差作为不确定性度量
        variance = np.average((predictions - mean_pred)**2, axis=0, weights=self.weights)
        std_pred = np.sqrt(variance)
        
        return mean_pred, std_pred

    def save(self, path):
        # 保存每个子模型
        base_dir = os.path.dirname(path)
        base_name = os.path.basename(path).split('.')[0]
        
        for i, model in enumerate(self.models):
            model_type = self.model_configs[i]['type']
            ext = ".pth" if model_type == 'lstm' else ".pkl"
            sub_path = os.path.join(base_dir, f"{base_name}_{model_type}_{i}{ext}")
            model.save(sub_path)
            
    def load(self, path):
        # 加载每个子模型
        base_dir = os.path.dirname(path)
        base_name = os.path.basename(path).split('.')[0]
        
        for i, model in enumerate(self.models):
            model_type = self.model_configs[i]['type']
            ext = ".pth" if model_type == 'lstm' else ".pkl"
            sub_path = os.path.join(base_dir, f"{base_name}_{model_type}_{i}{ext}")
            if os.path.exists(sub_path):
                model.load(sub_path)
            else:
                print(f"警告: 子模型文件 {sub_path} 不存在")
