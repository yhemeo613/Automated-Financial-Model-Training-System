'''
Author: dzy dzyperson@163.com
Date: 2026-01-22 14:02:31
LastEditors: dzy dzyperson@163.com
LastEditTime: 2026-01-22 14:55:11
FilePath: \模型训练\models\selector.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from models.deep_learning.lstm_model import LSTMModel
from models.machine_learning.rf_model import RandomForestModel
from models.ensemble import EnsembleModel

class ModelSelector:
    @staticmethod
    def get_model(task_type, model_name, **kwargs):
        """
        根据任务类型和模型名称获取模型实例
        :param task_type: 'prediction' (预测), 'risk_control' (风控), 'advisory' (投顾)
        :param model_name: 'lstm', 'rf', 'ensemble', 'transformer' 等
        """
        # 过滤 kwargs，移除不适用于 RF 的参数 (如 input_dim, epochs, lr, dropout)
        rf_kwargs = {k: v for k, v in kwargs.items() if k not in ['input_dim', 'epochs', 'lr', 'dropout', 'weight_decay']}
        
        if task_type == 'prediction':
            if model_name == 'lstm':
                # LSTM 不接受 n_estimators
                lstm_kwargs = {k: v for k, v in kwargs.items() if k not in ['n_estimators']}
                return LSTMModel(**lstm_kwargs)
            elif model_name == 'rf':
                 return RandomForestModel(task_type='regression', **rf_kwargs)
            elif model_name == 'ensemble':
                 # 使用 kwargs 中的 input_dim
                 input_dim = kwargs.get('input_dim', 1)
                 return EnsembleModel(input_dim=input_dim)
        
        elif task_type == 'risk_control':
            if model_name == 'rf':
                return RandomForestModel(task_type='classification', **rf_kwargs)
            elif model_name == 'ensemble':
                 input_dim = kwargs.get('input_dim', 1)
                 return EnsembleModel(input_dim=input_dim) # 风险控制也可以用集成回归预测
        
        # 默认返回随机森林回归
        print(f"未找到匹配模型 {model_name} for {task_type}，默认使用 RandomForestRegressor")
        return RandomForestModel(task_type='regression', **rf_kwargs)
