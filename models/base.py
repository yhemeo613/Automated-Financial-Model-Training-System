'''
Author: dzy dzyperson@163.com
Date: 2026-01-22 14:01:54
LastEditors: dzy dzyperson@163.com
LastEditTime: 2026-01-22 14:33:42
FilePath: \模型训练\models\base.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseModel(ABC):
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None, progress_callback=None):
        """训练模型"""
        pass

    @abstractmethod
    def predict(self, X):
        """预测"""
        pass

    @abstractmethod
    def save(self, path):
        """保存模型"""
        pass

    @abstractmethod
    def load(self, path):
        """加载模型"""
        pass
