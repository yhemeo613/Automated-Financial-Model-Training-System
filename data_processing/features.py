import pandas as pd
import numpy as np
from ta.trend import MACD, SMAIndicator
from ta.momentum import RSIIndicator

class FeatureEngineer:
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame):
        """
        添加技术指标：MA, RSI, MACD
        """
        df_feat = df.copy()
        
        # 确保 close 列存在且为数值型
        close = df_feat['close']
        
        # 移动平均线 (MA)
        df_feat['ma_7'] = SMAIndicator(close, window=7).sma_indicator()
        df_feat['ma_25'] = SMAIndicator(close, window=25).sma_indicator()
        
        # 相对强弱指数 (RSI)
        df_feat['rsi_14'] = RSIIndicator(close, window=14).rsi()
        
        # MACD
        macd = MACD(close)
        df_feat['macd'] = macd.macd()
        df_feat['macd_signal'] = macd.macd_signal()
        df_feat['macd_diff'] = macd.macd_diff()
        
        # 清除由于计算指标产生的NaN (前几行)
        df_feat = df_feat.dropna()
        
        return df_feat

    @staticmethod
    def add_sentiment_score(df: pd.DataFrame):
        """
        添加情绪因子 (模拟)
        实际应用中应连接NLP分析模块
        """
        df_feat = df.copy()
        # 模拟一个 -1 到 1 之间的随机情绪分数
        np.random.seed(42)
        df_feat['sentiment'] = np.random.uniform(-1, 1, size=len(df_feat))
        return df_feat

    @staticmethod
    def create_sequences(data: np.ndarray, seq_length: int):
        """
        创建时间序列窗口
        :param data: 特征数据 (numpy array)
        :param seq_length: 序列长度 (窗口大小)
        :return: X (样本数, 序列长度, 特征数), y (样本数, 目标值)
        注意：默认假设最后一列是目标值，或者需要外部指定。
        这里简单实现：预测下一时刻的 Close 价格（假设Close在某一列）
        为了通用性，仅返回滑动窗口，具体X, y拆分在模型训练时进行
        """
        xs = []
        ys = []
        # 假设我们预测的是 Close 价格，且 Close 是第4列 (0:timestamp, 1:open, 2:high, 3:low, 4:close)
        # 但传入的 data 可能是处理后的 DataFrame.values，列顺序不确定。
        # 因此，建议传入 DataFrame，或者假定最后一列是 target。
        # 这里仅做切片。
        pass
    
    @staticmethod
    def create_xy_windows(df: pd.DataFrame, target_col='close', window_size=60, prediction_horizon=1):
        """
        创建 (X, y) 用于监督学习
        :param df: 包含特征和目标的DataFrame
        :param target_col: 目标列名
        :param window_size: 输入时间步长
        :param prediction_horizon: 预测未来第几个时间步
        """
        data = df.values
        target_idx = df.columns.get_loc(target_col)
        
        X, y = [], []
        for i in range(len(data) - window_size - prediction_horizon + 1):
            X.append(data[i:(i + window_size)])
            y.append(data[i + window_size + prediction_horizon - 1, target_idx])
            
        return np.array(X), np.array(y)
