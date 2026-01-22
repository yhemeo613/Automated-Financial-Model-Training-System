import pandas as pd
import numpy as np

class DataCleaner:
    @staticmethod
    def remove_outliers_iqr(df: pd.DataFrame, columns: list, factor=1.5):
        """
        使用IQR方法处理异常值
        :param df: 数据DataFrame
        :param columns: 需要处理的列名列表
        :param factor: IQR系数，通常为1.5
        :return: 处理后的DataFrame
        """
        df_clean = df.copy()
        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            # 将异常值替换为边界值（Winsorization）或者直接删除
            # 这里选择替换为边界值，以保持时间序列连续性
            df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
            df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
            
        return df_clean

    @staticmethod
    def remove_outliers_sigma(df: pd.DataFrame, columns: list, sigma=3):
        """
        使用3σ法则处理异常值
        """
        df_clean = df.copy()
        for col in columns:
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            lower_bound = mean - sigma * std
            upper_bound = mean + sigma * std
            
            df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
            df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
            
        return df_clean

    @staticmethod
    def fill_missing_values(df: pd.DataFrame, method='interpolate'):
        """
        处理缺失值
        :param method: 'interpolate' (线性插值) 或 'ffill' (前向填充)
        """
        df_clean = df.copy()
        if method == 'interpolate':
            df_clean = df_clean.interpolate(method='linear')
        elif method == 'ffill':
            df_clean = df_clean.ffill().bfill()
        
        return df_clean
