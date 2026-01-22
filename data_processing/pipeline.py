from data_processing.cleaning import DataCleaner
from data_processing.features import FeatureEngineer
import pandas as pd

class DataPipeline:
    def __init__(self):
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()

    def run(self, df: pd.DataFrame):
        """
        运行数据处理流水线
        """
        print("开始数据清洗...")
        # 1. 缺失值处理
        df = self.cleaner.fill_missing_values(df)
        
        # 2. 异常值处理 (针对价格和成交量)
        # 注意：对于金融时间序列，价格剧烈波动可能不是异常，需谨慎处理
        # 这里仅对 volume 做处理作为示例
        df = self.cleaner.remove_outliers_iqr(df, columns=['volume'])
        
        print("开始特征工程...")
        # 3. 技术指标
        df = self.engineer.add_technical_indicators(df)
        
        # 4. 情绪因子
        df = self.engineer.add_sentiment_score(df)
        
        # 5. 波动率特征
        df = self.engineer.add_volatility_features(df)
        
        # 6. 微观结构特征
        df = self.engineer.add_microstructure_features(df)
        
        # 7. 清除最终的NaN值
        df = df.dropna()
        
        print(f"数据处理完成，最终维度: {df.shape}")
        return df
