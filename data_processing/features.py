import pandas as pd
import numpy as np
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, VolumePriceTrendIndicator
from ta.trend import IchimokuIndicator

class FeatureEngineer:
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame):
        """
        添加技术指标：MA, RSI, MACD, 波动率指标等
        """
        df_feat = df.copy()
        
        # 确保 close 列存在且为数值型
        close = df_feat['close']
        high = df_feat['high']
        low = df_feat['low']
        volume = df_feat['volume']
        
        # 移动平均线 (MA)
        df_feat['ma_7'] = SMAIndicator(close, window=7).sma_indicator()
        df_feat['ma_25'] = SMAIndicator(close, window=25).sma_indicator()
        df_feat['ma_50'] = SMAIndicator(close, window=50).sma_indicator()
        df_feat['ema_12'] = EMAIndicator(close, window=12).ema_indicator()
        df_feat['ema_26'] = EMAIndicator(close, window=26).ema_indicator()
        
        # 相对强弱指数 (RSI)
        df_feat['rsi_7'] = RSIIndicator(close, window=7).rsi()
        df_feat['rsi_14'] = RSIIndicator(close, window=14).rsi()
        df_feat['rsi_21'] = RSIIndicator(close, window=21).rsi()
        
        # MACD
        macd = MACD(close)
        df_feat['macd'] = macd.macd()
        df_feat['macd_signal'] = macd.macd_signal()
        df_feat['macd_diff'] = macd.macd_diff()
        
        # 波动率指标
        # 布林带
        bb = BollingerBands(close, window=20)
        df_feat['bb_high'] = bb.bollinger_hband()
        df_feat['bb_low'] = bb.bollinger_lband()
        df_feat['bb_mid'] = bb.bollinger_mavg()
        df_feat['bb_width'] = bb.bollinger_wband()
        df_feat['bb_pct'] = bb.bollinger_pband()
        
        # 平均真实波动幅度 (ATR)
        df_feat['atr_14'] = AverageTrueRange(high, low, close, window=14).average_true_range()
        df_feat['atr_21'] = AverageTrueRange(high, low, close, window=21).average_true_range()
        
        # Keltner通道
        kc = KeltnerChannel(high, low, close, window=20)
        df_feat['kc_high'] = kc.keltner_channel_hband()
        df_feat['kc_low'] = kc.keltner_channel_lband()
        df_feat['kc_mid'] = kc.keltner_channel_mband()
        
        # 动量指标
        # 随机振荡器
        stoch = StochasticOscillator(high, low, close, window=14)
        df_feat['stoch_k'] = stoch.stoch()
        df_feat['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df_feat['williams_r'] = WilliamsRIndicator(high, low, close, lbp=14).williams_r()
        
        # 成交量指标
        # 平衡交易量 (OBV)
        df_feat['obv'] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        df_feat['obv_ema_12'] = EMAIndicator(df_feat['obv'], window=12).ema_indicator()
        
        # 量价趋势 (VPT)
        df_feat['vpt'] = VolumePriceTrendIndicator(close, volume).volume_price_trend()
        
        # 日本蜡烛图指标
        # 一目均衡表
        ichimoku = IchimokuIndicator(high, low, window1=9, window2=26, window3=52)
        df_feat['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
        df_feat['ichimoku_base'] = ichimoku.ichimoku_base_line()
        df_feat['ichimoku_span_a'] = ichimoku.ichimoku_a()
        df_feat['ichimoku_span_b'] = ichimoku.ichimoku_b()
        
        # 清除由于计算指标产生的NaN (前几行)
        df_feat = df_feat.dropna()
        
        return df_feat

    @staticmethod
    def add_sentiment_score(df: pd.DataFrame):
        """
        添加情绪因子 (基于市场数据的情绪指标)
        实际应用中应连接NLP分析模块
        """
        df_feat = df.copy()
        
        # 基于价格变动的情绪指标
        # 1. 涨跌幅度
        df_feat['price_change'] = df_feat['close'].pct_change()
        
        # 2. 相对强度指标的衍生情绪指标
        df_feat['rsi_sentiment'] = np.where(df_feat['rsi_14'] > 70, 1, np.where(df_feat['rsi_14'] < 30, -1, 0))
        
        # 3. 成交量加权的情绪指标
        df_feat['volume_sentiment'] = df_feat['price_change'] * df_feat['volume'] / df_feat['volume'].rolling(window=20).mean()
        
        # 4. 动量情绪指标
        df_feat['momentum_sentiment'] = df_feat['close'] - df_feat['close'].shift(20)
        
        # 5. 布林带位置情绪指标
        df_feat['bb_sentiment'] = (df_feat['close'] - df_feat['bb_low']) / (df_feat['bb_high'] - df_feat['bb_low'])
        
        # 6. 综合情绪分数 (简单加权平均)
        df_feat['sentiment'] = (
            0.2 * df_feat['rsi_sentiment'] +
            0.3 * df_feat['volume_sentiment'].clip(-1, 1) +
            0.2 * np.sign(df_feat['momentum_sentiment']) +
            0.3 * (2 * df_feat['bb_sentiment'] - 1)  # 将bb_sentiment从[0,1]映射到[-1,1]
        )
        
        # 清除由于计算产生的NaN
        df_feat = df_feat.dropna()
        
        return df_feat
    
    @staticmethod
    def add_volatility_features(df: pd.DataFrame):
        """
        添加波动率特征
        """
        df_feat = df.copy()
        
        # 1. 历史波动率 (HV)
        df_feat['hv_5'] = df_feat['close'].pct_change().rolling(window=5).std() * np.sqrt(252)  # 年化
        df_feat['hv_20'] = df_feat['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        df_feat['hv_60'] = df_feat['close'].pct_change().rolling(window=60).std() * np.sqrt(252)
        
        # 2. 波动率的波动率
        df_feat['vov_20'] = df_feat['hv_5'].rolling(window=20).std()
        
        # 3. 日内波动率
        df_feat['intraday_vol'] = (df_feat['high'] / df_feat['low'] - 1)
        
        # 4. 波动率聚类指标 (GARCH效应)
        df_feat['volatility_cluster'] = df_feat['price_change'].rolling(window=20).std() / df_feat['price_change'].rolling(window=120).std()
        
        # 5. 跳跃检测 (基于Z-score)
        df_feat['price_jump'] = np.abs(df_feat['price_change']) > 3 * df_feat['price_change'].rolling(window=20).std()
        df_feat['price_jump'] = df_feat['price_jump'].astype(int)
        
        return df_feat
    
    @staticmethod
    def add_microstructure_features(df: pd.DataFrame):
        """
        添加微观结构特征
        """
        df_feat = df.copy()
        
        # 1. 价格范围特征
        df_feat['range_high_low'] = df_feat['high'] - df_feat['low']
        df_feat['range_open_close'] = np.abs(df_feat['close'] - df_feat['open'])
        df_feat['range_ratio'] = df_feat['range_open_close'] / df_feat['range_high_low']
        
        # 2. 成交量特征
        df_feat['volume_ratio'] = df_feat['volume'] / df_feat['volume'].rolling(window=20).mean()
        df_feat['volume_change'] = df_feat['volume'].pct_change()
        
        # 3. 价格动量特征
        df_feat['momentum_5'] = df_feat['close'] / df_feat['close'].shift(5) - 1
        df_feat['momentum_20'] = df_feat['close'] / df_feat['close'].shift(20) - 1
        
        # 4. 反转特征
        df_feat['reversal_5'] = -df_feat['momentum_5']
        
        # 5. 趋势强度特征
        df_feat['trend_strength'] = (df_feat['close'] - df_feat['low'].rolling(window=20).min()) / (df_feat['high'].rolling(window=20).max() - df_feat['low'].rolling(window=20).min())
        
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
