import ccxt
import pandas as pd
import os
import sys

# 添加项目根目录到 sys.path，以便导入 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system_integration.config import config

class BinanceDataFetcher:
    def __init__(self):
        """
        初始化币安数据获取器
        """
        exchange_config = {
            'apiKey': config.BINANCE_API_KEY,
            'secret': config.BINANCE_SECRET_KEY,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'  # U本位合约
            }
        }

        # 如果配置了代理，则添加到配置中
        if config.PROXY_URL:
            exchange_config['proxies'] = {
                'http': config.PROXY_URL,
                'https': config.PROXY_URL,
            }
            print(f"已启用代理: {config.PROXY_URL}")

        self.exchange = ccxt.binance(exchange_config)
        
    def fetch_data(self, symbol=config.SYMBOL, timeframe=config.TIMEFRAME, limit=config.LIMIT, since=None):
        """
        获取K线数据
        :param symbol: 交易对，如 'BTC/USDT'
        :param timeframe: 时间周期，如 '1h', '1d'
        :param limit: 获取条数
        :param since: 开始时间戳 (ms)
        :return: DataFrame
        """
        try:
            print(f"正在从币安获取 {symbol} ({timeframe}) 数据...")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            
            if not ohlcv:
                print("未获取到数据")
                return pd.DataFrame()

            # 转换为 DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            print(f"成功获取 {len(df)} 条数据")
            return df
        except Exception as e:
            print(f"获取数据失败: {e}")
            return pd.DataFrame()

    def save_to_csv(self, df, filename="btc_usdt_data.csv"):
        """
        保存数据到CSV文件
        :param df: 数据DataFrame
        :param filename: 文件名
        """
        if df.empty:
            print("数据为空，跳过保存")
            return

        filepath = os.path.join(config.DATA_DIR, filename)
        
        # 简单实现：直接覆盖。实际生产中可能需要增量更新
        df.to_csv(filepath, index=False)
        print(f"数据已保存至 {filepath}")

if __name__ == "__main__":
    # 测试代码
    fetcher = BinanceDataFetcher()
    df = fetcher.fetch_data()
    if not df.empty:
        fetcher.save_to_csv(df)
