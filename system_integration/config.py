'''
Author: dzy dzyperson@163.com
Date: 2026-01-22 14:00:36
LastEditors: dzy dzyperson@163.com
LastEditTime: 2026-01-22 14:10:38
FilePath: /模型训练/system_integration/config.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

class Config:
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
    
    # 代理设置 (可选)
    # 格式示例: "http://127.0.0.1:7890"
    PROXY_URL = os.getenv("PROXY_URL")
    
    # 交易对设置
    SYMBOL = "BTC/USDT"
    TIMEFRAME = "1h"  # 默认时间周期
    LIMIT = 1000      # 默认获取条数

    # 数据存储路径
    DATA_DIR = os.path.join(os.getcwd(), "data")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

config = Config()
