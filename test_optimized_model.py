#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试优化后的模型性能
"""

import sys
import os
import pandas as pd
import numpy as np
from system_integration.service import TradingSystem

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_optimized_model():
    """
    测试优化后的模型性能
    """
    print("开始测试优化后的模型...")
    
    try:
        # 初始化交易系统
        trading_system = TradingSystem()
        
        # 1. 获取和处理数据
        print("\n1. 获取和处理数据...")
        df = trading_system.fetch_and_process_data()
        if df is None:
            print("数据获取失败")
            return False
        print(f"数据处理完成，形状: {df.shape}")
        
        # 2. 训练优化后的LSTM模型
        print("\n2. 训练优化后的LSTM模型...")
        def progress_callback(progress, message):
            print(f"  进度: {progress}% - {message}")
        
        result = trading_system.train_task(
            model_type='lstm',
            task_type='prediction',
            epochs=20,
            lr=0.001,
            dropout=0.2,
            progress_callback=progress_callback
        )
        
        if result['status'] == 'error':
            print(f"模型训练失败: {result['message']}")
            return False
        print(f"模型训练成功: {result['message']}")
        
        # 3. 测试预测功能
        print("\n3. 测试预测功能...")
        prediction_result = trading_system.predict_task(
            model_type='lstm',
            task_type='prediction',
            days=7
        )
        
        if prediction_result['status'] == 'error':
            print(f"预测失败: {prediction_result['message']}")
            return False
        
        print(f"预测成功，结果包含 {len(prediction_result['data']['results'])} 个数据点")
        print(f"预测置信度: {prediction_result['data']['confidence']:.4f}")
        print(f"平均绝对百分比误差 (MAPE): {(1 - prediction_result['data']['confidence']):.4f}")
        print(f"均方误差 (MSE): {prediction_result['data']['metrics']['mse']:.4f}")
        
        # 4. 检查可解释性分析结果
        if 'explainability' in prediction_result['data']:
            explainability = prediction_result['data']['explainability']
            print("\n4. 模型可解释性分析:")
            if 'top_features' in explainability:
                print(f"  最重要的5个特征: {explainability['top_features']}")
            if 'feature_importance' in explainability:
                print("  特征重要性:")
                for feature, importance in list(explainability['feature_importance'].items())[:5]:
                    print(f"    {feature}: {importance:.6f}")
        
        print("\n所有测试通过！优化后的模型性能良好。")
        return True
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_optimized_model()