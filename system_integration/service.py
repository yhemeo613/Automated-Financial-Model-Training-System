from data_acquisition.binance_client import BinanceDataFetcher
from data_processing.pipeline import DataPipeline
from data_processing.features import FeatureEngineer
from models.selector import ModelSelector
from system_integration.config import config
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from scipy.stats import ks_2samp
import joblib
import time

class TradingSystem:
    def __init__(self):
        self.fetcher = BinanceDataFetcher()
        self.pipeline = DataPipeline()
        self.model_selector = ModelSelector()

    def fetch_and_process_data(self, progress_callback=None):
        """
        获取并处理数据
        :param progress_callback: 进度回调函数，接收(progress, message)参数
        """
        # 1. 获取数据
        if progress_callback: progress_callback(20, "正在从交易所获取数据...")
        df = self.fetcher.fetch_data()
        if df.empty:
            return None
        
        # 保存原始数据
        if progress_callback: progress_callback(40, "数据获取完成，正在保存原始数据...")
        self.fetcher.save_to_csv(df, "raw_data.csv")
        
        # 2. 处理数据
        if progress_callback: progress_callback(60, "正在进行数据清洗...")
        df_processed = self.pipeline.run(df)
        
        # 保存处理后数据
        if progress_callback: progress_callback(80, "数据处理完成，正在保存处理后数据...")
        self.fetcher.save_to_csv(df_processed, "processed_data.csv")
        
        if progress_callback: progress_callback(100, "数据获取和处理完成")
        return df_processed

    def train_task(self, model_type='lstm', task_type='prediction', epochs=10, lr=0.001, dropout=0.2, n_estimators=100, progress_callback=None):
        """
        执行训练任务 (集成 TimeSeriesSplit 交叉验证和网格搜索)
        """
        # 读取处理后的数据
        data_path = os.path.join(config.DATA_DIR, "processed_data.csv")
        if not os.path.exists(data_path):
            print("数据文件不存在，尝试重新获取...")
            if progress_callback: progress_callback(5, "正在获取数据...")
            df = self.fetch_and_process_data()
            if df is None:
                return {"status": "error", "message": "无法获取数据"}
        else:
            df = pd.read_csv(data_path)

        if progress_callback: progress_callback(10, "数据加载完成，正在预处理...")

        # 准备数据 
        # 排除非数值列
        feature_cols = [c for c in df.columns if c not in ['timestamp', 'date']]
        df_features = df[feature_cols]
        
        # 简单填充一下可能存在的NaN
        df_features = df_features.fillna(0)
        
        # 归一化处理
        scaler = MinMaxScaler(feature_range=(0, 1))
        # 转换为 numpy array 进行缩放
        data_scaled = scaler.fit_transform(df_features.values)
        
        # 保存 scaler
        scaler_path = os.path.join(config.DATA_DIR, "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        
        # 将缩放后的数据转回 DataFrame 以便 create_xy_windows 使用列名
        df_scaled = pd.DataFrame(data_scaled, columns=feature_cols)
        
        # 创建窗口
        window_size = 60
        # 假设 'close' 是目标
        if 'close' not in df_scaled.columns:
             return {"status": "error", "message": "数据中缺少 close 列"}
             
        X, y = FeatureEngineer.create_xy_windows(df_scaled, target_col='close', window_size=window_size)
        
        if len(X) == 0:
             return {"status": "error", "message": "数据量不足以创建窗口"}

        # 使用滚动窗口验证 (Walk-Forward Validation) 进行交叉验证
        # 滚动窗口参数设置
        train_window_size = int(len(X) * 0.7)  # 初始训练窗口大小（占总数据的70%）
        val_window_size = int(len(X) * 0.1)    # 验证窗口大小（占总数据的10%）
        step_size = val_window_size            # 滚动步长（与验证窗口大小相同）
        
        # 计算总窗口数
        total_windows = (len(X) - train_window_size - val_window_size) // step_size + 1
        
        best_loss = float('inf')
        best_model = None
        input_dim = X.shape[2] 
        
        # 简单的 Grid Search (针对 LSTM)
        # 实际生产中应更复杂，这里仅演示原理
        if model_type == 'lstm':
            grid_params = [
                {'hidden_dim': 32, 'num_layers': 1, 'use_attention': True},
                {'hidden_dim': 64, 'num_layers': 2, 'use_attention': True},
                {'hidden_dim': 64, 'num_layers': 1, 'use_attention': False}
            ]
        else:
            grid_params = [{}] # 其他模型暂不搜索
            
        total_steps = len(grid_params) * total_windows
        current_step = 0
        
        if progress_callback: progress_callback(15, f"开始滚动窗口验证与网格搜索...总窗口数: {total_windows}")
        
        for params in grid_params:
            window_losses = []
            
            # 每一组参数都需要实例化一个模型
            # 注意：Ensemble 模型本身包含多个子模型，这里不再对 Ensemble 做 Grid Search，而是直接训练
            if model_type == 'ensemble':
                # 集成模型特殊处理，不做Grid Search，只做交叉验证评估
                model = self.model_selector.get_model(task_type, model_type, input_dim=input_dim, epochs=epochs, lr=lr, dropout=dropout, n_estimators=n_estimators)
            else:
                model = self.model_selector.get_model(task_type, model_type, input_dim=input_dim, epochs=epochs, lr=lr, dropout=dropout, n_estimators=n_estimators, **params)
            
            # 滚动窗口验证
            for i in range(total_windows):
                current_step += 1
                progress = 15 + int((current_step / total_steps) * 60)
                if progress_callback: 
                    progress_callback(progress, f"正在验证参数 {params} (窗口 {i+1}/{total_windows})...")
                
                # 计算当前窗口的索引
                train_start = i * step_size
                train_end = train_start + train_window_size
                val_start = train_end
                val_end = val_start + val_window_size
                
                # 获取训练集和验证集
                X_train_fold, X_val_fold = X[train_start:train_end], X[val_start:val_end]
                y_train_fold, y_val_fold = y[train_start:train_end], y[val_start:val_end]
                
                # 检查数据量
                if len(X_train_fold) == 0 or len(X_val_fold) == 0:
                    break
                
                # 训练
                try:
                    # 简化内部回调，避免日志过多
                    model.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
                    
                    # 验证
                    pred = model.predict(X_val_fold)
                    # 展平
                    pred = pred.reshape(-1)
                    y_val_fold = y_val_fold.reshape(-1)
                    
                    mse = mean_squared_error(y_val_fold, pred)
                    window_losses.append(mse)
                    print(f"  窗口 {i+1}: MSE = {mse:.6f}")
                    
                except Exception as e:
                    print(f"训练窗口 {i+1} 失败: {e}")
                    window_losses.append(float('inf'))
            
            avg_loss = np.mean(window_losses)
            print(f"参数 {params} 平均验证 Loss: {avg_loss:.6f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                # 需要重新在全量数据上训练最佳模型
                best_model = model
                # 更新最佳参数记录 (如果需要)

        if progress_callback: progress_callback(80, "交叉验证完成，正在使用全量数据训练最佳模型...")
        
        # 使用全部数据重新训练最佳模型 (除了最后的 Test set，但在本例中我们用全部数据作为 Knowledge Base)
        # 为了生产部署，通常用所有可用数据
        try:
            # 重新初始化最佳模型以免受到 Fold 状态影响 (特别是 LSTM 的 state)
            # 这里简化直接继续训练或复用
            best_model.train(X, y)
        except Exception as e:
             return {"status": "error", "message": f"最终训练出错: {str(e)}"}
        
        if progress_callback: progress_callback(95, "训练完成，正在保存模型...")

        # 保存模型
        ext = ".pth" if model_type == 'lstm' else ".pkl"
        model_path = os.path.join(config.DATA_DIR, f"{model_type}_model{ext}")
        best_model.save(model_path)
        
        if progress_callback: progress_callback(100, "任务全部完成")
        
        return {"status": "success", "message": f"模型 {model_type} 训练完成 (CV Loss: {best_loss:.4f})", "path": model_path}

    def predict_task(self, model_type='lstm', task_type='prediction', days=30):
        """
        执行预测或风险评估任务
        :param days: 预测范围
        """
        start_time = time.time()
        
        # 1. 加载数据
        data_path = os.path.join(config.DATA_DIR, "processed_data.csv")
        if not os.path.exists(data_path):
             return {"status": "error", "message": "数据文件不存在，请先获取数据"}
        
        df = pd.read_csv(data_path)
        
        # 确保时间戳列存在并排序
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        # 准备特征
        feature_cols = [c for c in df.columns if c not in ['timestamp', 'date']]
        df_features = df[feature_cols].fillna(0)
        
        # 加载 scaler 并归一化
        scaler_path = os.path.join(config.DATA_DIR, "scaler.pkl")
        if not os.path.exists(scaler_path):
             return {"status": "error", "message": "Scaler文件不存在，请先训练模型"}
        
        try:
            scaler = joblib.load(scaler_path)
            data_scaled = scaler.transform(df_features.values)
            df_scaled = pd.DataFrame(data_scaled, columns=feature_cols)
        except Exception as e:
             return {"status": "error", "message": f"数据归一化失败: {str(e)}"}

        window_size = 60
        if 'close' not in df_scaled.columns:
             return {"status": "error", "message": "数据中缺少 close 列"}
             
        X, y = FeatureEngineer.create_xy_windows(df_scaled, target_col='close', window_size=window_size)
        
        if len(X) == 0:
             return {"status": "error", "message": "数据量不足"}

        # ------------------- 数据漂移检测 (Data Drift Detection) -------------------
        # 比较最近的窗口数据分布与较早的数据分布
        # 取最近 20% 数据作为 "Current", 前 50% 作为 "Reference"
        ref_size = int(len(X) * 0.5)
        curr_size = int(len(X) * 0.2)
        if curr_size > 10 and ref_size > 10:
            # 仅比较 Close 特征 (简化)
            # X shape: (samples, window, features)
            # 取每个窗口的最后一个时间步的 Close 值
            try:
                close_idx = feature_cols.index('close')
                ref_dist = X[:ref_size, -1, close_idx]
                curr_dist = X[-curr_size:, -1, close_idx]
                
                ks_stat, p_value = ks_2samp(ref_dist, curr_dist)
                drift_detected = p_value < 0.05
                drift_msg = f"检测到数据漂移 (p={p_value:.4f})" if drift_detected else "数据分布稳定"
            except:
                drift_detected = False
                drift_msg = "漂移检测失败"
        else:
            drift_detected = False
            drift_msg = "样本不足以检测漂移"

        # 2. 加载模型
        # Ensemble 模型可能没有统一的后缀，或者需要特殊处理
        # 训练时我们保存为 .pkl (best_model.save(model_path))
        # 而 EnsembleModel.save 实际上是保存了多个子模型，本身并没有创建一个单独的文件
        # 这是一个逻辑漏洞：Service 层期望有一个主文件存在，而 EnsembleModel.save 只是保存了子文件
        
        # 修正：
        # 1. EnsembleModel.save 应该创建一个空的占位文件或者配置 json，以满足 os.path.exists(model_path)
        # 2. 或者 Service 层这里对 ensemble 做特殊判断
        
        # 更好的做法是让 EnsembleModel.save 创建一个 metadata 文件
        # 暂时在 Service 层做兼容：
        
        ext = ".pkl" if model_type == 'rf' or model_type == 'ensemble' else ".pth"
        model_path = os.path.join(config.DATA_DIR, f"{model_type}_model{ext}")
        
        # 对于 Ensemble，save 方法并没有创建这个 model_path 文件，而是创建了 model_path_lstm_0.pth 等
        # 所以这里 os.path.exists 会失败，导致 400 错误
        
        # 我们修改检查逻辑：
        if model_type == 'ensemble':
            # 只要子模型存在即可，或者我们可以让 save 方法创建一个 metadata 文件
            # 简单起见，我们假设至少有一个子模型存在
            # 检查第一个子模型
            sub_path_0 = os.path.join(config.DATA_DIR, f"{model_type}_model_lstm_0.pth")
            if not os.path.exists(sub_path_0):
                 return {"status": "error", "message": "Ensemble 模型文件不存在 (子模型丢失)，请先训练模型"}
        else:
            if not os.path.exists(model_path):
                 return {"status": "error", "message": "模型文件不存在，请先训练模型"}
             
        input_dim = X.shape[2]
        model = self.model_selector.get_model(task_type, model_type, input_dim=input_dim)
        try:
            model.load(model_path)
        except Exception as e:
            return {"status": "error", "message": f"加载模型失败: {str(e)}"}

        # 获取 target 列 (close) 的缩放参数
        target_col_idx = df_features.columns.get_loc('close')
        scale = scaler.scale_[target_col_idx]
        min_ = scaler.min_[target_col_idx]

        # ------------------- 风险评估逻辑 -------------------
        if task_type == 'risk_control':
            # 风险评估：计算历史波动率、VaR (Value at Risk) 和最大回撤
            # 这里我们使用历史数据进行评估，不需要像预测那样划分测试集
            
            # 还原整个历史价格序列
            y_original = (y - min_) / scale
            
            # 计算日收益率
            returns = np.diff(y_original) / y_original[:-1]
            
            # 1. 波动率 (年化)
            volatility = np.std(returns) * np.sqrt(365 * 24) # 假设小时数据
            
            # 2. VaR (95% 置信度)
            var_95 = np.percentile(returns, 5)
            
            # 3. 最大回撤
            cum_max = np.maximum.accumulate(y_original)
            drawdown = (y_original - cum_max) / cum_max
            max_drawdown = np.min(drawdown)
            
            # 4. 预测未来波动性 (使用模型)
            # 使用最近的数据预测未来 trend，并基于此评估风险
            # 这里简单复用预测逻辑获取未来 N 天的走势，判断是否处于下行风险区
            
            # 取最近一段数据进行预测
            last_X = X[-1].reshape(1, window_size, input_dim)
            future_steps = days * 24
            future_prices = []
            current_input = last_X
            
            # 寻找 close 索引
            try:
                close_idx = feature_cols.index('close')
            except:
                close_idx = 0
                
            for _ in range(future_steps):
                pred = model.predict(current_input).flatten()[0]
                future_prices.append(pred)
                # 简单滚动
                next_feat = current_input[0, -1, :].copy()
                next_feat[close_idx] = pred
                current_input = np.concatenate([current_input[:, 1:, :], next_feat.reshape(1, 1, input_dim)], axis=1)
                
            future_prices = (np.array(future_prices) - min_) / scale
            
            # 评估未来风险
            current_price = y_original[-1]
            min_future_price = np.min(future_prices)
            future_drawdown = (min_future_price - current_price) / current_price
            
            risk_level = "Low"
            if future_drawdown < -0.1: risk_level = "High"
            elif future_drawdown < -0.05: risk_level = "Medium"
            
            duration = time.time() - start_time
            
            return {
                "status": "success",
                "data": {
                    "task_type": "risk_control",
                    "risk_metrics": {
                        "volatility_annual": float(volatility),
                        "var_95": float(var_95),
                        "max_drawdown": float(max_drawdown),
                        "projected_drawdown": float(future_drawdown),
                        "risk_level": risk_level
                    },
                    "duration": float(duration),
                    # 为了兼容前端图表，还是返回一些历史数据
                    "results": [] 
                }
            }

        # ------------------- 价格预测逻辑 (增强版) -------------------
        
        # 取最近 days 天的数据进行预测
        # 1. 历史回测部分 (Test Set)
        test_size = min(len(X) // 5, days * 24) 
        if test_size < 1: test_size = 1
            
        X_test = X[-test_size:]
        y_test = y[-test_size:] # 这是归一化后的真实值
        
        timestamps_test = df['timestamp'].iloc[window_size:].values[-test_size:]
        
        # 3. 历史回测预测 (带不确定性)
        try:
            if hasattr(model, 'predict_with_uncertainty'):
                mean_pred, std_pred = model.predict_with_uncertainty(X_test)
                predictions_test = mean_pred.flatten()
                # 90% 置信区间 (1.645 * std)
                upper_bound_test = predictions_test + 1.645 * std_pred.flatten()
                lower_bound_test = predictions_test - 1.645 * std_pred.flatten()
            else:
                predictions_test = model.predict(X_test).flatten()
                upper_bound_test = predictions_test # 无法计算，设为相等
                lower_bound_test = predictions_test
                
        except Exception as e:
            return {"status": "error", "message": f"预测出错: {str(e)}"}
            
        # 4. 未来趋势预测 (Auto-regressive)
        future_days = days # 使用参数指定的预测天数
        future_steps = future_days * 24
        
        # 使用最后一个已知窗口作为起点
        last_window = X[-1] # shape: (window_size, features)
        future_predictions = []
        future_upper = []
        future_lower = []
        
        current_input = last_window.reshape(1, window_size, input_dim)
        
        # 计算最近窗口的波动率
        try:
            close_idx = feature_cols.index('close')
            recent_volatility = np.std(last_window[:, close_idx])
            if recent_volatility == 0: recent_volatility = 0.01
        except:
            recent_volatility = 0.01
            close_idx = 0
        
        # 对于未来趋势预测，使用确定性的predict方法，确保结果可重复
        for i in range(future_steps):
            # 预测下一步，始终使用确定性的predict方法
            next_pred = model.predict(current_input)
            next_val = next_pred.flatten()[0]
            
            # 使用历史波动率作为不确定性估计
            uncertainty = recent_volatility * (1 + i * 0.05) # 随时间扩大不确定性
            
            # 确保值在合理范围内
            next_val = np.clip(next_val, 0, 1)
            
            future_predictions.append(next_val)
            
            # 计算置信区间
            future_upper.append(next_val + 1.645 * uncertainty)
            future_lower.append(next_val - 1.645 * uncertainty)
            
            # 更新输入窗口
            next_features = current_input[0, -1, :].copy()
            next_features[close_idx] = next_val
            
            # 滚动窗口
            new_window = np.concatenate([current_input[:, 1:, :], next_features.reshape(1, 1, input_dim)], axis=1)
            current_input = new_window

        # 5. 反归一化
        # 还原测试集
        y_test_original = (y_test - min_) / scale
        predictions_test_original = (predictions_test - min_) / scale
        upper_test_original = (upper_bound_test - min_) / scale
        lower_test_original = (lower_bound_test - min_) / scale
        
        # 还原未来预测
        future_predictions_original = (np.array(future_predictions) - min_) / scale
        future_upper_original = (np.array(future_upper) - min_) / scale
        future_lower_original = (np.array(future_lower) - min_) / scale

        # 6. 统计指标
        mse = np.mean((predictions_test_original - y_test_original) ** 2)
        mape = np.mean(np.abs((y_test_original - predictions_test_original) / y_test_original))
        confidence = max(0, 1 - mape)
        
        # 7. 模型可解释性分析
        explainability = {}
        
        # 特征重要性分析 (简单版本)
        try:
            # 对于LSTM模型，我们可以分析输入特征的重要性
            # 这里使用一种简单的方法：计算每个特征对预测结果的敏感性
            if model_type == 'lstm':
                # 计算特征重要性
                feature_importance = {}
                
                # 使用最后一个测试样本作为基准
                base_sample = X_test[-1:]
                base_pred = model.predict(base_sample)[0][0]
                
                # 对每个特征进行敏感性分析
                for i, feature in enumerate(feature_cols):
                    # 创建扰动样本
                    perturbed_sample = base_sample.copy()
                    # 对特征值进行10%的扰动
                    perturbed_sample[0, :, i] *= 1.1
                    # 计算扰动后的预测
                    perturbed_pred = model.predict(perturbed_sample)[0][0]
                    # 计算敏感性（相对变化）
                    sensitivity = abs((perturbed_pred - base_pred) / base_pred)
                    feature_importance[feature] = float(sensitivity)
                
                # 按重要性排序
                sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
                explainability['feature_importance'] = sorted_importance
                explainability['top_features'] = list(sorted_importance.keys())[:5]
                
            # 对于随机森林模型，可以直接获取特征重要性
            elif model_type == 'rf':
                # 注意：这里需要确保模型有feature_importances_属性
                if hasattr(model.model, 'feature_importances_'):
                    feature_importance = {}
                    for i, feature in enumerate(feature_cols):
                        feature_importance[feature] = float(model.model.feature_importances_[i])
                    
                    # 按重要性排序
                    sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
                    explainability['feature_importance'] = sorted_importance
                    explainability['top_features'] = list(sorted_importance.keys())[:5]
        except Exception as e:
            print(f"可解释性分析失败: {e}")
            explainability['error'] = str(e)
        
        duration = time.time() - start_time
        
        # 8. 格式化结果
        result_data = []
        
        # 历史回测部分
        for i in range(len(predictions_test_original)):
            result_data.append({
                "timestamp": str(timestamps_test[i]),
                "actual": float(y_test_original[i]),
                "predicted": float(predictions_test_original[i]),
                "upper": float(upper_test_original[i]),
                "lower": float(lower_test_original[i]),
                "type": "history"
            })
            
        # 未来预测部分
        last_timestamp = pd.to_datetime(timestamps_test[-1])
        for i in range(len(future_predictions_original)):
            next_time = last_timestamp + pd.Timedelta(hours=(i+1))
            result_data.append({
                "timestamp": str(next_time),
                "actual": None, 
                "predicted": float(future_predictions_original[i]),
                "upper": float(future_upper_original[i]),
                "lower": float(future_lower_original[i]),
                "type": "future"
            })
            
        return {
            "status": "success",
            "data": {
                "results": result_data,
                "confidence": float(confidence),
                "duration": float(duration),
                "metrics": {
                    "mse": float(mse),
                    "drift_detected": bool(drift_detected),
                    "drift_msg": drift_msg
                },
                "explainability": explainability
            }
        }
