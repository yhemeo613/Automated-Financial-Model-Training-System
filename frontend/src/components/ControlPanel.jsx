/*
 * @Author: dzy dzyperson@163.com
 * @Date: 2026-01-22 14:15:40
 * @LastEditors: dzy dzyperson@163.com
 * @LastEditTime: 2026-01-22 15:08:50
 * @FilePath: \模型训练\frontend\src\components\ControlPanel.jsx
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
import React, { useEffect } from "react";
import { Settings, Play, Database, Server } from "lucide-react";

const ControlPanel = ({ config, setConfig, onFetchData, onTrain, onPredict, loading, status }) => {
    // 当任务类型改变时，更新可用的模型选项
    useEffect(() => {
        if (config.task_type === "risk_control" && config.model_type === "lstm") {
            // 如果切到风控，且当前是LSTM，强制切到RF (因为LSTM风控逻辑还没显式支持，或者保持Ensemble/RF)
            // 根据后端 selector.py，风控支持 rf 和 ensemble
            setConfig((prev) => ({ ...prev, model_type: "rf" }));
        }
    }, [config.task_type]);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setConfig((prev) => ({
            ...prev,
            [name]: value,
        }));
    };

    return (
        <div className="control-panel">
            <div className="panel-section-title">
                <Settings size={16} />
                <span>参数配置</span>
            </div>

            <div className="form-grid">
                <div className="form-group">
                    <label>任务目标</label>
                    <select name="task_type" value={config.task_type} onChange={handleChange}>
                        <option value="prediction">趋势预测 (Prediction)</option>
                        <option value="risk_control">风险评估 (Risk Control)</option>
                    </select>
                </div>

                <div className="form-group">
                    <label>模型架构</label>
                    <select name="model_type" value={config.model_type} onChange={handleChange}>
                        {config.task_type === "prediction" && <option value="lstm">LSTM (深度学习)</option>}
                        <option value="rf">Random Forest (机器学习)</option>
                        <option value="ensemble">Ensemble (集成模型)</option>
                    </select>
                </div>

                <div className="form-row">
                    {(config.model_type === "lstm" || config.model_type === "ensemble") && (
                        <div className="form-group">
                            <label>训练轮数</label>
                            <input
                                type="number"
                                name="epochs"
                                value={config.epochs}
                                onChange={handleChange}
                                min="1"
                                max="1000"
                            />
                        </div>
                    )}

                    <div
                        className="form-group"
                        style={{
                            gridColumn:
                                config.model_type === "lstm" || config.model_type === "ensemble" ? "auto" : "span 2",
                        }}>
                        <label>预测天数</label>
                        <input
                            type="number"
                            name="days"
                            value={config.days}
                            onChange={handleChange}
                            min="1"
                            max="365"
                        />
                    </div>
                </div>

                {/* 高级参数区域 */}
                {(config.model_type === "lstm" || config.model_type === "ensemble") && (
                    <div className="form-row">
                        <div className="form-group">
                            <label>学习率 (LR)</label>
                            <input
                                type="number"
                                name="lr"
                                value={config.lr}
                                onChange={handleChange}
                                step="0.0001"
                                min="0.0001"
                                max="0.1"
                            />
                        </div>
                        <div className="form-group">
                            <label>Dropout</label>
                            <input
                                type="number"
                                name="dropout"
                                value={config.dropout}
                                onChange={handleChange}
                                step="0.1"
                                min="0"
                                max="0.5"
                            />
                        </div>
                    </div>
                )}

                {(config.model_type === "rf" || config.model_type === "ensemble") && (
                    <div className="form-group">
                        <label>决策树数量 (N Estimators)</label>
                        <input
                            type="number"
                            name="n_estimators"
                            value={config.n_estimators}
                            onChange={handleChange}
                            min="10"
                            max="1000"
                            step="10"
                        />
                    </div>
                )}
            </div>

            <div className="action-buttons">
                <button onClick={onFetchData} disabled={loading} className="btn-secondary">
                    <Database size={16} /> 获取数据
                </button>
                <button onClick={onTrain} disabled={loading} className="btn-primary">
                    <Server size={16} /> 训练模型
                </button>
                <button onClick={onPredict} disabled={loading} className="btn-success">
                    <Play size={16} /> {config.task_type === "risk_control" ? "评估风险" : "运行预测"}
                </button>
            </div>

            {status && (
                <div className={`status-message ${status.type}`}>
                    {status.type === "info" && <span className="loader"></span>}
                    {status.message}
                </div>
            )}
        </div>
    );
};

export default ControlPanel;
