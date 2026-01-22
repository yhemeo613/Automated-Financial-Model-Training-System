/*
 * @Author: dzy dzyperson@163.com
 * @Date: 2026-01-22 14:15:55
 * @LastEditors: dzy dzyperson@163.com
 * @LastEditTime: 2026-01-22 14:21:08
 * @FilePath: \模型训练\frontend\src\components\ResultsPanel.jsx
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
import React from "react";
import {
    History,
    Download,
    Clock,
    Activity,
    CheckCircle,
    XCircle,
    AlertCircle,
    ShieldAlert,
    TrendingDown,
} from "lucide-react";

const ResultsPanel = ({ results, metrics, history, onExport }) => {
    return (
        <div className="results-panel">
            {metrics && metrics.task_type === "risk_control" && (
                <>
                    <div className="panel-section-title">
                        <ShieldAlert size={16} />
                        <span>风险评估报告</span>
                    </div>
                    <div className="metrics-grid">
                        <div className="metric-card">
                            <div className="metric-label">风险等级</div>
                            <div
                                className="metric-value"
                                style={{
                                    color:
                                        metrics.risk_metrics.risk_level === "High"
                                            ? "var(--error)"
                                            : metrics.risk_metrics.risk_level === "Medium"
                                              ? "var(--warning)"
                                              : "var(--success)",
                                }}>
                                {metrics.risk_metrics.risk_level}
                            </div>
                        </div>
                        <div className="metric-card">
                            <div className="metric-label">最大回撤</div>
                            <div className="metric-value">{(metrics.risk_metrics.max_drawdown * 100).toFixed(2)}%</div>
                        </div>
                        <div className="metric-card">
                            <div className="metric-label">年化波动率</div>
                            <div className="metric-value">
                                {(metrics.risk_metrics.volatility_annual * 100).toFixed(2)}%
                            </div>
                        </div>
                        <div className="metric-card">
                            <div className="metric-label">VaR (95%)</div>
                            <div className="metric-value">{(metrics.risk_metrics.var_95 * 100).toFixed(2)}%</div>
                        </div>
                    </div>
                </>
            )}

            {metrics && metrics.task_type === "prediction" && (
                <>
                    <div className="panel-section-title">
                        <Activity size={16} />
                        <span>核心指标</span>
                    </div>
                    <div className="metrics-grid">
                        <div className="metric-card">
                            <div className="metric-label">模型置信度</div>
                            <div
                                className="metric-value"
                                style={{ color: metrics.confidence > 0.8 ? "var(--success)" : "var(--warning)" }}>
                                {(metrics.confidence * 100).toFixed(2)}%
                            </div>
                        </div>
                        {metrics.drift_detected !== undefined && (
                            <div
                                className="metric-card"
                                style={{
                                    borderColor: metrics.drift_detected ? "var(--error)" : "var(--success)",
                                    backgroundColor: metrics.drift_detected ? "rgba(239, 68, 68, 0.1)" : undefined,
                                }}>
                                <div className="metric-label">数据状态</div>
                                <div
                                    className="metric-value"
                                    style={{
                                        fontSize: "0.8rem",
                                        color: metrics.drift_detected ? "var(--error)" : "var(--success)",
                                    }}>
                                    {metrics.drift_msg || (metrics.drift_detected ? "检测到漂移" : "分布稳定")}
                                </div>
                            </div>
                        )}
                        <div className="metric-card">
                            <div className="metric-label">执行耗时</div>
                            <div className="metric-value">{metrics.duration.toFixed(2)}s</div>
                        </div>
                        <div className="metric-card export-card" onClick={() => onExport("csv")} role="button">
                            <div className="metric-label">导出结果</div>
                            <div className="metric-value">
                                <Download size={20} />
                            </div>
                        </div>
                    </div>
                </>
            )}

            <div className="history-section">
                <div className="panel-section-title">
                    <History size={16} />
                    <span>执行日志</span>
                </div>
                <div className="history-list">
                    {history.length === 0 ? (
                        <div className="empty-state">
                            <Clock size={32} />
                            <p>暂无操作记录</p>
                        </div>
                    ) : (
                        history.map((item, index) => (
                            <div key={index} className={`history-item ${item.type}`}>
                                <div className="history-info">
                                    <span className="history-action">{item.action}</span>
                                    <span className="history-time">{item.time}</span>
                                </div>
                                <div className="history-status">
                                    {item.type === "success" && <CheckCircle size={14} color="var(--success)" />}
                                    {item.type === "error" && <XCircle size={14} color="var(--error)" />}
                                    {item.type === "info" && <AlertCircle size={14} color="var(--primary)" />}
                                    <span>{item.status}</span>
                                </div>
                            </div>
                        ))
                    )}
                </div>
            </div>
        </div>
    );
};

export default ResultsPanel;
