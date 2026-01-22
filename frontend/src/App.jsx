import { useState } from "react";
import axios from "axios";
import { Activity, BarChart2 } from "lucide-react";
import ChartPanel from "./components/ChartPanel";
import ControlPanel from "./components/ControlPanel";
import ResultsPanel from "./components/ResultsPanel";
import "./App.css";

function App() {
    const [config, setConfig] = useState({
        model_type: "lstm",
        task_type: "prediction",
        epochs: 10,
        days: 30,
        lr: 0.001,
        dropout: 0.2,
        n_estimators: 100,
    });

    const [chartData, setChartData] = useState([]);
    const [metrics, setMetrics] = useState(null);
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(false);
    const [status, setStatus] = useState(null);

    const API_URL = "http://localhost:8000/api";

    const addHistory = (action, statusMsg) => {
        const newItem = {
            time: new Date().toLocaleTimeString(),
            model_type: config.model_type,
            action: action,
            status: statusMsg,
            type: statusMsg === "Success" ? "success" : statusMsg === "Failed" ? "error" : "info",
        };
        setHistory((prev) => [newItem, ...prev].slice(0, 20));
    };

    const handleFetchData = async () => {
        setLoading(true);
        setStatus({ type: "info", message: "正在获取数据..." });
        try {
            const res = await axios.post(`${API_URL}/data/fetch`);
            setStatus({ type: "success", message: res.data.message });
            addHistory("Fetch Data", "Success");
        } catch (error) {
            setStatus({ type: "error", message: error.message });
            addHistory("Fetch Data", "Failed");
        } finally {
            setLoading(false);
        }
    };

    const handleTrain = async () => {
        setLoading(true);
        setStatus({ type: "info", message: "正在提交训练任务..." });
        try {
            const res = await axios.post(`${API_URL}/train`, {
                model_type: config.model_type,
                task_type: config.task_type,
                epochs: Number(config.epochs),
                lr: Number(config.lr),
                dropout: Number(config.dropout),
                n_estimators: Number(config.n_estimators),
            });
            setStatus({ type: "success", message: res.data.message });
            addHistory("Train Model", "Started");

            // 开始轮询进度
            pollProgress();
        } catch (error) {
            setStatus({ type: "error", message: error.message });
            addHistory("Train Model", "Failed");
            setLoading(false);
        }
    };

    const pollProgress = () => {
        const interval = setInterval(async () => {
            try {
                const res = await axios.get(`${API_URL}/train/status`);
                const { status, progress, message } = res.data;

                setStatus({ type: "info", message: `训练中: ${message} (${progress}%)` });

                if (status === "completed") {
                    clearInterval(interval);
                    setLoading(false);
                    setStatus({ type: "success", message: "训练任务完成" });
                    addHistory("Train Model", "Success");
                } else if (status === "error") {
                    clearInterval(interval);
                    setLoading(false);
                    setStatus({ type: "error", message: `训练失败: ${message}` });
                    addHistory("Train Model", "Failed");
                }
            } catch (error) {
                console.error("Poll progress failed:", error);
                clearInterval(interval);
                setLoading(false);
            }
        }, 1000);
    };

    const handlePredict = async () => {
        setLoading(true);
        setStatus({ type: "info", message: "正在执行任务..." });
        try {
            const res = await axios.post(`${API_URL}/predict`, {
                model_type: config.model_type,
                task_type: config.task_type,
                days: Number(config.days),
            });

            const { data } = res.data;

            if (data.task_type === "risk_control") {
                // 风险控制任务处理
                setMetrics({
                    task_type: "risk_control",
                    risk_metrics: data.risk_metrics,
                    duration: data.duration,
                });
                // 清空图表数据或显示特定信息
                setChartData([]);
            } else {
                // 预测任务处理
                setChartData(data.results);
                setMetrics({
                    task_type: "prediction",
                    confidence: data.confidence,
                    duration: data.duration,
                    mse: data.metrics.mse,
                });
            }

            setStatus({ type: "success", message: "任务完成" });
            addHistory("Predict", "Success");
        } catch (error) {
            const msg = error.response?.data?.detail || error.message;
            setStatus({ type: "error", message: msg });
            addHistory("Predict", "Failed");
        } finally {
            setLoading(false);
        }
    };

    const handleExport = (format) => {
        if (!chartData.length) return;

        if (format === "csv") {
            const headers = ["Timestamp", "Actual", "Predicted"];
            const csvContent = [
                headers.join(","),
                ...chartData.map((row) => `${row.timestamp},${row.actual},${row.predicted}`),
            ].join("\n");

            const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
            const link = document.createElement("a");
            link.href = URL.createObjectURL(blob);
            link.download = `prediction_results_${new Date().getTime()}.csv`;
            link.click();
        }
    };

    return (
        <div className="app-container">
            {/* 侧边栏 */}
            <aside className="app-sidebar">
                <div className="brand">
                    <div className="brand-icon">
                        <Activity size={20} />
                    </div>
                    <div>
                        <h1>FinModel AI</h1>
                        <span className="badge">v1.0 Pro</span>
                    </div>
                </div>

                <ControlPanel
                    config={config}
                    setConfig={setConfig}
                    onFetchData={handleFetchData}
                    onTrain={handleTrain}
                    onPredict={handlePredict}
                    loading={loading}
                    status={status}
                />

                <ResultsPanel results={chartData} metrics={metrics} history={history} onExport={handleExport} />
            </aside>

            {/* 主内容区 */}
            <main className="app-main">
                <header className="main-header">
                    <h2>
                        <BarChart2 size={24} />
                        市场预测分析
                    </h2>
                </header>

                <div className="chart-wrapper">
                    <ChartPanel data={chartData} chartType="line" />
                </div>
            </main>
        </div>
    );
}

export default App;
