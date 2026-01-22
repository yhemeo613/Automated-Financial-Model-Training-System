/*
 * @Author: dzy dzyperson@163.com
 * @Date: 2026-01-22 14:15:24
 * @LastEditors: dzy dzyperson@163.com
 * @LastEditTime: 2026-01-22 14:59:17
 * @FilePath: \模型训练\frontend\src\components\ChartPanel.jsx
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
import React, { useEffect } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area } from "recharts";

const ChartPanel = ({ data, chartType = "line" }) => {
    useEffect(() => {
        console.log("ChartPanel received data:", data);
    }, [data]);

    if (!data || data.length === 0) {
        return (
            <div className="chart-placeholder">
                <p>暂无预测数据，请运行预测任务</p>
            </div>
        );
    }

    // 自定义 Tooltip
    const CustomTooltip = ({ active, payload, label }) => {
        if (active && payload && payload.length) {
            return (
                <div
                    className="custom-tooltip"
                    style={{
                        backgroundColor: "rgba(30, 41, 59, 0.9)",
                        border: "1px solid #334155",
                        padding: "10px",
                        borderRadius: "4px",
                        boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
                        color: "#f1f5f9",
                    }}>
                    <p className="label" style={{ margin: "0 0 5px", fontSize: "0.8rem", color: "#94a3b8" }}>
                        {new Date(label).toLocaleString()}
                    </p>
                    {payload.map(
                        (entry, index) =>
                            // 过滤掉置信区间的辅助线
                            !["upper", "lower"].includes(entry.dataKey) && (
                                <p
                                    key={index}
                                    style={{ margin: 0, color: entry.color, fontSize: "0.9rem", fontWeight: 600 }}>
                                    {entry.name}: {Number(entry.value).toFixed(2)}
                                </p>
                            ),
                    )}
                </div>
            );
        }
        return null;
    };

    return (
        <div style={{ width: "100%", height: "100%", minHeight: "500px" }}>
            <ResponsiveContainer>
                {/* 使用 ComposedChart 以支持 Area 和 Line 混合 */}
                <LineChart data={data} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
                    <defs>
                        <linearGradient id="colorActual" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#8884d8" stopOpacity={0.8} />
                            <stop offset="95%" stopColor="#8884d8" stopOpacity={0} />
                        </linearGradient>
                        <linearGradient id="colorPredicted" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#82ca9d" stopOpacity={0.8} />
                            <stop offset="95%" stopColor="#82ca9d" stopOpacity={0} />
                        </linearGradient>
                        <linearGradient id="colorConfidence" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.2} />
                            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                        </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                    <XAxis
                        dataKey="timestamp"
                        stroke="#94a3b8"
                        tickFormatter={(t) => new Date(t).toLocaleDateString()}
                        tick={{ fontSize: 12 }}
                        tickMargin={10}
                    />
                    <YAxis stroke="#94a3b8" domain={["auto", "auto"]} tick={{ fontSize: 12 }} />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend wrapperStyle={{ paddingTop: "20px" }} />

                    {/* 置信区间 (使用 Area 模拟范围，需要数据处理技巧，Recharts 原生支持较弱，这里简化为上下界线或者用 Area 覆盖) */}
                    {/* 由于 Recharts Area 需要同一个 dataKey 的 range，这里如果数据结构是 flat 的，可以用 Stacked Area 或自定义 Shape。
                        简化方案：显示上下界虚线 */}

                    <Line
                        type="monotone"
                        dataKey="upper"
                        stroke="#3b82f6"
                        strokeWidth={1}
                        strokeDasharray="5 5"
                        dot={false}
                        name="90% 置信上限"
                        activeDot={false}
                        connectNulls={true}
                        opacity={0.5}
                    />

                    <Line
                        type="monotone"
                        dataKey="lower"
                        stroke="#3b82f6"
                        strokeWidth={1}
                        strokeDasharray="5 5"
                        dot={false}
                        name="90% 置信下限"
                        activeDot={false}
                        connectNulls={true}
                        opacity={0.5}
                    />

                    {/* 历史真实值 */}
                    <Line
                        type="monotone"
                        dataKey="actual"
                        stroke="#64748b"
                        strokeWidth={2}
                        dot={false}
                        name="真实值 (历史)"
                        activeDot={{ r: 6 }}
                        connectNulls={false}
                    />

                    {/* 预测值 (历史回测 + 未来预测) */}
                    <Line
                        type="monotone"
                        dataKey="predicted"
                        stroke="#3b82f6"
                        strokeWidth={3}
                        dot={false}
                        name="预测值"
                        activeDot={{ r: 6 }}
                        animationDuration={1500}
                        connectNulls={true}
                    />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
};

export default ChartPanel;
