'''
Author: dzy dzyperson@163.com
Date: 2026-01-22 14:03:09
LastEditors: dzy dzyperson@163.com
LastEditTime: 2026-01-22 15:00:48
FilePath: \模型训练\system_integration\api\main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from system_integration.service import TradingSystem
import uvicorn
import os

app = FastAPI(title="自动化金融模型训练系统")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

system = TradingSystem()

class TrainRequest(BaseModel):
    model_type: str = "lstm"
    task_type: str = "prediction"
    epochs: int = 10
    lr: float = 0.001
    dropout: float = 0.2
    n_estimators: int = 100

class PredictRequest(BaseModel):
    model_type: str = "lstm"
    task_type: str = "prediction"
    days: int = 30

@app.post("/api/data/fetch")
async def fetch_data(background_tasks: BackgroundTasks):
    """
    触发数据抓取任务 (后台运行)
    """
    background_tasks.add_task(system.fetch_and_process_data)
    return {"message": "数据抓取任务已启动"}

# 全局变量存储任务状态
# 实际生产中应使用 Redis 或数据库
task_status = {
    "train": {
        "status": "idle", # idle, running, completed, error
        "progress": 0,
        "message": ""
    }
}

@app.post("/api/train")
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    触发模型训练任务 (后台运行)
    """
    if task_status["train"]["status"] == "running":
        raise HTTPException(status_code=400, detail="训练任务正在进行中")

    # 简单的包装函数以便放入后台任务
    def run_train():
        task_status["train"]["status"] = "running"
        task_status["train"]["progress"] = 0
        task_status["train"]["message"] = "开始训练..."
        
        try:
            # 传递回调函数更新进度
            def progress_callback(progress, message):
                task_status["train"]["progress"] = progress
                task_status["train"]["message"] = message
                
            result = system.train_task(
                request.model_type, 
                request.task_type, 
                request.epochs,
                lr=request.lr,
                dropout=request.dropout,
                n_estimators=request.n_estimators,
                progress_callback=progress_callback
            )
            print(f"训练结果: {result}")
            
            if result["status"] == "success":
                task_status["train"]["status"] = "completed"
                task_status["train"]["progress"] = 100
                task_status["train"]["message"] = "训练完成"
            else:
                task_status["train"]["status"] = "error"
                task_status["train"]["message"] = result["message"]
                
        except Exception as e:
            task_status["train"]["status"] = "error"
            task_status["train"]["message"] = str(e)

    background_tasks.add_task(run_train)
    return {"message": f"模型 {request.model_type} 训练任务已启动"}

@app.get("/api/train/status")
async def get_train_status():
    """
    获取训练任务状态
    """
    return task_status["train"]

@app.post("/api/predict")
async def predict(request: PredictRequest):
    """
    执行预测任务
    """
    result = system.predict_task(request.model_type, request.task_type, request.days)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@app.get("/")
def read_root():
    return {"status": "System is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
