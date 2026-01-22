import uvicorn
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("启动自动化金融模型训练系统...")
    # 启动 FastAPI 服务
    uvicorn.run("system_integration.api.main:app", host="0.0.0.0", port=8000, reload=True)
