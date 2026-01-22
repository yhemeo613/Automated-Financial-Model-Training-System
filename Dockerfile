# 第一阶段：构建环境
FROM python:3.10-slim AS builder

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libc-dev \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# 创建并激活虚拟环境
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# 安装Python依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 第二阶段：生产环境
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 复制虚拟环境和应用代码
COPY --from=builder /venv /venv
COPY . .

# 设置环境变量
ENV PATH="/venv/bin:$PATH"

# 清理不必要的文件
RUN rm -rf /venv/lib/python3.10/site-packages/*/__pycache__ \
    && rm -rf /venv/lib/python3.10/site-packages/*/*.pyc \
    && rm -rf /venv/lib/python3.10/site-packages/*/*.pyo

# 暴露端口
EXPOSE 8000

# 启动应用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]