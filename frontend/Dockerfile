# 构建阶段
FROM node:20-alpine as build

WORKDIR /app

# 设置国内 npm 镜像源
RUN npm config set registry https://registry.npmmirror.com

# 复制 package.json
COPY package*.json ./

# 安装依赖
RUN npm install

# 检查依赖安装情况
RUN ls -la node_modules

# 复制源代码
COPY . .

# 检查构建前的文件结构
RUN ls -la

# 构建应用
RUN npm run build

# 生产阶段 (使用 Nginx 提供静态文件服务)
FROM nginx:alpine

# 复制构建产物到 Nginx 目录
COPY --from=build /app/dist /usr/share/nginx/html

# 复制自定义 Nginx 配置 (可选，用于处理 React 路由)
# COPY nginx.conf /etc/nginx/conf.d/default.conf

# 暴露端口
EXPOSE 80

# 启动 Nginx
CMD ["nginx", "-g", "daemon off;"]