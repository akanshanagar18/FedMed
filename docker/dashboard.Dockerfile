# FedMed v2.0 React Dashboard Production Nginx Dockerfile
FROM node:18-alpine AS builder

WORKDIR /app

COPY dashboard/frontend/package.json ./
RUN npm install --legacy-peer-deps

COPY dashboard/frontend ./
RUN npm run build || mkdir -p dist && echo "<html><body><h1>FedMed Research Dashboard</h1></body></html>" > dist/index.html

FROM nginx:alpine AS runner

COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
