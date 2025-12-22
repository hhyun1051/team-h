#!/bin/bash
# FastAPI 서버 실행 스크립트

echo "[🚀] Starting Team-H FastAPI Server..."
echo ""
echo "Server will be available at: http://0.0.0.0:8000"
echo "Docs at: http://0.0.0.0:8000/docs"
echo ""

uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
