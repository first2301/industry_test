from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import data_routes, augmentation_routes, visualization_routes

app = FastAPI(
    title="Data Augmentation API",
    description="CSV 데이터 증강 및 시각화를 위한 FastAPI 백엔드",
    version="1.0.0"
)

# CORS 설정 (Streamlit과 통신을 위해)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],  # Streamlit 기본 포트
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(data_routes.router, prefix="/api/data", tags=["data"])
app.include_router(augmentation_routes.router, prefix="/api/augmentation", tags=["augmentation"])
app.include_router(visualization_routes.router, prefix="/api/visualization", tags=["visualization"])

@app.get("/")
async def root():
    return {"message": "Data Augmentation API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
