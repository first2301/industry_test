from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from io import StringIO
import pandas as pd
import json
import logging
from ..services.flaml import FlamlRegression, FlamlClustering, FlamlClassification

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post('/flaml/regression')
async def flaml_regression(request: Request):
    try:
        data = await request.json()
        logger.info(f"FLAML 회귀 요청 데이터: {data.keys()}")
        
        # 데이터 검증
        if 'json_data' not in data or 'target' not in data:
            return JSONResponse(
                status_code=400,
                content={'error': '필수 필드가 누락되었습니다: json_data, target'}
            )
        
        df = pd.read_json(StringIO(data['json_data']))
        target = data['target']
        logger.info(f"데이터프레임 형태: {df.shape}")
        
        # 수치형 데이터만 선택
        numeric_df = df.select_dtypes(include=['number'])
        if target not in numeric_df.columns:
            return JSONResponse(
                status_code=400,
                content={'error': f'타겟 컬럼 "{target}"이 수치형이 아닙니다.'}
            )
        
        # 결측값 처리
        numeric_df = numeric_df.dropna()
        if len(numeric_df) == 0:
            return JSONResponse(
                status_code=400,
                content={'error': '결측값 제거 후 데이터가 없습니다.'}
            )
        
        flaml_reg = FlamlRegression(numeric_df, target)
        result = flaml_reg.run_automl()
        dumps_data = json.dumps(result)
        logger.info('FLAML 회귀 모델 학습 완료')
        
        return JSONResponse(content={'result': dumps_data})
    except Exception as e:
        logger.error(f"FLAML 회귀 모델 학습 오류: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={'error': f'FLAML 회귀 모델 학습 오류: {str(e)}'}
        )

@router.post('/flaml/clustering')
async def flaml_clustering(request: Request):
    try:
        data = await request.json()
        logger.info(f"FLAML 군집화 요청 데이터: {data.keys()}")
        
        # 데이터 검증
        if 'json_data' not in data:
            return JSONResponse(
                status_code=400,
                content={'error': '필수 필드가 누락되었습니다: json_data'}
            )
        
        df = pd.read_json(StringIO(data['json_data']))
        logger.info(f"데이터프레임 형태: {df.shape}")
        
        # 수치형 데이터만 선택
        numeric_df = df.select_dtypes(include=['number'])
        if len(numeric_df.columns) == 0:
            return JSONResponse(
                status_code=400,
                content={'error': '수치형 데이터가 없습니다.'}
            )
        
        # 결측값 처리
        numeric_df = numeric_df.dropna()
        if len(numeric_df) == 0:
            return JSONResponse(
                status_code=400,
                content={'error': '결측값 제거 후 데이터가 없습니다.'}
            )
        
        flaml_cluster = FlamlClustering(numeric_df)
        result = flaml_cluster.run_automl()
        dumps_data = json.dumps(result)
        logger.info('FLAML 군집화 모델 학습 완료')
        
        return JSONResponse(content={'result': dumps_data})
    except Exception as e:
        logger.error(f"FLAML 군집화 모델 학습 오류: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={'error': f'FLAML 군집화 모델 학습 오류: {str(e)}'}
        )

@router.post('/flaml/classification')
async def flaml_classification(request: Request):
    try:
        data = await request.json()
        logger.info(f"FLAML 분류 요청 데이터: {data.keys()}")

        # 데이터 검증
        if 'json_data' not in data or 'target' not in data:
            return JSONResponse(
                status_code=400,
                content={'error': '필수 필드가 누락되었습니다: json_data, target'}
            )

        df = pd.read_json(StringIO(data['json_data']))
        target = data['target']
        logger.info(f"데이터프레임 형태: {df.shape}, 타겟: {target}")

        # 타겟 컬럼 존재 여부 확인
        if target not in df.columns:
            return JSONResponse(
                status_code=400,
                content={'error': f'타겟 컬럼 "{target}"이 데이터에 존재하지 않습니다.'}
            )

        # 결측값 처리
        df = df.dropna()
        if len(df) == 0:
            return JSONResponse(
                status_code=400,
                content={'error': '결측값 제거 후 데이터가 없습니다.'}
            )

        flaml_clf = FlamlClassification(df, target)
        result = flaml_clf.run_automl()
        dumps_data = json.dumps(result)
        logger.info('FLAML 분류 모델 학습 완료')

        return JSONResponse(content={'result': dumps_data})
    except Exception as e:
        logger.error(f"FLAML 분류 모델 학습 오류: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={'error': f'FLAML 분류 모델 학습 오류: {str(e)}'}
        )
