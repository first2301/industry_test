from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from io import StringIO
from services.clf_automl import compare_clf_models
from services.reg_automl import compare_reg_models
from services.cluster_automl import compare_cluster_models
import pandas as pd
import json

router = APIRouter()

@router.post('/classification')
async def new_clf(request: Request):
    try:
        data = await request.json()
        df = pd.read_json(StringIO(data['json_data']))

        reg_compare = compare_clf_models(df, data['target'])
        dumps_data = json.dumps(reg_compare)
        print('result_data: ', dumps_data)

        return JSONResponse(content={'result': dumps_data})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={'error': f'분류 모델 학습 오류: {str(e)}'}
        )

@router.post('/regression')
async def new_reg(request: Request):
    try:
        data = await request.json()
        df = pd.read_json(StringIO(data['json_data']))
        
        reg_compare = compare_reg_models(df, data['target'], n_trials=5)
        dumps_data = json.dumps(reg_compare)
        print('result_data: ', dumps_data)
        return JSONResponse(content={'result': dumps_data})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={'error': f'회귀 모델 학습 오류: {str(e)}'}
        )

@router.post('/clustering')
async def new_cluster(request: Request):
    try:
        data = await request.json()
        df = pd.read_json(StringIO(data['json_data']))
        
        cluster_compare = compare_cluster_models(df, n_trials=5)
        dumps_data = json.dumps(cluster_compare)
        print('result_data: ', dumps_data)
        return JSONResponse(content={'result': dumps_data})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={'error': f'군집화 모델 학습 오류: {str(e)}'}
        )