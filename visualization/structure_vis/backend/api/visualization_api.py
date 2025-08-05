"""
시각화 API 엔드포인트
services의 visualization_service를 호출하여 시각화 기능을 제공합니다.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Dict, Any, Optional
import pandas as pd
import json
from io import StringIO

from ..services.visualization_service import VisualizationService

router = APIRouter(prefix="/visualization", tags=["visualization"])
visualization_service = VisualizationService()


@router.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    """데이터 파일을 업로드합니다."""
    try:
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다. CSV 또는 Excel 파일을 업로드해주세요.")
        
        content = await file.read()
        
        if file.filename.endswith('.csv'):
            # CSV 파일 처리
            text_content = content.decode('utf-8')
            df = pd.read_csv(StringIO(text_content))
        else:
            # Excel 파일 처리
            df = pd.read_excel(content)
        
        # 컬럼 타입 분석
        numeric_cols = visualization_service.get_numeric_columns(df)
        categorical_cols = visualization_service.get_categorical_columns(df)
        
        return {
            "success": True,
            "message": "데이터 업로드 성공",
            "data_info": {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict(),
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols
            },
            "data": df.to_dict(orient="records")
        }
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="파일 인코딩 오류입니다. UTF-8 인코딩으로 저장된 파일을 업로드해주세요.")
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="빈 파일입니다. 데이터가 포함된 파일을 업로드해주세요.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터 업로드 실패: {str(e)}")


@router.post("/get-column-types")
async def get_column_types(data: List[Dict[str, Any]]):
    """데이터의 컬럼 타입을 분석합니다."""
    try:
        df = pd.DataFrame(data)
        
        numeric_cols = visualization_service.get_numeric_columns(df)
        categorical_cols = visualization_service.get_categorical_columns(df)
        
        return {
            "success": True,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "total_columns": len(df.columns)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"컬럼 타입 분석 실패: {str(e)}")


@router.post("/create-histogram-comparison")
async def create_histogram_comparison(
    original_data: List[Dict[str, Any]],
    augmented_data: List[Dict[str, Any]],
    column: str
):
    """히스토그램 비교 차트를 생성합니다."""
    try:
        df_orig = pd.DataFrame(original_data)
        df_aug = pd.DataFrame(augmented_data)
        
        if column not in df_orig.columns or column not in df_aug.columns:
            raise HTTPException(status_code=400, detail=f"컬럼 '{column}'이 데이터에 존재하지 않습니다.")
        
        result = visualization_service.create_histogram_comparison(df_orig, df_aug, column)
        
        return {
            "success": True,
            "chart_type": "histogram_comparison",
            "column": column,
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"히스토그램 생성 실패: {str(e)}")


@router.post("/create-boxplot-comparison")
async def create_boxplot_comparison(
    original_data: List[Dict[str, Any]],
    augmented_data: List[Dict[str, Any]],
    column: str
):
    """박스플롯 비교 차트를 생성합니다."""
    try:
        df_orig = pd.DataFrame(original_data)
        df_aug = pd.DataFrame(augmented_data)
        
        if column not in df_orig.columns or column not in df_aug.columns:
            raise HTTPException(status_code=400, detail=f"컬럼 '{column}'이 데이터에 존재하지 않습니다.")
        
        result = visualization_service.create_boxplot_comparison(df_orig, df_aug, column)
        
        return {
            "success": True,
            "chart_type": "boxplot_comparison",
            "column": column,
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"박스플롯 생성 실패: {str(e)}")


@router.post("/create-scatter-comparison")
async def create_scatter_comparison(
    original_data: List[Dict[str, Any]],
    augmented_data: List[Dict[str, Any]],
    x_column: str,
    y_column: str
):
    """산점도 비교 차트를 생성합니다."""
    try:
        df_orig = pd.DataFrame(original_data)
        df_aug = pd.DataFrame(augmented_data)
        
        if x_column not in df_orig.columns or y_column not in df_orig.columns:
            raise HTTPException(status_code=400, detail=f"컬럼 '{x_column}' 또는 '{y_column}'이 데이터에 존재하지 않습니다.")
        
        result = visualization_service.create_scatter_comparison(df_orig, df_aug, x_column, y_column)
        
        return {
            "success": True,
            "chart_type": "scatter_comparison",
            "x_column": x_column,
            "y_column": y_column,
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"산점도 생성 실패: {str(e)}")


@router.post("/create-categorical-chart")
async def create_categorical_chart(
    data: List[Dict[str, Any]],
    column: str,
    chart_type: str = "막대그래프"
):
    """범주형 차트를 생성합니다."""
    try:
        df = pd.DataFrame(data)
        
        if column not in df.columns:
            raise HTTPException(status_code=400, detail=f"컬럼 '{column}'이 데이터에 존재하지 않습니다.")
        
        result = visualization_service.create_categorical_chart(df, column, chart_type)
        
        return {
            "success": True,
            "chart_type": "categorical",
            "column": column,
            "chart_style": chart_type,
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"범주형 차트 생성 실패: {str(e)}")


@router.post("/create-comparison-dashboard")
async def create_comparison_dashboard(
    original_data: List[Dict[str, Any]],
    augmented_data: List[Dict[str, Any]],
    numeric_columns: List[str],
    categorical_columns: List[str]
):
    """비교 대시보드를 생성합니다."""
    try:
        df_orig = pd.DataFrame(original_data)
        df_aug = pd.DataFrame(augmented_data)
        
        result = visualization_service.create_comparison_dashboard(
            df_orig, df_aug, numeric_columns, categorical_columns
        )
        
        return {
            "success": True,
            "chart_type": "comparison_dashboard",
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"대시보드 생성 실패: {str(e)}")


@router.post("/get-comparison-summary")
async def get_comparison_summary(
    original_data: List[Dict[str, Any]],
    augmented_data: List[Dict[str, Any]],
    numeric_columns: List[str]
):
    """비교 요약 정보를 반환합니다."""
    try:
        df_orig = pd.DataFrame(original_data)
        df_aug = pd.DataFrame(augmented_data)
        
        result = visualization_service.get_comparison_summary(df_orig, df_aug, numeric_columns)
        
        return {
            "success": True,
            "summary": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"요약 정보 생성 실패: {str(e)}") 