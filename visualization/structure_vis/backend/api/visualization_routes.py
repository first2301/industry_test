from fastapi import APIRouter, HTTPException
import logging
from typing import List, Optional

from models.request_models import VisualizationRequest
from models.response_models import VisualizationResponse, ErrorResponse
from services.visualization_service import VisualizationService
from utils.file_utils import (
    get_data_from_session, get_session_metadata
)

logger = logging.getLogger(__name__)
router = APIRouter()

# 서비스 인스턴스
visualization_service = VisualizationService()

@router.get("/histogram/{session_id}/{column}")
async def create_histogram_comparison(session_id: str, column: str):
    """히스토그램 비교 차트를 생성합니다."""
    try:
        # 세션에서 데이터 가져오기
        original_df = get_data_from_session(session_id)
        if original_df is None:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        metadata = get_session_metadata(session_id)
        if not metadata or 'augmented_data' not in metadata:
            raise HTTPException(status_code=404, detail="증강된 데이터가 없습니다.")
        
        augmented_df = metadata['augmented_data']
        
        # 히스토그램 생성
        result = visualization_service.create_histogram_comparison(original_df, augmented_df, column)
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return VisualizationResponse(
            success=True,
            chart_data=result['chart_data'],
            chart_type=result['chart_type'],
            columns_used=[column]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating histogram for column {column} in session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="히스토그램 생성 중 오류가 발생했습니다.")

@router.get("/boxplot/{session_id}/{column}")
async def create_boxplot_comparison(session_id: str, column: str):
    """박스플롯 비교 차트를 생성합니다."""
    try:
        # 세션에서 데이터 가져오기
        original_df = get_data_from_session(session_id)
        if original_df is None:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        metadata = get_session_metadata(session_id)
        if not metadata or 'augmented_data' not in metadata:
            raise HTTPException(status_code=404, detail="증강된 데이터가 없습니다.")
        
        augmented_df = metadata['augmented_data']
        
        # 박스플롯 생성
        result = visualization_service.create_boxplot_comparison(original_df, augmented_df, column)
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return VisualizationResponse(
            success=True,
            chart_data=result['chart_data'],
            chart_type=result['chart_type'],
            columns_used=[column]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating boxplot for column {column} in session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="박스플롯 생성 중 오류가 발생했습니다.")

@router.get("/scatter/{session_id}")
async def create_scatter_comparison(session_id: str, x_column: str, y_column: str):
    """산점도 비교 차트를 생성합니다."""
    try:
        # 세션에서 데이터 가져오기
        original_df = get_data_from_session(session_id)
        if original_df is None:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        metadata = get_session_metadata(session_id)
        if not metadata or 'augmented_data' not in metadata:
            raise HTTPException(status_code=404, detail="증강된 데이터가 없습니다.")
        
        augmented_df = metadata['augmented_data']
        
        # 산점도 생성
        result = visualization_service.create_scatter_comparison(original_df, augmented_df, x_column, y_column)
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return VisualizationResponse(
            success=True,
            chart_data=result['chart_data'],
            chart_type=result['chart_type'],
            columns_used=[x_column, y_column]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating scatter plot for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="산점도 생성 중 오류가 발생했습니다.")

@router.get("/categorical/{session_id}/{column}")
async def create_categorical_comparison(session_id: str, column: str):
    """범주형 데이터 비교 차트를 생성합니다."""
    try:
        # 세션에서 데이터 가져오기
        original_df = get_data_from_session(session_id)
        if original_df is None:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        metadata = get_session_metadata(session_id)
        if not metadata or 'augmented_data' not in metadata:
            raise HTTPException(status_code=404, detail="증강된 데이터가 없습니다.")
        
        augmented_df = metadata['augmented_data']
        
        # 범주형 비교 생성
        result = visualization_service.create_categorical_comparison(original_df, augmented_df, column)
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return {
            "success": True,
            "chart_type": result['chart_type'],
            "column": result['column'],
            "comparison_data": result['comparison_data'],
            "categories": result['categories']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating categorical comparison for column {column} in session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="범주형 비교 생성 중 오류가 발생했습니다.")

@router.get("/summary/{session_id}")
async def get_comparison_summary(session_id: str, numeric_columns: Optional[List[str]] = None):
    """비교 요약 통계를 생성합니다."""
    try:
        # 세션에서 데이터 가져오기
        original_df = get_data_from_session(session_id)
        if original_df is None:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        metadata = get_session_metadata(session_id)
        if not metadata or 'augmented_data' not in metadata:
            raise HTTPException(status_code=404, detail="증강된 데이터가 없습니다.")
        
        augmented_df = metadata['augmented_data']
        
        # 수치형 컬럼이 지정되지 않은 경우 자동으로 찾기
        if numeric_columns is None:
            from services.data_service import DataService
            data_service = DataService()
            analysis_result = data_service.analyze_data(original_df)
            if analysis_result['success']:
                numeric_columns = analysis_result['numeric_columns']
            else:
                raise HTTPException(status_code=500, detail="수치형 컬럼 분석 중 오류가 발생했습니다.")
        
        # 요약 통계 생성
        result = visualization_service.get_comparison_summary(original_df, augmented_df, numeric_columns)
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return {
            "success": True,
            "summary_stats": result['summary_stats'],
            "columns_analyzed": result['columns_analyzed']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting comparison summary for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="비교 요약 생성 중 오류가 발생했습니다.")

@router.get("/report/{session_id}")
async def create_augmentation_report(session_id: str):
    """증강 결과 리포트를 생성합니다."""
    try:
        # 세션에서 데이터 가져오기
        original_df = get_data_from_session(session_id)
        if original_df is None:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        metadata = get_session_metadata(session_id)
        if not metadata or 'augmented_data' not in metadata:
            raise HTTPException(status_code=404, detail="증강된 데이터가 없습니다.")
        
        augmented_df = metadata['augmented_data']
        params = metadata.get('augmentation_params', {})
        
        # 리포트 생성
        result = visualization_service.create_augmentation_report(original_df, augmented_df, params)
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return {
            "success": True,
            "report": result['report']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating augmentation report for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="증강 리포트 생성 중 오류가 발생했습니다.")

@router.get("/available-charts/{session_id}")
async def get_available_charts(session_id: str):
    """사용 가능한 차트 목록을 반환합니다."""
    try:
        # 세션에서 데이터 가져오기
        original_df = get_data_from_session(session_id)
        if original_df is None:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        metadata = get_session_metadata(session_id)
        if not metadata or 'augmented_data' not in metadata:
            raise HTTPException(status_code=404, detail="증강된 데이터가 없습니다.")
        
        # 데이터 분석
        from services.data_service import DataService
        data_service = DataService()
        analysis_result = data_service.analyze_data(original_df)
        
        if not analysis_result['success']:
            raise HTTPException(status_code=500, detail="데이터 분석 중 오류가 발생했습니다.")
        
        available_charts = {
            "numeric_charts": {
                "histogram": analysis_result['numeric_columns'],
                "boxplot": analysis_result['numeric_columns'],
                "scatter": analysis_result['numeric_columns'] if len(analysis_result['numeric_columns']) >= 2 else []
            },
            "categorical_charts": {
                "categorical_comparison": analysis_result['categorical_columns']
            },
            "summary_charts": {
                "comparison_summary": analysis_result['numeric_columns'],
                "augmentation_report": ["all"]
            }
        }
        
        return {
            "success": True,
            "available_charts": available_charts,
            "numeric_columns": analysis_result['numeric_columns'],
            "categorical_columns": analysis_result['categorical_columns']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting available charts for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="사용 가능한 차트 목록 조회 중 오류가 발생했습니다.") 