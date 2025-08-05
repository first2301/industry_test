from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import pandas as pd
import io
import logging
from typing import Optional

from models.request_models import DataAnalysisRequest
from models.response_models import (
    DataAnalysisResponse, UploadResponse, ErrorResponse
)
from services.data_service import DataService
from utils.file_utils import (
    generate_session_id, store_data_in_session, get_data_from_session,
    validate_csv_file, get_session_metadata, update_session_metadata
)

logger = logging.getLogger(__name__)
router = APIRouter()

# 서비스 인스턴스
data_service = DataService()

@router.post("/upload", response_model=UploadResponse)
async def upload_csv_file(file: UploadFile = File(...)):
    """CSV 파일을 업로드하고 세션에 저장합니다."""
    try:
        # 파일 검증
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="CSV 파일만 업로드 가능합니다.")
        
        # 파일 내용 읽기
        file_content = await file.read()
        
        # 파일 검증
        validation_result = validate_csv_file(file_content)
        if not validation_result['valid']:
            raise HTTPException(status_code=400, detail=validation_result['error'])
        
        # CSV 파싱
        df = pd.read_csv(io.BytesIO(file_content))
        
        # 데이터 검증
        validation_result = data_service.validate_data(df)
        if not validation_result['valid']:
            raise HTTPException(status_code=400, detail=validation_result['error'])
        
        # 세션 ID 생성
        session_id = generate_session_id()
        
        # 메타데이터 생성
        metadata = {
            'filename': file.filename,
            'file_size': len(file_content),
            'row_count': len(df),
            'column_count': len(df.columns),
            'memory_usage_mb': validation_result['memory_usage_mb']
        }
        
        # 세션에 데이터 저장
        if not store_data_in_session(session_id, df, metadata):
            raise HTTPException(status_code=500, detail="데이터 저장 중 오류가 발생했습니다.")
        
        logger.info(f"File uploaded successfully: {file.filename}, Session: {session_id}")
        
        return UploadResponse(
            success=True,
            session_id=session_id,
            file_size=len(file_content),
            row_count=len(df),
            column_count=len(df.columns)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail="파일 업로드 중 오류가 발생했습니다.")

@router.get("/analyze/{session_id}", response_model=DataAnalysisResponse)
async def analyze_data(session_id: str):
    """세션의 데이터를 분석합니다."""
    try:
        # 세션에서 데이터 가져오기
        df = get_data_from_session(session_id)
        if df is None:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        # 데이터 분석
        analysis_result = data_service.analyze_data(df)
        if not analysis_result['success']:
            raise HTTPException(status_code=500, detail=analysis_result['error'])
        
        return DataAnalysisResponse(
            success=True,
            data_shape=analysis_result['data_shape'],
            numeric_columns=analysis_result['numeric_columns'],
            categorical_columns=analysis_result['categorical_columns'],
            missing_data=analysis_result['missing_data'],
            duplicate_count=analysis_result['duplicate_count'],
            column_info=analysis_result['column_info']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing data for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="데이터 분석 중 오류가 발생했습니다.")

@router.get("/preview/{session_id}")
async def get_data_preview(session_id: str, rows: int = 10):
    """데이터 미리보기를 반환합니다."""
    try:
        # 세션에서 데이터 가져오기
        df = get_data_from_session(session_id)
        if df is None:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        # 미리보기 생성
        preview_result = data_service.get_data_preview(df, rows)
        if not preview_result['success']:
            raise HTTPException(status_code=500, detail=preview_result['error'])
        
        return {
            "success": True,
            "preview_data": preview_result['preview_data'],
            "total_rows": preview_result['total_rows'],
            "total_columns": preview_result['total_columns']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting preview for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="미리보기 생성 중 오류가 발생했습니다.")

@router.get("/statistics/{session_id}/{column}")
async def get_column_statistics(session_id: str, column: str):
    """특정 컬럼의 통계 정보를 반환합니다."""
    try:
        # 세션에서 데이터 가져오기
        df = get_data_from_session(session_id)
        if df is None:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        # 통계 계산
        stats_result = data_service.get_column_statistics(df, column)
        if not stats_result['success']:
            raise HTTPException(status_code=400, detail=stats_result['error'])
        
        return {
            "success": True,
            "column": column,
            "statistics": stats_result['statistics']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting statistics for column {column} in session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="통계 계산 중 오류가 발생했습니다.")

@router.get("/download/{session_id}")
async def download_data(session_id: str, data_type: str = "original"):
    """데이터를 CSV로 다운로드합니다."""
    try:
        # 세션에서 데이터 가져오기
        df = get_data_from_session(session_id)
        if df is None:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        # 증강된 데이터가 요청된 경우
        if data_type == "augmented":
            metadata = get_session_metadata(session_id)
            if metadata and 'augmented_data' in metadata:
                df = metadata['augmented_data']
            else:
                raise HTTPException(status_code=404, detail="증강된 데이터가 없습니다.")
        
        # CSV로 변환
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        # 파일명 생성
        filename = f"data_{data_type}_{session_id[:8]}.csv"
        
        return StreamingResponse(
            io.StringIO(csv_content),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading data for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="데이터 다운로드 중 오류가 발생했습니다.")

@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """세션을 삭제합니다."""
    try:
        from utils.file_utils import delete_session as delete_session_util
        
        if delete_session_util(session_id):
            return {"success": True, "message": "세션이 삭제되었습니다."}
        else:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="세션 삭제 중 오류가 발생했습니다.") 