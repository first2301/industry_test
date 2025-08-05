from fastapi import APIRouter, HTTPException
import logging
from typing import List

from models.request_models import AugmentationRequest
from models.response_models import AugmentationResponse, ErrorResponse
from services.augmentation_service import AugmentationService
from utils.file_utils import (
    get_data_from_session, get_session_metadata, update_session_metadata
)

logger = logging.getLogger(__name__)
router = APIRouter()

# 서비스 인스턴스
augmentation_service = AugmentationService()

@router.post("/process", response_model=AugmentationResponse)
async def process_augmentation(request: AugmentationRequest):
    """데이터 증강을 실행합니다."""
    try:
        # 세션에서 원본 데이터 가져오기
        original_df = get_data_from_session(request.session_id)
        if original_df is None:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        # 파라미터 검증
        validation_result = augmentation_service.validate_augmentation_params(
            methods=request.methods,
            noise_level=request.noise_level,
            dup_count=request.dup_count,
            augmentation_ratio=request.augmentation_ratio,
            target_col=request.target_col,
            imb_method=request.imb_method
        )
        
        if not validation_result['valid']:
            raise HTTPException(status_code=400, detail=validation_result['error'])
        
        # 예상 처리 시간 계산
        time_estimate = augmentation_service.estimate_processing_time(
            original_df, request.methods,
            noise_level=request.noise_level,
            dup_count=request.dup_count,
            augmentation_ratio=request.augmentation_ratio
        )
        
        # 증강 파라미터 준비
        params = {
            'noise_level': request.noise_level,
            'dup_count': request.dup_count,
            'augmentation_ratio': request.augmentation_ratio
        }
        
        if request.target_col:
            params['target_col'] = request.target_col
        if request.imb_method:
            params['imb_method'] = request.imb_method
        
        # 증강 실행
        augmentation_result = augmentation_service.augment_data(
            original_df, request.methods, **params
        )
        
        if not augmentation_result['success']:
            raise HTTPException(status_code=500, detail=augmentation_result['error'])
        
        # 증강된 데이터를 세션에 저장
        metadata = get_session_metadata(request.session_id) or {}
        metadata['augmented_data'] = augmentation_result['augmented_data']
        metadata['augmentation_params'] = params
        metadata['augmentation_result'] = {
            'original_shape': augmentation_result['original_shape'],
            'augmented_shape': augmentation_result['augmented_shape'],
            'processing_time': augmentation_result['processing_time'],
            'methods_used': augmentation_result['methods_used']
        }
        
        update_session_metadata(request.session_id, metadata)
        
        logger.info(f"Augmentation completed for session {request.session_id}")
        
        return AugmentationResponse(
            success=True,
            original_shape=augmentation_result['original_shape'],
            augmented_shape=augmentation_result['augmented_shape'],
            augmentation_ratio=augmentation_result['augmentation_ratio'],
            processing_time=augmentation_result['processing_time'],
            methods_used=augmentation_result['methods_used']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing augmentation for session {request.session_id}: {e}")
        raise HTTPException(status_code=500, detail="데이터 증강 중 오류가 발생했습니다.")

@router.get("/estimate-time/{session_id}")
async def estimate_processing_time(session_id: str, methods: List[str], **params):
    """증강 처리 시간을 예측합니다."""
    try:
        # 세션에서 데이터 가져오기
        df = get_data_from_session(session_id)
        if df is None:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        # 처리 시간 예측
        time_estimate = augmentation_service.estimate_processing_time(df, methods, **params)
        
        if not time_estimate['success']:
            raise HTTPException(status_code=500, detail=time_estimate['error'])
        
        return {
            "success": True,
            "estimated_time_seconds": time_estimate['estimated_time_seconds'],
            "estimated_time_minutes": time_estimate['estimated_time_minutes'],
            "methods": methods,
            "data_size": len(df)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error estimating processing time for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="처리 시간 예측 중 오류가 발생했습니다.")

@router.get("/summary/{session_id}")
async def get_augmentation_summary(session_id: str):
    """증강 결과 요약을 반환합니다."""
    try:
        # 세션에서 데이터 가져오기
        original_df = get_data_from_session(session_id)
        if original_df is None:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        metadata = get_session_metadata(session_id)
        if not metadata or 'augmented_data' not in metadata:
            raise HTTPException(status_code=404, detail="증강된 데이터가 없습니다.")
        
        augmented_df = metadata['augmented_data']
        
        # 요약 생성
        summary_result = augmentation_service.get_augmentation_summary(original_df, augmented_df)
        
        if not summary_result['success']:
            raise HTTPException(status_code=500, detail=summary_result['error'])
        
        return {
            "success": True,
            "summary": summary_result['summary'],
            "augmentation_params": metadata.get('augmentation_params', {}),
            "processing_info": metadata.get('augmentation_result', {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting augmentation summary for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="증강 요약 생성 중 오류가 발생했습니다.")

@router.get("/status/{session_id}")
async def get_augmentation_status(session_id: str):
    """증강 상태를 확인합니다."""
    try:
        metadata = get_session_metadata(session_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        has_augmented_data = 'augmented_data' in metadata
        
        return {
            "success": True,
            "session_id": session_id,
            "has_augmented_data": has_augmented_data,
            "augmentation_params": metadata.get('augmentation_params', {}),
            "processing_info": metadata.get('augmentation_result', {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting augmentation status for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="증강 상태 확인 중 오류가 발생했습니다.") 