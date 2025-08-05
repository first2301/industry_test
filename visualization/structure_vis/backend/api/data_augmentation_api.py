"""
데이터 증강 API 엔드포인트
services의 data_augmentation_service를 호출하여 데이터 증강 기능을 제공합니다.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Dict, Any, Optional
import pandas as pd
import json
from io import StringIO

from ..services.data_augmentation_service import DataAugmentationService

router = APIRouter(prefix="/augmentation", tags=["augmentation"])
augmentation_service = DataAugmentationService()


@router.get("/methods")
async def get_available_methods():
    """사용 가능한 증강 방법 목록을 반환합니다."""
    try:
        methods = augmentation_service.get_available_methods()
        return {
            "success": True,
            "available_methods": methods,
            "count": len(methods)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"증강 방법 목록 조회 실패: {str(e)}")


@router.post("/validate-params")
async def validate_augmentation_params(
    method: str,
    parameters: Dict[str, Any]
):
    """증강 파라미터의 유효성을 검증합니다."""
    try:
        result = augmentation_service.validate_augmentation_params(method, **parameters)
        return {
            "success": True,
            "validation_result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파라미터 검증 실패: {str(e)}")


@router.post("/preview")
async def get_augmentation_preview(
    data: List[Dict[str, Any]],
    method: str,
    sample_size: int = 100,
    parameters: Dict[str, Any] = {}
):
    """증강 결과 미리보기를 제공합니다."""
    try:
        df = pd.DataFrame(data)
        
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="데이터가 비어있습니다.")
        
        result = augmentation_service.get_augmentation_preview(
            df, method, sample_size, **parameters
        )
        
        return {
            "success": True,
            "preview_result": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"미리보기 생성 실패: {str(e)}")


@router.post("/augment")
async def augment_data(
    data: List[Dict[str, Any]],
    method: str,
    parameters: Dict[str, Any] = {}
):
    """데이터 증강을 수행합니다."""
    try:
        df = pd.DataFrame(data)
        
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="데이터가 비어있습니다.")
        
        # 파라미터 검증
        validation = augmentation_service.validate_augmentation_params(method, **parameters)
        if not validation.get("valid", False):
            raise HTTPException(status_code=400, detail=validation.get("error", "잘못된 파라미터입니다."))
        
        # 데이터 증강 수행
        result = augmentation_service.augment_data(df, method, **parameters)
        
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error", "데이터 증강 실패"))
        
        return {
            "success": True,
            "augmentation_result": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터 증강 실패: {str(e)}")


@router.post("/batch-augment")
async def batch_augment_data(
    data: List[Dict[str, Any]],
    augmentations: List[Dict[str, Any]]
):
    """
    여러 증강 방법을 순차적으로 적용합니다.
    
    augmentations: [
        {"method": "조합 증강", "parameters": {...}},
        {"method": "조합 증강", "parameters": {...}}
    ]
    """
    try:
        df = pd.DataFrame(data)
        
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="데이터가 비어있습니다.")
        
        results = []
        current_df = df.copy()
        
        for i, aug_config in enumerate(augmentations):
            method = aug_config.get("method")
            parameters = aug_config.get("parameters", {})
            
            if not method:
                raise HTTPException(status_code=400, detail=f"증강 설정 {i+1}에 method가 없습니다.")
            
            # 파라미터 검증
            validation = augmentation_service.validate_augmentation_params(method, **parameters)
            if not validation.get("valid", False):
                raise HTTPException(status_code=400, detail=f"증강 설정 {i+1} 파라미터 오류: {validation.get('error')}")
            
            # 데이터 증강 수행
            result = augmentation_service.augment_data(current_df, method, **parameters)
            
            if not result.get("success", False):
                raise HTTPException(status_code=500, detail=f"증강 설정 {i+1} 실패: {result.get('error')}")
            
            results.append({
                "step": i + 1,
                "method": method,
                "parameters": parameters,
                "result": result
            })
            
            # 다음 단계를 위해 증강된 데이터로 업데이트
            current_df = pd.DataFrame(result.get("augmented_data", []))
        
        return {
            "success": True,
            "batch_results": results,
            "final_data_info": {
                "original_rows": len(df),
                "final_rows": len(current_df),
                "total_augmentations": len(augmentations)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"배치 증강 실패: {str(e)}")


@router.get("/health")
async def health_check():
    """서비스 상태를 확인합니다."""
    try:
        methods = augmentation_service.get_available_methods()
        return {
            "status": "healthy",
            "available_methods_count": len(methods),
            "service": "data_augmentation"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "service": "data_augmentation"
        } 