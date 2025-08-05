from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class DataAnalysisResponse(BaseModel):
    success: bool = Field(..., description="성공 여부")
    data_shape: Dict[str, int] = Field(..., description="데이터 형태 (행, 열)")
    numeric_columns: List[str] = Field(..., description="수치형 컬럼 목록")
    categorical_columns: List[str] = Field(..., description="범주형 컬럼 목록")
    missing_data: Dict[str, int] = Field(..., description="결측값 정보")
    duplicate_count: int = Field(..., description="중복 행 수")
    column_info: List[Dict[str, Any]] = Field(..., description="컬럼 상세 정보")

class AugmentationResponse(BaseModel):
    success: bool = Field(..., description="성공 여부")
    original_shape: Dict[str, int] = Field(..., description="원본 데이터 형태")
    augmented_shape: Dict[str, int] = Field(..., description="증강된 데이터 형태")
    augmentation_ratio: float = Field(..., description="실제 증강 비율")
    processing_time: float = Field(..., description="처리 시간 (초)")
    methods_used: List[str] = Field(..., description="사용된 증강 방법들")

class VisualizationResponse(BaseModel):
    success: bool = Field(..., description="성공 여부")
    chart_data: Dict[str, Any] = Field(..., description="차트 데이터")
    chart_type: str = Field(..., description="차트 타입")
    columns_used: List[str] = Field(..., description="사용된 컬럼들")

class ErrorResponse(BaseModel):
    success: bool = Field(default=False, description="성공 여부")
    error: str = Field(..., description="에러 메시지")
    error_code: Optional[str] = Field(None, description="에러 코드")

class UploadResponse(BaseModel):
    success: bool = Field(..., description="성공 여부")
    session_id: str = Field(..., description="세션 ID")
    file_size: int = Field(..., description="파일 크기 (bytes)")
    row_count: int = Field(..., description="행 수")
    column_count: int = Field(..., description="열 수") 