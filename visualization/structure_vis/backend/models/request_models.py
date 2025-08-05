from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class AugmentationMethod(str, Enum):
    NOISE = "noise"
    DUPLICATE = "duplicate"
    FEATURE = "feature"
    SMOTE = "smote"
    GENERAL = "general"

class AugmentationRequest(BaseModel):
    session_id: str = Field(..., description="세션 ID")
    methods: List[AugmentationMethod] = Field(default=["noise", "duplicate", "feature"], description="사용할 증강 방법들")
    noise_level: float = Field(default=0.03, ge=0.01, le=0.2, description="노이즈 레벨")
    dup_count: int = Field(default=2, ge=2, le=10, description="중복 횟수")
    augmentation_ratio: float = Field(default=0.5, ge=0.1, le=2.0, description="증강 비율")
    target_col: Optional[str] = Field(None, description="SMOTE용 타겟 컬럼")
    imb_method: Optional[str] = Field(None, description="불균형 처리 방법")

class VisualizationRequest(BaseModel):
    session_id: str = Field(..., description="세션 ID")
    chart_type: str = Field(..., description="차트 타입")
    columns: List[str] = Field(..., description="사용할 컬럼들")
    params: Optional[Dict[str, Any]] = Field(default={}, description="추가 파라미터")

class DataAnalysisRequest(BaseModel):
    session_id: str = Field(..., description="세션 ID") 