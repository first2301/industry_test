"""
시각화 서비스 레이어
lib의 visualization 모듈을 호출하여 시각화 기능을 제공합니다.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
import json
import base64
from io import BytesIO

from ..lib.visualization import DataVisualizer


class VisualizationService:
    """시각화 서비스 클래스"""
    
    def __init__(self):
        """초기화"""
        self.visualizer = DataVisualizer()
    
    def get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """수치형 컬럼 목록을 반환합니다."""
        return self.visualizer.get_numeric_columns(df)
    
    def get_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """범주형 컬럼 목록을 반환합니다."""
        return self.visualizer.get_categorical_columns(df)
    
    def create_histogram_comparison(self, df_orig: pd.DataFrame, df_aug: pd.DataFrame, column: str) -> Dict[str, Any]:
        """히스토그램 비교 차트를 생성합니다."""
        fig = self.visualizer.create_overlapping_histogram(df_orig, df_aug, column)
        return self._fig_to_dict(fig)
    
    def create_boxplot_comparison(self, df_orig: pd.DataFrame, df_aug: pd.DataFrame, column: str) -> Dict[str, Any]:
        """박스플롯 비교 차트를 생성합니다."""
        fig = self.visualizer.create_overlapping_boxplot(df_orig, df_aug, column)
        return self._fig_to_dict(fig)
    
    def create_scatter_comparison(self, df_orig: pd.DataFrame, df_aug: pd.DataFrame, x_col: str, y_col: str) -> Dict[str, Any]:
        """산점도 비교 차트를 생성합니다."""
        fig = self.visualizer.create_overlapping_scatter(df_orig, df_aug, x_col, y_col)
        return self._fig_to_dict(fig)
    
    def create_categorical_chart(self, df: pd.DataFrame, column: str, chart_type: str = "막대그래프") -> Dict[str, Any]:
        """범주형 차트를 생성합니다."""
        fig = self.visualizer.create_categorical_visualization(df, column, chart_type)
        return self._fig_to_dict(fig)
    
    def create_comparison_dashboard(self, df_orig: pd.DataFrame, df_aug: pd.DataFrame, 
                                  numeric_cols: List[str], categorical_cols: List[str]) -> Dict[str, Any]:
        """비교 대시보드를 생성합니다."""
        fig = self.visualizer.create_comparison_dashboard(df_orig, df_aug, numeric_cols, categorical_cols)
        return self._fig_to_dict(fig)
    
    def create_augmentation_report(self, df_orig: pd.DataFrame, df_aug: pd.DataFrame, 
                                 params: Dict[str, Any]) -> Dict[str, Any]:
        """증강 리포트를 생성합니다."""
        fig = self.visualizer.create_augmentation_report(df_orig, df_aug, params)
        return self._fig_to_dict(fig)
    
    def get_comparison_summary(self, df_orig: pd.DataFrame, df_aug: pd.DataFrame, 
                             numeric_cols: List[str]) -> Dict[str, Any]:
        """비교 요약 정보를 반환합니다."""
        summary = self.visualizer.display_comparison_summary(df_orig, df_aug, numeric_cols)
        return {"summary": summary}
    
    def _fig_to_dict(self, fig) -> Dict[str, Any]:
        """Plotly Figure를 딕셔너리로 변환합니다."""
        # JSON으로 직렬화 가능한 형태로 변환
        fig_dict = fig.to_dict()
        
        # 이미지 변환 시도 (kaleido 의존성이 있을 경우)
        img_base64 = None
        try:
            img_bytes = fig.to_image(format="png")
            img_base64 = base64.b64encode(img_bytes).decode()
        except Exception:
            # kaleido가 설치되지 않았거나 이미지 생성에 실패한 경우
            pass
        
        return {
            "figure": fig_dict,
            "image_base64": img_base64,
            "layout": fig_dict.get("layout", {}),
            "data": fig_dict.get("data", [])
        } 