import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List, Any, Optional
import logging

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lib import DataUtils, DataVisualizer

logger = logging.getLogger(__name__)

class DataService:
    def __init__(self):
        self.data_utils = DataUtils()
        self.visualizer = DataVisualizer()
    
    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """데이터를 분석하고 결과를 반환합니다."""
        try:
            # 컬럼 타입 분석
            numeric_cols = self.visualizer.get_numeric_columns(df)
            categorical_cols = self.visualizer.get_categorical_columns(df)
            
            # 결측값 분석
            missing_data = df.isnull().sum().to_dict()
            
            # 중복값 분석
            duplicate_count = df.duplicated().sum()
            
            # 컬럼 정보 생성
            column_info = []
            for col in df.columns:
                col_type = "Numeric" if col in numeric_cols else "Categorical"
                unique_count = df[col].nunique()
                missing_count = df[col].isnull().sum()
                column_info.append({
                    "column_name": col,
                    "data_type": col_type,
                    "unique_count": unique_count,
                    "missing_count": missing_count,
                    "missing_rate": f"{(missing_count/len(df)*100):.1f}%"
                })
            
            return {
                'success': True,
                'data_shape': {'rows': len(df), 'columns': len(df.columns)},
                'numeric_columns': numeric_cols,
                'categorical_columns': categorical_cols,
                'missing_data': missing_data,
                'duplicate_count': duplicate_count,
                'column_info': column_info
            }
            
        except Exception as e:
            logger.error(f"Error analyzing data: {e}")
            return {
                'success': False,
                'error': '데이터 분석 중 오류가 발생했습니다.'
            }
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """데이터 유효성을 검증합니다."""
        try:
            # 기본 검증
            if df.empty:
                return {
                    'valid': False,
                    'error': '빈 데이터프레임입니다.'
                }
            
            # 컬럼 수 검증
            if len(df.columns) == 0:
                return {
                    'valid': False,
                    'error': '컬럼이 없습니다.'
                }
            
            # 행 수 검증
            if len(df) == 0:
                return {
                    'valid': False,
                    'error': '데이터가 없습니다.'
                }
            
            # 메모리 사용량 확인
            memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
            if memory_usage > 1000:  # 1GB 이상
                return {
                    'valid': False,
                    'error': f'데이터가 너무 큽니다. (메모리 사용량: {memory_usage:.1f}MB)'
                }
            
            return {
                'valid': True,
                'memory_usage_mb': memory_usage
            }
            
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return {
                'valid': False,
                'error': '데이터 검증 중 오류가 발생했습니다.'
            }
    
    def get_data_preview(self, df: pd.DataFrame, rows: int = 10) -> Dict[str, Any]:
        """데이터 미리보기를 반환합니다."""
        try:
            preview_data = df.head(rows).to_dict('records')
            return {
                'success': True,
                'preview_data': preview_data,
                'total_rows': len(df),
                'total_columns': len(df.columns)
            }
        except Exception as e:
            logger.error(f"Error getting data preview: {e}")
            return {
                'success': False,
                'error': '데이터 미리보기 생성 중 오류가 발생했습니다.'
            }
    
    def get_column_statistics(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """특정 컬럼의 통계 정보를 반환합니다."""
        try:
            if column not in df.columns:
                return {
                    'success': False,
                    'error': f'컬럼 "{column}"이 존재하지 않습니다.'
                }
            
            col_data = df[column]
            stats = {}
            
            # 기본 통계
            stats['data_type'] = str(col_data.dtype)
            stats['missing_count'] = col_data.isnull().sum()
            stats['unique_count'] = col_data.nunique()
            
            # 수치형 컬럼인 경우
            if pd.api.types.is_numeric_dtype(col_data):
                stats['min'] = float(col_data.min()) if not col_data.empty else None
                stats['max'] = float(col_data.max()) if not col_data.empty else None
                stats['mean'] = float(col_data.mean()) if not col_data.empty else None
                stats['median'] = float(col_data.median()) if not col_data.empty else None
                stats['std'] = float(col_data.std()) if not col_data.empty else None
            
            # 범주형 컬럼인 경우
            else:
                value_counts = col_data.value_counts().head(10).to_dict()
                stats['top_values'] = value_counts
            
            return {
                'success': True,
                'column': column,
                'statistics': stats
            }
            
        except Exception as e:
            logger.error(f"Error getting column statistics for {column}: {e}")
            return {
                'success': False,
                'error': f'컬럼 "{column}" 통계 계산 중 오류가 발생했습니다.'
            } 