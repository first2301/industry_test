import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List, Any, Optional
import logging
import json

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lib import DataVisualizer

logger = logging.getLogger(__name__)

class VisualizationService:
    def __init__(self):
        self.visualizer = DataVisualizer()
    
    def create_histogram_comparison(self, original_df: pd.DataFrame, augmented_df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """히스토그램 비교 차트를 생성합니다."""
        try:
            if column not in original_df.columns:
                return {
                    'success': False,
                    'error': f'컬럼 "{column}"이 원본 데이터에 존재하지 않습니다.'
                }
            
            if column not in augmented_df.columns:
                return {
                    'success': False,
                    'error': f'컬럼 "{column}"이 증강된 데이터에 존재하지 않습니다.'
                }
            
            # Plotly 차트 생성
            fig = self.visualizer.create_overlapping_histogram(original_df, augmented_df, column)
            
            # 차트 데이터를 JSON으로 변환
            chart_data = json.loads(fig.to_json())
            
            return {
                'success': True,
                'chart_type': 'histogram_comparison',
                'column': column,
                'chart_data': chart_data
            }
            
        except Exception as e:
            logger.error(f"Error creating histogram comparison for {column}: {e}")
            return {
                'success': False,
                'error': f'히스토그램 비교 차트 생성 중 오류가 발생했습니다.'
            }
    
    def create_boxplot_comparison(self, original_df: pd.DataFrame, augmented_df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """박스플롯 비교 차트를 생성합니다."""
        try:
            if column not in original_df.columns:
                return {
                    'success': False,
                    'error': f'컬럼 "{column}"이 원본 데이터에 존재하지 않습니다.'
                }
            
            if column not in augmented_df.columns:
                return {
                    'success': False,
                    'error': f'컬럼 "{column}"이 증강된 데이터에 존재하지 않습니다.'
                }
            
            # Plotly 차트 생성
            fig = self.visualizer.create_overlapping_boxplot(original_df, augmented_df, column)
            
            # 차트 데이터를 JSON으로 변환
            chart_data = json.loads(fig.to_json())
            
            return {
                'success': True,
                'chart_type': 'boxplot_comparison',
                'column': column,
                'chart_data': chart_data
            }
            
        except Exception as e:
            logger.error(f"Error creating boxplot comparison for {column}: {e}")
            return {
                'success': False,
                'error': f'박스플롯 비교 차트 생성 중 오류가 발생했습니다.'
            }
    
    def create_scatter_comparison(self, original_df: pd.DataFrame, augmented_df: pd.DataFrame, x_col: str, y_col: str) -> Dict[str, Any]:
        """산점도 비교 차트를 생성합니다."""
        try:
            # 컬럼 존재 확인
            for col in [x_col, y_col]:
                if col not in original_df.columns:
                    return {
                        'success': False,
                        'error': f'컬럼 "{col}"이 원본 데이터에 존재하지 않습니다.'
                    }
                if col not in augmented_df.columns:
                    return {
                        'success': False,
                        'error': f'컬럼 "{col}"이 증강된 데이터에 존재하지 않습니다.'
                    }
            
            # Plotly 차트 생성
            fig = self.visualizer.create_overlapping_scatter(original_df, augmented_df, x_col, y_col)
            
            # 차트 데이터를 JSON으로 변환
            chart_data = json.loads(fig.to_json())
            
            return {
                'success': True,
                'chart_type': 'scatter_comparison',
                'x_column': x_col,
                'y_column': y_col,
                'chart_data': chart_data
            }
            
        except Exception as e:
            logger.error(f"Error creating scatter comparison: {e}")
            return {
                'success': False,
                'error': f'산점도 비교 차트 생성 중 오류가 발생했습니다.'
            }
    
    def create_categorical_comparison(self, original_df: pd.DataFrame, augmented_df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """범주형 데이터 비교 차트를 생성합니다."""
        try:
            if column not in original_df.columns:
                return {
                    'success': False,
                    'error': f'컬럼 "{column}"이 원본 데이터에 존재하지 않습니다.'
                }
            
            if column not in augmented_df.columns:
                return {
                    'success': False,
                    'error': f'컬럼 "{column}"이 증강된 데이터에 존재하지 않습니다.'
                }
            
            # 원본 데이터 카운트
            orig_counts = original_df[column].value_counts().sort_index()
            aug_counts = augmented_df[column].value_counts().sort_index()
            
            # 모든 카테고리 통합
            all_categories = sorted(set(orig_counts.index) | set(aug_counts.index))
            
            # 데이터 준비
            comparison_data = []
            for cat in all_categories:
                orig_count = orig_counts.get(cat, 0)
                aug_count = aug_counts.get(cat, 0)
                comparison_data.append({
                    'category': cat,
                    'original': orig_count,
                    'augmented': aug_count,
                    'increase': aug_count - orig_count,
                    'increase_percentage': ((aug_count - orig_count) / orig_count * 100) if orig_count > 0 else float('inf')
                })
            
            return {
                'success': True,
                'chart_type': 'categorical_comparison',
                'column': column,
                'comparison_data': comparison_data,
                'categories': all_categories
            }
            
        except Exception as e:
            logger.error(f"Error creating categorical comparison for {column}: {e}")
            return {
                'success': False,
                'error': f'범주형 비교 차트 생성 중 오류가 발생했습니다.'
            }
    
    def get_comparison_summary(self, original_df: pd.DataFrame, augmented_df: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Any]:
        """비교 요약 통계를 생성합니다."""
        try:
            summary_stats = {}
            
            for col in numeric_columns:
                if col in original_df.columns and col in augmented_df.columns:
                    orig_stats = original_df[col].describe()
                    aug_stats = augmented_df[col].describe()
                    
                    summary_stats[col] = {
                        'original': {
                            'count': int(orig_stats['count']),
                            'mean': float(orig_stats['mean']),
                            'std': float(orig_stats['std']),
                            'min': float(orig_stats['min']),
                            'max': float(orig_stats['max']),
                            'median': float(original_df[col].median())
                        },
                        'augmented': {
                            'count': int(aug_stats['count']),
                            'mean': float(aug_stats['mean']),
                            'std': float(aug_stats['std']),
                            'min': float(aug_stats['min']),
                            'max': float(aug_stats['max']),
                            'median': float(augmented_df[col].median())
                        },
                        'changes': {
                            'count_change': int(aug_stats['count'] - orig_stats['count']),
                            'mean_change': float(aug_stats['mean'] - orig_stats['mean']),
                            'std_change': float(aug_stats['std'] - orig_stats['std']),
                            'range_change': float((aug_stats['max'] - aug_stats['min']) - (orig_stats['max'] - orig_stats['min']))
                        }
                    }
            
            return {
                'success': True,
                'summary_stats': summary_stats,
                'columns_analyzed': list(summary_stats.keys())
            }
            
        except Exception as e:
            logger.error(f"Error generating comparison summary: {e}")
            return {
                'success': False,
                'error': '비교 요약 생성 중 오류가 발생했습니다.'
            }
    
    def create_augmentation_report(self, original_df: pd.DataFrame, augmented_df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """증강 결과 리포트를 생성합니다."""
        try:
            # 기본 정보
            report = {
                'original_shape': {'rows': len(original_df), 'columns': len(original_df.columns)},
                'augmented_shape': {'rows': len(augmented_df), 'columns': len(augmented_df.columns)},
                'augmentation_ratio': len(augmented_df) / len(original_df) - 1,
                'parameters_used': params,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            # 컬럼 타입 분석
            numeric_cols = self.visualizer.get_numeric_columns(original_df)
            categorical_cols = self.visualizer.get_categorical_columns(original_df)
            
            report['column_analysis'] = {
                'numeric_columns': numeric_cols,
                'categorical_columns': categorical_cols,
                'total_columns': len(original_df.columns)
            }
            
            # 품질 지표
            report['quality_metrics'] = {
                'original_missing_rate': (original_df.isnull().sum().sum() / (len(original_df) * len(original_df.columns))) * 100,
                'augmented_missing_rate': (augmented_df.isnull().sum().sum() / (len(augmented_df) * len(augmented_df.columns))) * 100,
                'original_duplicates': original_df.duplicated().sum(),
                'augmented_duplicates': augmented_df.duplicated().sum()
            }
            
            return {
                'success': True,
                'report': report
            }
            
        except Exception as e:
            logger.error(f"Error creating augmentation report: {e}")
            return {
                'success': False,
                'error': '증강 리포트 생성 중 오류가 발생했습니다.'
            } 