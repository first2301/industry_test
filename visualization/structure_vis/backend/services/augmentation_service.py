import pandas as pd
import numpy as np
import sys
import os
import time
from typing import Dict, List, Any, Optional
import logging

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lib import DataAugmenter

logger = logging.getLogger(__name__)

class AugmentationService:
    def __init__(self):
        self.augmenter = DataAugmenter()
    
    def augment_data(self, df: pd.DataFrame, methods: List[str], **params) -> Dict[str, Any]:
        """데이터 증강을 수행합니다."""
        try:
            start_time = time.time()
            
            # 원본 데이터 정보
            original_shape = {'rows': len(df), 'columns': len(df.columns)}
            
            # 증강 실행
            augmented_df = self.augmenter._combined_augmentation(df, methods=methods, **params)
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            
            # 증강 결과 정보
            augmented_shape = {'rows': len(augmented_df), 'columns': len(augmented_df.columns)}
            augmentation_ratio = len(augmented_df) / len(df) - 1  # 실제 증강 비율
            
            return {
                'success': True,
                'original_shape': original_shape,
                'augmented_shape': augmented_shape,
                'augmentation_ratio': augmentation_ratio,
                'processing_time': processing_time,
                'methods_used': methods,
                'augmented_data': augmented_df
            }
            
        except Exception as e:
            logger.error(f"Error during data augmentation: {e}")
            return {
                'success': False,
                'error': '데이터 증강 중 오류가 발생했습니다.'
            }
    
    def validate_augmentation_params(self, methods: List[str], **params) -> Dict[str, Any]:
        """증강 파라미터를 검증합니다."""
        try:
            valid_methods = ['noise', 'duplicate', 'feature', 'smote', 'general']
            
            # 메서드 검증
            for method in methods:
                if method not in valid_methods:
                    return {
                        'valid': False,
                        'error': f'잘못된 증강 방법입니다: {method}'
                    }
            
            # 파라미터 검증
            if 'noise_level' in params:
                noise_level = params['noise_level']
                if not (0.01 <= noise_level <= 0.2):
                    return {
                        'valid': False,
                        'error': '노이즈 레벨은 0.01에서 0.2 사이여야 합니다.'
                    }
            
            if 'dup_count' in params:
                dup_count = params['dup_count']
                if not (2 <= dup_count <= 10):
                    return {
                        'valid': False,
                        'error': '중복 횟수는 2에서 10 사이여야 합니다.'
                    }
            
            if 'augmentation_ratio' in params:
                aug_ratio = params['augmentation_ratio']
                if not (0.1 <= aug_ratio <= 2.0):
                    return {
                        'valid': False,
                        'error': '증강 비율은 0.1에서 2.0 사이여야 합니다.'
                    }
            
            # SMOTE 관련 검증
            if 'smote' in methods:
                if 'target_col' not in params or not params['target_col']:
                    return {
                        'valid': False,
                        'error': 'SMOTE 사용 시 타겟 컬럼을 지정해야 합니다.'
                    }
            
            return {'valid': True}
            
        except Exception as e:
            logger.error(f"Error validating augmentation parameters: {e}")
            return {
                'valid': False,
                'error': '파라미터 검증 중 오류가 발생했습니다.'
            }
    
    def estimate_processing_time(self, df: pd.DataFrame, methods: List[str], **params) -> Dict[str, Any]:
        """예상 처리 시간을 계산합니다."""
        try:
            # 기본 처리 시간 (초)
            base_time = 0.1
            
            # 데이터 크기에 따른 시간
            data_size_factor = len(df) / 10000  # 1만 행 기준
            
            # 메서드별 시간 가중치
            method_weights = {
                'noise': 1.0,
                'duplicate': 0.5,
                'feature': 1.5,
                'smote': 2.0,
                'general': 1.0
            }
            
            total_weight = sum(method_weights.get(method, 1.0) for method in methods)
            
            # 예상 시간 계산
            estimated_time = base_time * data_size_factor * total_weight
            
            # 최소/최대 시간 제한
            estimated_time = max(1.0, min(estimated_time, 300.0))  # 1초 ~ 5분
            
            return {
                'success': True,
                'estimated_time_seconds': estimated_time,
                'estimated_time_minutes': estimated_time / 60
            }
            
        except Exception as e:
            logger.error(f"Error estimating processing time: {e}")
            return {
                'success': False,
                'error': '처리 시간 예측 중 오류가 발생했습니다.'
            }
    
    def get_augmentation_summary(self, original_df: pd.DataFrame, augmented_df: pd.DataFrame) -> Dict[str, Any]:
        """증강 결과 요약을 생성합니다."""
        try:
            summary = {
                'original_rows': len(original_df),
                'augmented_rows': len(augmented_df),
                'row_increase': len(augmented_df) - len(original_df),
                'row_increase_percentage': ((len(augmented_df) - len(original_df)) / len(original_df)) * 100,
                'columns_unchanged': len(original_df.columns) == len(augmented_df.columns),
                'data_types_preserved': True  # 기본적으로 True로 설정
            }
            
            # 컬럼별 변화 확인
            column_changes = {}
            for col in original_df.columns:
                if col in augmented_df.columns:
                    orig_unique = original_df[col].nunique()
                    aug_unique = augmented_df[col].nunique()
                    column_changes[col] = {
                        'original_unique': orig_unique,
                        'augmented_unique': aug_unique,
                        'unique_increase': aug_unique - orig_unique
                    }
            
            summary['column_changes'] = column_changes
            
            return {
                'success': True,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Error generating augmentation summary: {e}")
            return {
                'success': False,
                'error': '증강 요약 생성 중 오류가 발생했습니다.'
            } 