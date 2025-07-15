import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
import re

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataType(Enum):
    """데이터 타입 분류"""
    NUMERIC_CONTINUOUS = "numeric_continuous"
    NUMERIC_DISCRETE = "numeric_discrete"
    CATEGORICAL_NOMINAL = "categorical_nominal"
    CATEGORICAL_ORDINAL = "categorical_ordinal"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"
    ID = "id"

class DataContext(Enum):
    """데이터 컨텍스트 분류"""
    GENERAL = "general"
    TIMESERIES = "timeseries"
    IMBALANCED = "imbalanced"
    HIGH_DIMENSIONAL = "high_dimensional"
    SPARSE = "sparse"
    MULTIMODAL = "multimodal"

@dataclass
class DataProfiler:
    """데이터 특성 프로파일링"""
    column_name: str
    data_type: DataType
    missing_ratio: float
    unique_count: int
    unique_ratio: float
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    cv: Optional[float] = None  # 변동계수
    outlier_ratio: Optional[float] = None
    is_constant: bool = False
    is_id_like: bool = False
    text_length_stats: Optional[Dict] = None
    datetime_range: Optional[Tuple] = None

class EnhancedPreprocessingRecommender:
    """
    고도화된 전처리 추천 시스템
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ▸ 다양한 데이터 타입 지원
    ▸ 컨텍스트 기반 추천
    ▸ 적응형 임계값
    ▸ 우선순위 기반 추천
    """
    
    def __init__(self):
        self.profiles: Dict[str, DataProfiler] = {}
        self.context: DataContext = DataContext.GENERAL
        self.thresholds = self._get_adaptive_thresholds()
    
    def _get_adaptive_thresholds(self) -> Dict[str, float]:
        """적응형 임계값 설정"""
        return {
            'missing_low': 0.05,
            'missing_medium': 0.15,
            'missing_high': 0.40,
            'skewness_moderate': 1.0,
            'skewness_high': 2.0,
            'kurtosis_high': 7.0,
            'cv_low': 0.1,
            'cv_high': 1.0,
            'correlation_high': 0.8,
            'correlation_very_high': 0.95,
            'unique_ratio_low': 0.01,
            'unique_ratio_high': 0.8,
            'outlier_ratio_threshold': 0.05
        }
    
    def profile_data(self, df: pd.DataFrame) -> None:
        """데이터 프로파일링"""
        self.profiles.clear()
        
        # 전체 데이터 컨텍스트 분석
        self.context = self._analyze_data_context(df)
        
        # 컨텍스트에 따른 임계값 조정
        self._adjust_thresholds_by_context()
        
        for col in df.columns:
            profile = self._profile_column(df, col)
            self.profiles[col] = profile
    
    def _analyze_data_context(self, df: pd.DataFrame) -> DataContext:
        """데이터 컨텍스트 분석"""
        n_rows, n_cols = df.shape
        
        # 시계열 데이터 검사
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0 or any('date' in col.lower() or 'time' in col.lower() for col in df.columns):
            return DataContext.TIMESERIES
        
        # 고차원 데이터 검사
        if n_cols > 100:
            return DataContext.HIGH_DIMENSIONAL
        
        # 희소 데이터 검사
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            zero_ratio = (df[numeric_cols] == 0).sum().sum() / (n_rows * len(numeric_cols))
            if zero_ratio > 0.8:
                return DataContext.SPARSE
        
        # 불균형 데이터 검사 (타겟 변수가 있는 경우)
        # 여기서는 간단히 범주형 변수의 불균형 정도로 판단
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df[col].nunique() < 10:
                value_counts = df[col].value_counts()
                if len(value_counts) > 1:
                    imbalance_ratio = value_counts.iloc[0] / value_counts.iloc[1]
                    if imbalance_ratio > 10:
                        return DataContext.IMBALANCED
        
        return DataContext.GENERAL
    
    def _adjust_thresholds_by_context(self) -> None:
        """컨텍스트에 따른 임계값 조정"""
        if self.context == DataContext.HIGH_DIMENSIONAL:
            self.thresholds['missing_high'] = 0.20  # 고차원에서는 더 엄격
            self.thresholds['correlation_high'] = 0.7
        elif self.context == DataContext.IMBALANCED:
            self.thresholds['missing_medium'] = 0.10  # 불균형 데이터에서는 더 보수적
        elif self.context == DataContext.SPARSE:
            self.thresholds['missing_high'] = 0.60  # 희소 데이터에서는 더 관대
    
    def _profile_column(self, df: pd.DataFrame, col: str) -> DataProfiler:
        """개별 컬럼 프로파일링"""
        series = df[col]
        missing_ratio = series.isnull().mean()
        unique_count = series.nunique()
        unique_ratio = unique_count / len(series)
        
        # 데이터 타입 분류
        data_type = self._classify_data_type(series)
        
        # 기본 프로파일 생성
        profile = DataProfiler(
            column_name=col,
            data_type=data_type,
            missing_ratio=missing_ratio,
            unique_count=unique_count,
            unique_ratio=unique_ratio,
            is_constant=unique_count <= 1,
            is_id_like=self._is_id_like(series)
        )
        
        # 수치형 데이터 추가 분석
        if data_type in [DataType.NUMERIC_CONTINUOUS, DataType.NUMERIC_DISCRETE]:
            numeric_data = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_data) > 0:
                profile.mean = numeric_data.mean()
                profile.std = numeric_data.std()
                profile.cv = profile.std / profile.mean if profile.mean != 0 else np.inf
                profile.skewness = numeric_data.skew()
                profile.kurtosis = numeric_data.kurtosis()
                profile.outlier_ratio = self._calculate_outlier_ratio(numeric_data)
        
        # 텍스트 데이터 추가 분석
        elif data_type == DataType.TEXT:
            text_data = series.dropna().astype(str)
            if len(text_data) > 0:
                lengths = text_data.str.len()
                profile.text_length_stats = {
                    'mean_length': lengths.mean(),
                    'max_length': lengths.max(),
                    'min_length': lengths.min(),
                    'std_length': lengths.std()
                }
        
        # 날짜/시간 데이터 추가 분석
        elif data_type == DataType.DATETIME:
            datetime_data = pd.to_datetime(series, errors='coerce').dropna()
            if len(datetime_data) > 0:
                profile.datetime_range = (datetime_data.min(), datetime_data.max())
        
        return profile
    
    def _classify_data_type(self, series: pd.Series) -> DataType:
        """데이터 타입 분류"""
        col_name = series.name.lower()
        
        # ID 컬럼 검사
        if self._is_id_like(series):
            return DataType.ID
        
        # 불린 타입 검사
        if series.dtype == bool or set(series.dropna().unique()) <= {0, 1, True, False}:
            return DataType.BOOLEAN
        
        # 날짜/시간 타입 검사
        if series.dtype.name.startswith('datetime') or 'date' in col_name or 'time' in col_name:
            return DataType.DATETIME
        
        # 수치형 타입 검사
        if pd.api.types.is_numeric_dtype(series):
            # 이산형 vs 연속형 판단
            unique_count = series.nunique()
            if unique_count <= 20 and series.dtype in ['int64', 'int32']:
                return DataType.NUMERIC_DISCRETE
            else:
                return DataType.NUMERIC_CONTINUOUS
        
        # 텍스트 길이 기반 분류
        if series.dtype == 'object':
            text_data = series.dropna().astype(str)
            if len(text_data) > 0:
                avg_length = text_data.str.len().mean()
                if avg_length > 50:  # 평균 길이 50자 이상이면 텍스트
                    return DataType.TEXT
        
        # 범주형 타입 검사
        if series.dtype.name == 'category' or series.dtype == 'object':
            unique_count = series.nunique()
            if unique_count <= 50:
                # 순서형 vs 명목형 판단 (간단한 휴리스틱)
                unique_vals = series.dropna().astype(str).unique()
                if any(val in ['low', 'medium', 'high', 'small', 'large'] for val in unique_vals):
                    return DataType.CATEGORICAL_ORDINAL
                return DataType.CATEGORICAL_NOMINAL
        
        return DataType.CATEGORICAL_NOMINAL
    
    def _is_id_like(self, series: pd.Series) -> bool:
        """ID 컬럼 여부 판단"""
        col_name = series.name.lower()
        
        # 컬럼명 기반 판단
        if any(keyword in col_name for keyword in ['id', 'key', 'index', 'idx']):
            return True
        
        # 고유값 비율 기반 판단
        if series.nunique() / len(series) > 0.95:
            return True
        
        return False
    
    def _calculate_outlier_ratio(self, data: pd.Series) -> float:
        """이상치 비율 계산 (IQR 방법)"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        return len(outliers) / len(data)
    
    def recommend(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """전처리 추천"""
        self.profile_data(df)
        recommendations = {}
        
        for col, profile in self.profiles.items():
            col_recs = self._generate_column_recommendations(df, profile)
            recommendations[col] = {
                'data_type': profile.data_type.value,
                'recommendations': col_recs,
                'priority': self._calculate_priority(profile),
                'context': self.context.value
            }
        
        # 전역 추천사항 추가
        global_recs = self._generate_global_recommendations(df)
        if global_recs:
            recommendations['_global'] = {
                'data_type': 'global',
                'recommendations': global_recs,
                'priority': 'high',
                'context': self.context.value
            }
        
        return recommendations
    
    def _generate_column_recommendations(self, df: pd.DataFrame, profile: DataProfiler) -> List[str]:
        """컬럼별 추천 생성"""
        recs = []
        
        # ID 컬럼 처리
        if profile.data_type == DataType.ID:
            recs.append('consider_dropping_id_column')
            return recs
        
        # 상수 컬럼 처리
        if profile.is_constant:
            recs.append('drop_constant_column')
            return recs
        
        # 결측치 처리
        if profile.missing_ratio > self.thresholds['missing_high']:
            recs.append('drop_column_excessive_missing')
        elif profile.missing_ratio > self.thresholds['missing_medium']:
            recs.append('advanced_imputation_knn_iterative')
        elif profile.missing_ratio > self.thresholds['missing_low']:
            if profile.data_type in [DataType.NUMERIC_CONTINUOUS, DataType.NUMERIC_DISCRETE]:
                recs.append('impute_median_mode')
            else:
                recs.append('impute_mode_frequent')
        
        # 수치형 데이터 처리
        if profile.data_type in [DataType.NUMERIC_CONTINUOUS, DataType.NUMERIC_DISCRETE]:
            recs.extend(self._get_numeric_recommendations(profile))
        
        # 범주형 데이터 처리
        elif profile.data_type in [DataType.CATEGORICAL_NOMINAL, DataType.CATEGORICAL_ORDINAL]:
            recs.extend(self._get_categorical_recommendations(profile))
        
        # 텍스트 데이터 처리
        elif profile.data_type == DataType.TEXT:
            recs.extend(self._get_text_recommendations(profile))
        
        # 날짜/시간 데이터 처리
        elif profile.data_type == DataType.DATETIME:
            recs.extend(self._get_datetime_recommendations(profile))
        
        return recs
    
    def _get_numeric_recommendations(self, profile: DataProfiler) -> List[str]:
        """수치형 데이터 추천"""
        recs = []
        
        # 분산 검사
        if profile.cv and profile.cv < self.thresholds['cv_low']:
            recs.append('consider_low_variance_removal')
        
        # 이상치 처리
        if profile.outlier_ratio and profile.outlier_ratio > self.thresholds['outlier_ratio_threshold']:
            recs.append('outlier_detection_treatment')
        
        # 분포 특성 기반 추천
        if profile.skewness and profile.kurtosis:
            if abs(profile.skewness) > self.thresholds['skewness_high'] or profile.kurtosis > self.thresholds['kurtosis_high']:
                recs.append('distribution_transformation_log_boxcox')
            elif abs(profile.skewness) > self.thresholds['skewness_moderate']:
                recs.append('robust_scaling_iqr')
        
        # 스케일링 추천
        if profile.cv and profile.cv > self.thresholds['cv_high']:
            recs.append('standardization_zscore')
        else:
            recs.append('normalization_minmax')
        
        # 컨텍스트 기반 추천
        if self.context == DataContext.HIGH_DIMENSIONAL:
            recs.append('dimensionality_reduction_pca')
        elif self.context == DataContext.SPARSE:
            recs.append('sparse_feature_selection')
        
        return recs
    
    def _get_categorical_recommendations(self, profile: DataProfiler) -> List[str]:
        """범주형 데이터 추천"""
        recs = []
        
        # 인코딩 추천
        if profile.unique_count <= 10:
            recs.append('one_hot_encoding')
        elif profile.unique_count <= 50:
            if profile.data_type == DataType.CATEGORICAL_ORDINAL:
                recs.append('ordinal_encoding')
            else:
                recs.append('label_encoding')
        elif profile.unique_count <= 1000:
            recs.append('target_encoding')
        else:
            recs.append('hashing_encoding')
        
        # 고카디널리티 처리
        if profile.unique_count > 100:
            recs.append('high_cardinality_feature_selection')
        
        # 희소 범주 처리
        if profile.unique_ratio < self.thresholds['unique_ratio_low']:
            recs.append('rare_category_grouping')
        
        return recs
    
    def _get_text_recommendations(self, profile: DataProfiler) -> List[str]:
        """텍스트 데이터 추천"""
        recs = []
        
        if profile.text_length_stats:
            avg_length = profile.text_length_stats['mean_length']
            
            if avg_length < 20:
                recs.extend(['text_preprocessing_basic', 'bag_of_words'])
            elif avg_length < 100:
                recs.extend(['text_preprocessing_advanced', 'tfidf_vectorization'])
            else:
                recs.extend(['text_preprocessing_advanced', 'text_embeddings', 'topic_modeling'])
        
        recs.append('text_feature_extraction')
        return recs
    
    def _get_datetime_recommendations(self, profile: DataProfiler) -> List[str]:
        """날짜/시간 데이터 추천"""
        recs = []
        
        recs.extend([
            'datetime_feature_extraction',
            'time_based_features',
            'cyclical_encoding'
        ])
        
        if self.context == DataContext.TIMESERIES:
            recs.extend([
                'time_series_decomposition',
                'lag_features',
                'rolling_statistics'
            ])
        
        return recs
    
    def _calculate_priority(self, profile: DataProfiler) -> str:
        """우선순위 계산"""
        if profile.is_constant or profile.missing_ratio > self.thresholds['missing_high']:
            return 'high'
        elif profile.missing_ratio > self.thresholds['missing_medium']:
            return 'medium'
        else:
            return 'low'
    
    def _generate_global_recommendations(self, df: pd.DataFrame) -> List[str]:
        """전역 추천사항"""
        recs = []
        
        # 다중공선성 검사
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > self.thresholds['correlation_very_high']:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            if high_corr_pairs:
                recs.append('multicollinearity_detection_vif')
        
        # 컨텍스트 기반 전역 추천
        if self.context == DataContext.IMBALANCED:
            recs.extend(['class_balancing_smote', 'stratified_sampling'])
        elif self.context == DataContext.HIGH_DIMENSIONAL:
            recs.extend(['feature_selection_univariate', 'regularization_techniques'])
        elif self.context == DataContext.SPARSE:
            recs.extend(['sparse_matrix_optimization', 'feature_hashing'])
        
        return recs

class EnhancedVisualizationRecommender:
    """
    고도화된 시각화 추천 시스템
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ▸ 분석 목적 기반 추천
    ▸ 다변량 시각화 지원
    ▸ 상호작용 시각화
    ▸ 데이터 품질 시각화
    """
    
    def __init__(self, analysis_purpose: str = 'exploratory'):
        self.analysis_purpose = analysis_purpose  # 'exploratory', 'confirmatory', 'presentation'
        self.data_profiles: Dict[str, DataProfiler] = {}
        self.context: DataContext = DataContext.GENERAL
    
    def recommend(self, df: pd.DataFrame, data_profiles: Dict[str, DataProfiler] = None, 
                 context: DataContext = DataContext.GENERAL) -> Dict[str, Dict[str, Any]]:
        """시각화 추천"""
        
        if data_profiles:
            self.data_profiles = data_profiles
        else:
            # 간단한 프로파일링
            preprocessor = EnhancedPreprocessingRecommender()
            preprocessor.profile_data(df)
            self.data_profiles = preprocessor.profiles
        
        self.context = context
        
        recommendations = {}
        
        # 단변량 시각화
        for col, profile in self.data_profiles.items():
            if col.startswith('_'):
                continue
            
            col_viz = self._get_univariate_visualizations(profile)
            recommendations[col] = {
                'type': 'univariate',
                'visualizations': col_viz,
                'priority': self._calculate_viz_priority(profile)
            }
        
        # 이변량 시각화
        bivariate_viz = self._get_bivariate_visualizations(df)
        if bivariate_viz:
            recommendations['_bivariate'] = {
                'type': 'bivariate',
                'visualizations': bivariate_viz,
                'priority': 'medium'
            }
        
        # 다변량 시각화
        multivariate_viz = self._get_multivariate_visualizations(df)
        if multivariate_viz:
            recommendations['_multivariate'] = {
                'type': 'multivariate',
                'visualizations': multivariate_viz,
                'priority': 'high'
            }
        
        # 데이터 품질 시각화
        quality_viz = self._get_data_quality_visualizations(df)
        if quality_viz:
            recommendations['_data_quality'] = {
                'type': 'data_quality',
                'visualizations': quality_viz,
                'priority': 'high'
            }
        
        return recommendations
    
    def _get_univariate_visualizations(self, profile: DataProfiler) -> List[str]:
        """단변량 시각화 추천"""
        viz_list = []
        
        if profile.data_type in [DataType.NUMERIC_CONTINUOUS, DataType.NUMERIC_DISCRETE]:
            viz_list.extend(['histogram', 'box_plot', 'violin_plot'])
            
            if profile.skewness and abs(profile.skewness) > 1:
                viz_list.append('qq_plot')
            
            if profile.outlier_ratio and profile.outlier_ratio > 0.05:
                viz_list.append('outlier_detection_plot')
        
        elif profile.data_type in [DataType.CATEGORICAL_NOMINAL, DataType.CATEGORICAL_ORDINAL]:
            viz_list.extend(['bar_chart', 'horizontal_bar_chart'])
            
            if profile.unique_count <= 10:
                viz_list.append('pie_chart')
            
            if profile.unique_count > 20:
                viz_list.append('pareto_chart')
        
        elif profile.data_type == DataType.TEXT:
            viz_list.extend(['word_cloud', 'text_length_distribution', 'n_gram_frequency'])
        
        elif profile.data_type == DataType.DATETIME:
            viz_list.extend(['time_series_plot', 'temporal_distribution'])
            
            if self.context == DataContext.TIMESERIES:
                viz_list.extend(['seasonal_decomposition', 'autocorrelation_plot'])
        
        return viz_list
    
    def _get_bivariate_visualizations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """이변량 시각화 추천"""
        viz_list = []
        
        numeric_cols = [col for col, profile in self.data_profiles.items() 
                       if profile.data_type in [DataType.NUMERIC_CONTINUOUS, DataType.NUMERIC_DISCRETE]]
        
        categorical_cols = [col for col, profile in self.data_profiles.items() 
                          if profile.data_type in [DataType.CATEGORICAL_NOMINAL, DataType.CATEGORICAL_ORDINAL]]
        
        # 수치형 vs 수치형
        if len(numeric_cols) >= 2:
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    
                    # 상관관계 계산
                    corr_val = df[col1].corr(df[col2])
                    
                    viz_recommendations = ['scatter_plot']
                    
                    if abs(corr_val) > 0.7:
                        viz_recommendations.extend(['regression_line', 'correlation_heatmap'])
                    
                    if abs(corr_val) > 0.3:
                        viz_recommendations.append('joint_plot')
                    
                    viz_list.append({
                        'variables': [col1, col2],
                        'relationship': 'numeric_numeric',
                        'correlation': corr_val,
                        'visualizations': viz_recommendations
                    })
        
        # 범주형 vs 수치형
        for cat_col in categorical_cols:
            for num_col in numeric_cols:
                viz_recommendations = ['grouped_box_plot', 'violin_plot_grouped']
                
                if self.data_profiles[cat_col].unique_count <= 10:
                    viz_recommendations.append('strip_plot')
                
                viz_list.append({
                    'variables': [cat_col, num_col],
                    'relationship': 'categorical_numeric',
                    'visualizations': viz_recommendations
                })
        
        # 범주형 vs 범주형
        if len(categorical_cols) >= 2:
            for i in range(len(categorical_cols)):
                for j in range(i+1, len(categorical_cols)):
                    col1, col2 = categorical_cols[i], categorical_cols[j]
                    
                    viz_recommendations = ['contingency_table_heatmap', 'grouped_bar_chart']
                    
                    if (self.data_profiles[col1].unique_count <= 10 and 
                        self.data_profiles[col2].unique_count <= 10):
                        viz_recommendations.append('mosaic_plot')
                    
                    viz_list.append({
                        'variables': [col1, col2],
                        'relationship': 'categorical_categorical',
                        'visualizations': viz_recommendations
                    })
        
        return viz_list[:10]  # 최대 10개 추천
    
    def _get_multivariate_visualizations(self, df: pd.DataFrame) -> List[str]:
        """다변량 시각화 추천"""
        viz_list = []
        
        numeric_cols = [col for col, profile in self.data_profiles.items() 
                       if profile.data_type in [DataType.NUMERIC_CONTINUOUS, DataType.NUMERIC_DISCRETE]]
        
        if len(numeric_cols) >= 3:
            viz_list.extend(['correlation_matrix', 'pair_plot', 'parallel_coordinates'])
            
            if len(numeric_cols) >= 4:
                viz_list.append('3d_scatter_plot')
            
            if len(numeric_cols) >= 5:
                viz_list.extend(['pca_biplot', 'radar_chart'])
        
        if self.context == DataContext.HIGH_DIMENSIONAL:
            viz_list.extend(['dimensionality_reduction_plot', 'feature_importance_plot'])
        
        return viz_list
    
    def _get_data_quality_visualizations(self, df: pd.DataFrame) -> List[str]:
        """데이터 품질 시각화 추천"""
        viz_list = []
        
        # 결측값 시각화
        total_missing = df.isnull().sum().sum()
        if total_missing > 0:
            viz_list.extend(['missing_value_matrix', 'missing_value_heatmap'])
        
        # 중복값 시각화
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            viz_list.append('duplicate_detection_plot')
        
        # 이상치 시각화
        numeric_cols = [col for col, profile in self.data_profiles.items() 
                       if profile.data_type in [DataType.NUMERIC_CONTINUOUS, DataType.NUMERIC_DISCRETE]]
        
        if numeric_cols:
            viz_list.extend(['outlier_detection_plot', 'data_distribution_overview'])
        
        # 데이터 타입 검증 시각화
        viz_list.append('data_type_validation_plot')
        
        return viz_list
    
    def _calculate_viz_priority(self, profile: DataProfiler) -> str:
        """시각화 우선순위 계산"""
        if profile.missing_ratio > 0.3 or profile.is_constant:
            return 'high'
        elif profile.data_type in [DataType.NUMERIC_CONTINUOUS, DataType.NUMERIC_DISCRETE]:
            return 'medium'
        else:
            return 'low'

class EnhancedRecommendationEngine:
    """
    고도화된 추천 엔진
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ▸ 통합 추천 시스템
    ▸ 우선순위 기반 정렬
    ▸ 실행 가능한 코드 생성
    """
    
    def __init__(self, analysis_purpose: str = 'exploratory'):
        self.preprocessor = EnhancedPreprocessingRecommender()
        self.visualizer = EnhancedVisualizationRecommender(analysis_purpose)
        self.analysis_purpose = analysis_purpose
    
    def run(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """통합 추천 실행"""
        
        # 데이터 검증
        if df.empty:
            raise ValueError("입력 데이터프레임이 비어있습니다.")
        
        # 전처리 추천
        logger.info("전처리 추천 시작...")
        preprocessing_recs = self.preprocessor.recommend(df)
        
        # 시각화 추천
        logger.info("시각화 추천 시작...")
        visualization_recs = self.visualizer.recommend(
            df, 
            self.preprocessor.profiles, 
            self.preprocessor.context
        )
        
        # 결과 통합
        result = {
            'data_info': {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'context': self.preprocessor.context.value
            },
            'preprocessing': preprocessing_recs,
            'visualization': visualization_recs,
            'summary': self._generate_summary(preprocessing_recs, visualization_recs),
            'code_templates': self._generate_code_templates(preprocessing_recs, visualization_recs)
        }
        
        return result
    
    def _generate_summary(self, preprocessing_recs: Dict, visualization_recs: Dict) -> Dict[str, Any]:
        """추천 요약 생성"""
        
        # 우선순위별 전처리 추천 집계
        priority_counts = {'high': 0, 'medium': 0, 'low': 0}
        for col, rec in preprocessing_recs.items():
            if col.startswith('_'):
                continue
            priority = rec.get('priority', 'low')
            priority_counts[priority] += 1
        
        # 주요 추천사항 추출
        high_priority_recs = []
        for col, rec in preprocessing_recs.items():
            if rec.get('priority') == 'high':
                high_priority_recs.append(f"{col}: {rec['recommendations'][:2]}")
        
        return {
            'total_columns': len([col for col in preprocessing_recs.keys() if not col.startswith('_')]),
            'priority_distribution': priority_counts,
            'high_priority_recommendations': high_priority_recs[:5],
            'recommended_visualizations': len(visualization_recs),
            'analysis_context': self.preprocessor.context.value
        }
    
    def _generate_code_templates(self, preprocessing_recs: Dict, visualization_recs: Dict) -> Dict[str, str]:
        """실행 가능한 코드 템플릿 생성"""
        
        templates = {}
        
        # 전처리 코드 템플릿
        preprocessing_code = self._generate_preprocessing_code(preprocessing_recs)
        templates['preprocessing'] = preprocessing_code
        
        # 시각화 코드 템플릿
        visualization_code = self._generate_visualization_code(visualization_recs)
        templates['visualization'] = visualization_code
        
        return templates
    
    def _generate_preprocessing_code(self, recommendations: Dict) -> str:
        """전처리 코드 생성"""
        code_lines = [
            "import pandas as pd",
            "import numpy as np",
            "from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder",
            "from sklearn.impute import SimpleImputer, KNNImputer",
            "",
            "# 데이터 전처리 파이프라인",
            "def preprocess_data(df):",
            "    df_processed = df.copy()",
            ""
        ]
        
        for col, rec in recommendations.items():
            if col.startswith('_'):
                continue
            
            for recommendation in rec['recommendations'][:3]:  # 상위 3개 추천만
                if 'drop' in recommendation:
                    code_lines.append(f"    # {col}: {recommendation}")
                    code_lines.append(f"    df_processed = df_processed.drop('{col}', axis=1)")
                elif 'impute' in recommendation:
                    code_lines.append(f"    # {col}: {recommendation}")
                    code_lines.append(f"    imputer = SimpleImputer(strategy='median')")
                    code_lines.append(f"    df_processed['{col}'] = imputer.fit_transform(df_processed[['{col}']])")
                elif 'standardization' in recommendation:
                    code_lines.append(f"    # {col}: {recommendation}")
                    code_lines.append(f"    scaler = StandardScaler()")
                    code_lines.append(f"    df_processed['{col}'] = scaler.fit_transform(df_processed[['{col}']])")
        
        code_lines.extend([
            "",
            "    return df_processed",
            "",
            "# 사용 예시:",
            "# processed_df = preprocess_data(df)"
        ])
        
        return "\n".join(code_lines)
    
    def _generate_visualization_code(self, recommendations: Dict) -> str:
        """시각화 코드 생성"""
        code_lines = [
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "import plotly.express as px",
            "",
            "# 데이터 시각화 함수들",
            "def create_visualizations(df):",
            "    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))",
            "    axes = axes.flatten()",
            ""
        ]
        
        plot_idx = 0
        for col, rec in recommendations.items():
            if col.startswith('_') or plot_idx >= 4:
                continue
            
            viz_type = rec['visualizations'][0] if rec['visualizations'] else 'histogram'
            
            if viz_type == 'histogram':
                code_lines.append(f"    # {col} 히스토그램")
                code_lines.append(f"    axes[{plot_idx}].hist(df['{col}'].dropna(), bins=30)")
                code_lines.append(f"    axes[{plot_idx}].set_title('{col} Distribution')")
            elif viz_type == 'bar_chart':
                code_lines.append(f"    # {col} 막대 차트")
                code_lines.append(f"    df['{col}'].value_counts().plot(kind='bar', ax=axes[{plot_idx}])")
                code_lines.append(f"    axes[{plot_idx}].set_title('{col} Value Counts')")
            
            plot_idx += 1
        
        code_lines.extend([
            "",
            "    plt.tight_layout()",
            "    plt.show()",
            "",
            "# 사용 예시:",
            "# create_visualizations(df)"
        ])
        
        return "\n".join(code_lines)

# 사용 예시
if __name__ == "__main__":
    # 샘플 데이터 생성
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'id': range(1000),
        'age': np.random.normal(35, 10, 1000),
        'income': np.random.exponential(50000, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'text_field': ['sample text ' * np.random.randint(1, 10) for _ in range(1000)],
        'binary_field': np.random.choice([0, 1], 1000)
    })
    
    # 결측값 추가
    sample_data.loc[np.random.choice(1000, 100), 'age'] = np.nan
    sample_data.loc[np.random.choice(1000, 50), 'income'] = np.nan
    
    # 추천 엔진 실행
    engine = EnhancedRecommendationEngine(analysis_purpose='exploratory')
    recommendations = engine.run(sample_data)
    
    # 결과 출력
    print("=" * 60)
    print("📊 고도화된 데이터 분석 추천 시스템")
    print("=" * 60)
    print(f"데이터 형태: {recommendations['data_info']['shape']}")
    print(f"분석 컨텍스트: {recommendations['data_info']['context']}")
    print("\n전처리 추천:")
    for col, rec in recommendations['preprocessing'].items():
        if not col.startswith('_'):
            print(f"  {col}: {rec['recommendations'][:2]}")
    
    print(f"\n코드 템플릿이 생성되었습니다.") 