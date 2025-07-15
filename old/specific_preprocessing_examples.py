import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from scipy import stats
from scipy.stats import boxcox, yeojohnson
import warnings
warnings.filterwarnings('ignore')

class SpecificPreprocessingMethods:
    """구체적인 전처리 방법들의 실제 구현"""
    
    def __init__(self):
        self.fitted_transformers = {}
    
    # =============================================================================
    # 변환 방법들 (Transform Methods)
    # =============================================================================
    
    def yeo_johnson_transform(self, data: pd.Series, column_name: str) -> pd.Series:
        """극도로 왜도가 높은 경우 - Yeo-Johnson 변환"""
        transformed_data, lambda_param = yeojohnson(data.dropna())
        result = pd.Series(transformed_data, index=data.dropna().index)
        print(f"🔄 {column_name}: Yeo-Johnson 변환 적용 (λ={lambda_param:.3f})")
        return result
    
    def boxcox_transform(self, data: pd.Series, column_name: str) -> pd.Series:
        """높은 왜도 + 첨도 - Box-Cox 변환"""
        if (data <= 0).any():
            # 음수나 0이 있으면 상수를 더해서 양수로 만듦
            data_positive = data - data.min() + 1
            transformed_data, lambda_param = boxcox(data_positive.dropna())
        else:
            transformed_data, lambda_param = boxcox(data.dropna())
        
        result = pd.Series(transformed_data, index=data.dropna().index)
        print(f"🔄 {column_name}: Box-Cox 변환 적용 (λ={lambda_param:.3f})")
        return result
    
    def log_transform(self, data: pd.Series, column_name: str) -> pd.Series:
        """적당히 높은 왜도 - 로그 변환"""
        if (data <= 0).any():
            # 음수나 0이 있으면 상수를 더해서 양수로 만듦
            data_positive = data - data.min() + 1
            result = np.log(data_positive)
        else:
            result = np.log(data)
        print(f"🔄 {column_name}: 로그 변환 적용")
        return result
    
    def sqrt_transform(self, data: pd.Series, column_name: str) -> pd.Series:
        """중간 왜도 - 제곱근 변환"""
        if (data < 0).any():
            # 음수가 있으면 상수를 더해서 양수로 만듦
            data_positive = data - data.min()
            result = np.sqrt(data_positive)
        else:
            result = np.sqrt(data)
        print(f"🔄 {column_name}: 제곱근 변환 적용")
        return result
    
    def winsorize(self, data: pd.Series, column_name: str, limits: tuple = (0.05, 0.05)) -> pd.Series:
        """높은 첨도 - 윈저화 (극값 제한)"""
        from scipy.stats.mstats import winsorize
        result = pd.Series(winsorize(data, limits=limits), index=data.index)
        print(f"🔄 {column_name}: 윈저화 적용 (하위 {limits[0]*100}%, 상위 {limits[1]*100}% 제한)")
        return result
    
    # =============================================================================
    # 이상치 탐지 방법들 (Outlier Detection)
    # =============================================================================
    
    def isolation_forest(self, data: pd.DataFrame, column_name: str, contamination: float = 0.1) -> pd.Series:
        """다변량 이상치 탐지 - Isolation Forest"""
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers = iso_forest.fit_predict(data.select_dtypes(include=[np.number]))
        result = pd.Series(outliers == -1, index=data.index)
        print(f"🔍 {column_name}: Isolation Forest 이상치 탐지 ({result.sum()}개 이상치 발견)")
        return result
    
    def iqr_method(self, data: pd.Series, column_name: str) -> pd.Series:
        """단변량 이상치 탐지 - IQR 방법"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = (data < lower_bound) | (data > upper_bound)
        print(f"🔍 {column_name}: IQR 방법 이상치 탐지 ({outliers.sum()}개 이상치 발견)")
        return outliers
    
    def z_score_method(self, data: pd.Series, column_name: str, threshold: float = 3.0) -> pd.Series:
        """정규분포 이상치 탐지 - Z-score 방법"""
        z_scores = np.abs(stats.zscore(data.dropna()))
        outliers = pd.Series(False, index=data.index)
        outliers.loc[data.dropna().index] = z_scores > threshold
        print(f"🔍 {column_name}: Z-score 방법 이상치 탐지 ({outliers.sum()}개 이상치 발견)")
        return outliers
    
    # =============================================================================
    # 스케일링 방법들 (Scaling Methods)
    # =============================================================================
    
    def robust_scaler(self, data: pd.Series, column_name: str) -> pd.Series:
        """높은 분산 + 이상치 - Robust Scaler"""
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
        self.fitted_transformers[f'{column_name}_robust_scaler'] = scaler
        print(f"⚖️ {column_name}: Robust Scaler 적용 (중앙값 기반)")
        return pd.Series(scaled_data, index=data.index)
    
    def standard_scaler(self, data: pd.Series, column_name: str) -> pd.Series:
        """적당한 분산 - Standard Scaler"""
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
        self.fitted_transformers[f'{column_name}_standard_scaler'] = scaler
        print(f"⚖️ {column_name}: Standard Scaler 적용 (평균=0, 표준편차=1)")
        return pd.Series(scaled_data, index=data.index)
    
    def min_max_scaler(self, data: pd.Series, column_name: str) -> pd.Series:
        """기본 스케일링 - MinMax Scaler"""
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
        self.fitted_transformers[f'{column_name}_minmax_scaler'] = scaler
        print(f"⚖️ {column_name}: MinMax Scaler 적용 (범위 [0,1])")
        return pd.Series(scaled_data, index=data.index)
    
    # =============================================================================
    # 인코딩 방법들 (Encoding Methods)
    # =============================================================================
    
    def binary_encode(self, data: pd.Series, column_name: str) -> pd.DataFrame:
        """이진 변수 - Binary Encoding"""
        # 간단한 이진 인코딩
        unique_values = data.unique()
        mapping = {unique_values[0]: 0, unique_values[1]: 1}
        result = pd.DataFrame({f'{column_name}_binary': data.map(mapping)})
        print(f"🏷️ {column_name}: Binary Encoding 적용 ({mapping})")
        return result
    
    def one_hot_encode(self, data: pd.Series, column_name: str) -> pd.DataFrame:
        """낮은 카디널리티 - One-Hot Encoding"""
        encoded = pd.get_dummies(data, prefix=column_name)
        print(f"🏷️ {column_name}: One-Hot Encoding 적용 ({encoded.shape[1]}개 컬럼 생성)")
        return encoded
    
    def ordinal_encode(self, data: pd.Series, column_name: str) -> pd.Series:
        """중간 카디널리티 - Ordinal Encoding"""
        encoder = OrdinalEncoder()
        encoded_data = encoder.fit_transform(data.values.reshape(-1, 1)).flatten()
        self.fitted_transformers[f'{column_name}_ordinal_encoder'] = encoder
        print(f"🏷️ {column_name}: Ordinal Encoding 적용 ({len(encoder.categories_[0])}개 카테고리)")
        return pd.Series(encoded_data, index=data.index)
    
    def target_encode(self, data: pd.Series, target: pd.Series, column_name: str) -> pd.Series:
        """높은 카디널리티 - Target Encoding"""
        target_means = data.groupby(data).apply(lambda x: target.loc[x.index].mean())
        encoded_data = data.map(target_means)
        print(f"🏷️ {column_name}: Target Encoding 적용 ({len(target_means)}개 카테고리)")
        return encoded_data
    
    def feature_hashing(self, data: pd.Series, column_name: str, n_features: int = 100) -> pd.DataFrame:
        """매우 높은 카디널리티 - Feature Hashing"""
        from sklearn.feature_extraction import FeatureHasher
        hasher = FeatureHasher(n_features=n_features, input_type='string')
        hashed_features = hasher.fit_transform(data.astype(str).values.reshape(-1, 1))
        
        result = pd.DataFrame(
            hashed_features.toarray(),
            columns=[f'{column_name}_hash_{i}' for i in range(n_features)],
            index=data.index
        )
        print(f"🏷️ {column_name}: Feature Hashing 적용 ({n_features}개 해시 피처)")
        return result
    
    # =============================================================================
    # 특성 선택 방법들 (Feature Selection)
    # =============================================================================
    
    def drop_constant(self, data: pd.Series, column_name: str) -> None:
        """상수 컬럼 제거"""
        print(f"🗑️ {column_name}: 상수 컬럼 제거 (고유값 {data.nunique()}개)")
        return None
    
    def feature_selection(self, data: pd.Series, column_name: str, threshold: float = 0.1) -> pd.Series:
        """저분산 특성 선택"""
        selector = VarianceThreshold(threshold=threshold)
        selected = selector.fit_transform(data.values.reshape(-1, 1))
        if selected.shape[1] == 0:
            print(f"🗑️ {column_name}: 저분산으로 인한 특성 제거")
            return None
        else:
            print(f"✅ {column_name}: 분산 임계값 통과")
            return data

# =============================================================================
# 사용 예시
# =============================================================================

def demonstrate_specific_preprocessing():
    """구체적인 전처리 방법들의 사용 예시"""
    print("=" * 80)
    print("🔧 구체적인 전처리 방법들 - 실제 구현 예시")
    print("=" * 80)
    
    # 테스트 데이터 생성
    np.random.seed(42)
    n_samples = 1000
    
    # 다양한 분포의 데이터 생성
    normal_data = pd.Series(np.random.normal(0, 1, n_samples), name='normal_col')
    skewed_data = pd.Series(np.random.exponential(2, n_samples), name='skewed_col')
    very_skewed_data = pd.Series(np.random.pareto(0.1, n_samples), name='very_skewed_col')
    
    # 범주형 데이터 생성
    low_card_data = pd.Series(np.random.choice(['A', 'B', 'C'], n_samples), name='low_card')
    high_card_data = pd.Series([f'category_{i}' for i in np.random.randint(0, 500, n_samples)], name='high_card')
    
    # 전처리 객체 생성
    preprocessor = SpecificPreprocessingMethods()
    
    print("\n📊 변환 방법들 테스트:")
    print("-" * 50)
    
    # 변환 방법들 테스트
    transformed_normal = preprocessor.winsorize(normal_data, 'normal_col')
    transformed_skewed = preprocessor.log_transform(skewed_data, 'skewed_col')
    transformed_very_skewed = preprocessor.yeo_johnson_transform(very_skewed_data, 'very_skewed_col')
    
    print("\n🔍 이상치 탐지 방법들 테스트:")
    print("-" * 50)
    
    # 이상치 탐지 방법들 테스트
    outliers_iqr = preprocessor.iqr_method(normal_data, 'normal_col')
    outliers_zscore = preprocessor.z_score_method(skewed_data, 'skewed_col')
    
    print("\n⚖️ 스케일링 방법들 테스트:")
    print("-" * 50)
    
    # 스케일링 방법들 테스트
    scaled_robust = preprocessor.robust_scaler(very_skewed_data, 'very_skewed_col')
    scaled_standard = preprocessor.standard_scaler(normal_data, 'normal_col')
    scaled_minmax = preprocessor.min_max_scaler(skewed_data, 'skewed_col')
    
    print("\n🏷️ 인코딩 방법들 테스트:")
    print("-" * 50)
    
    # 인코딩 방법들 테스트
    encoded_onehot = preprocessor.one_hot_encode(low_card_data, 'low_card')
    encoded_ordinal = preprocessor.ordinal_encode(low_card_data, 'low_card')
    encoded_hash = preprocessor.feature_hashing(high_card_data, 'high_card', n_features=50)
    
    print("\n✅ 모든 전처리 방법 테스트 완료!")
    print("=" * 80)

if __name__ == "__main__":
    demonstrate_specific_preprocessing() 