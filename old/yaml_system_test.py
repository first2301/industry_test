import pandas as pd
import numpy as np
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import tempfile
import os

# YAML 기반 시스템 복제 (테스트용)
@dataclass
class RuleSet:
    numeric: Dict[str, Any] = field(default_factory=dict)
    categorical: Dict[str, Any] = field(default_factory=dict)
    datetime: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str = "rules.yaml") -> "RuleSet":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

class PreprocessingRecommender:
    def __init__(self, rules: RuleSet):
        self.rules = rules

    def _apply_numeric_rules(self, col: str, s: pd.Series) -> List[Tuple[str, str]]:
        rec: List[Tuple[str, str]] = []
        r = self.rules.numeric

        # ① Missing
        miss = s.isna().mean()
        for cond in r["missing_ratio"]:
            if miss > cond.get("gt", -np.inf):
                rec.append((cond["action"], cond["why"]))
                break

        # ② Skew / Kurtosis
        skew, kurt = s.skew(), s.kurtosis()
        for cond in r["skew_kurt"]:
            if abs(skew) > cond.get("skew", 0) or kurt > cond.get("kurt", 1e9):
                rec.append((cond["action"], cond["why"]))
                break

        # ③ Coefficient of Variation
        mean, std = s.mean(), s.std()
        cv = std / mean if mean and not np.isnan(mean) and mean != 0 else np.inf
        for cond in r["variance"]:
            if cv < cond.get("cv_lt", -np.inf):
                rec.append((cond["action"], cond["why"]))
                break

        # ④ Outlier Detection (항상 첫 번째 방법 추천)
        if "outliers" in r and r["outliers"]:
            outlier_method = r["outliers"][0]
            rec.append((outlier_method["action"], outlier_method["why"]))

        # ⑤ Scaling (CV 기반)
        if "scaling" in r:
            for cond in r["scaling"]:
                if cv > cond.get("cv_gt", np.inf):
                    rec.append((cond["action"], cond["why"]))
                    break
            else:
                # 기본 스케일링
                if r["scaling"]:
                    default_scaling = r["scaling"][-1]
                    rec.append((default_scaling["action"], default_scaling["why"]))

        return rec

    def _apply_categorical_rules(self, col: str, s: pd.Series) -> List[Tuple[str, str]]:
        rec: List[Tuple[str, str]] = []
        r = self.rules.categorical

        miss = s.isna().mean()
        unique = s.nunique(dropna=True)

        for cond in r["missing_ratio"]:
            if miss > cond.get("gt", -np.inf):
                rec.append((cond["action"], cond["why"]))
                break

        for cond in r["cardinality"]:
            if (
                ("lte" in cond and unique <= cond["lte"])
                or ("gt" in cond and unique > cond["gt"])
            ):
                rec.append((cond["action"], cond["why"]))
                break
        return rec

    def recommend(self, df: pd.DataFrame) -> Dict[str, List[Dict[str, str]]]:
        out: Dict[str, List[Dict[str, str]]] = {}

        numeric_cols = df.select_dtypes(include="number").columns
        categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns

        # Numeric
        for c in numeric_cols:
            rec = self._apply_numeric_rules(c, df[c].dropna())
            out[c] = [{"action": a, "why": w} for a, w in rec]

        # Categorical
        for c in categorical_cols:
            rec = self._apply_categorical_rules(c, df[c])
            out[c] = [{"action": a, "why": w} for a, w in rec]

        return out

class RecommendationEngine:
    def __init__(self, rule_path: str = "rules.yaml"):
        self.rules = RuleSet.load(rule_path)
        self.preproc = PreprocessingRecommender(self.rules)

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "preprocessing": self.preproc.recommend(df),
        }

# 테스트용 YAML 규칙 파일 생성
def create_test_rules():
    rules = {
        'numeric': {
            'missing_ratio': [
                {'gt': 0.40, 'action': 'drop', 'why': 'missing_ratio_over_40'},
                {'gt': 0.15, 'action': 'advanced_impute', 'why': 'missing_ratio_15_40'},
                {'gt': 0.00, 'action': 'simple_impute', 'why': 'small_missing_values'}
            ],
            'skew_kurt': [
                {'skew': 3, 'kurt': 10, 'action': 'yeo_johnson_transform', 'why': 'extreme_skew_kurt'},
                {'skew': 2, 'kurt': 7, 'action': 'boxcox_transform', 'why': 'high_skew_kurt'},
                {'skew': 1.5, 'action': 'log_transform', 'why': 'moderate_high_skew'},
                {'skew': 1, 'action': 'sqrt_transform', 'why': 'moderate_skew'},
                {'kurt': 7, 'action': 'winsorize', 'why': 'high_kurtosis_only'}
            ],
            'variance': [
                {'cv_lt': 0.05, 'action': 'drop_constant', 'why': 'near_constant_values'},
                {'cv_lt': 0.1, 'action': 'feature_selection', 'why': 'low_variance'}
            ],
            'outliers': [
                {'action': 'isolation_forest', 'why': 'multivariate_outlier_detection'},
                {'action': 'iqr_method', 'why': 'univariate_outlier_detection'},
                {'action': 'z_score_method', 'why': 'normal_distribution_outliers'}
            ],
            'scaling': [
                {'cv_gt': 1.0, 'action': 'robust_scaler', 'why': 'high_variance_outliers'},
                {'cv_gt': 0.5, 'action': 'standard_scaler', 'why': 'moderate_variance'},
                {'action': 'min_max_scaler', 'why': 'bounded_scaling'}
            ]
        },
        'categorical': {
            'missing_ratio': [
                {'gt': 0.30, 'action': 'drop_column', 'why': 'missing_ratio_over_30'},
                {'gt': 0.10, 'action': 'impute_mode', 'why': 'missing_ratio_10_30'},
                {'gt': 0.05, 'action': 'impute_frequent', 'why': 'missing_ratio_5_10'}
            ],
            'cardinality': [
                {'lte': 2, 'action': 'binary_encode', 'why': 'binary_variable'},
                {'lte': 10, 'action': 'one_hot_encode', 'why': 'low_cardinality'},
                {'lte': 50, 'action': 'ordinal_encode', 'why': 'medium_cardinality'},
                {'lte': 1000, 'action': 'target_encode', 'why': 'high_cardinality'},
                {'gt': 1000, 'action': 'feature_hashing', 'why': 'very_high_cardinality'}
            ]
        }
    }
    return rules

# 테스트 데이터 생성
def create_test_data():
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        # 수치형 데이터 - 다양한 분포와 결측 패턴
        'normal_col': np.random.normal(50, 15, n_samples),  # 정규분포
        'skewed_col': np.random.exponential(2, n_samples),  # 왜도 높음
        'high_missing': np.random.normal(100, 20, n_samples),  # 높은 결측률
        'low_variance': np.ones(n_samples) * 10 + np.random.normal(0, 0.5, n_samples),  # 낮은 분산
        'very_skewed': np.random.pareto(0.5, n_samples),  # 매우 높은 왜도
        
        # 범주형 데이터 - 다양한 카디널리티
        'low_card': np.random.choice(['A', 'B', 'C'], n_samples),  # 저카디널리티
        'medium_card': np.random.choice([f'Cat_{i}' for i in range(25)], n_samples),  # 중간 카디널리티
        'high_card': np.random.choice([f'Item_{i}' for i in range(500)], n_samples),  # 고카디널리티
        'very_high_card': [f'ID_{i}' for i in range(n_samples)],  # 매우 고카디널리티
        'high_missing_cat': np.random.choice(['X', 'Y', 'Z'], n_samples),  # 높은 결측률 범주형
    }
    
    df = pd.DataFrame(data)
    
    # 결측값 추가
    df.loc[np.random.choice(n_samples, 500), 'high_missing'] = np.nan  # 50% 결측
    df.loc[np.random.choice(n_samples, 400), 'high_missing_cat'] = np.nan  # 40% 결측
    df.loc[np.random.choice(n_samples, 100), 'normal_col'] = np.nan  # 10% 결측
    
    return df

# 테스트 실행
def run_yaml_system_test():
    print("=" * 80)
    print("🧪 YAML 기반 추천 시스템 테스트")
    print("=" * 80)
    
    # 1. 테스트 데이터 및 규칙 생성
    df = create_test_data()
    rules = create_test_rules()
    
    # 임시 YAML 파일 생성
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(rules, f, default_flow_style=False, allow_unicode=True)
        rules_file = f.name
    
    try:
        print(f"📊 테스트 데이터 정보:")
        print(f"   - 데이터 형태: {df.shape}")
        print(f"   - 수치형 컬럼: {len(df.select_dtypes(include='number').columns)}개")
        print(f"   - 범주형 컬럼: {len(df.select_dtypes(include=['object']).columns)}개")
        print(f"   - 결측값 있는 컬럼: {df.isnull().any().sum()}개")
        
        # 2. 추천 엔진 실행
        engine = RecommendationEngine(rules_file)
        recommendations = engine.run(df)
        
        print(f"\n🔧 YAML 추천 결과:")
        print(f"   - 총 추천된 컬럼: {len(recommendations['preprocessing'])}개")
        
        # 3. 상세 추천 결과 출력
        print(f"\n📋 상세 추천 내용:")
        for col, recs in recommendations['preprocessing'].items():
            print(f"\n🔹 {col}:")
            
            # 컬럼 특성 정보
            if col in df.select_dtypes(include='number').columns:
                series = df[col]
                missing_ratio = series.isnull().mean()
                if len(series.dropna()) > 0:
                    skew = series.skew()
                    print(f"   📊 특성: 결측률={missing_ratio:.2%}, 왜도={skew:.2f}")
                else:
                    print(f"   📊 특성: 결측률={missing_ratio:.2%}")
            else:
                missing_ratio = df[col].isnull().mean()
                unique_count = df[col].nunique()
                print(f"   📊 특성: 결측률={missing_ratio:.2%}, 고유값={unique_count}개")
            
            # 추천 결과
            if recs:
                for rec in recs:
                    print(f"   ✅ {rec['action']}: {rec['why']}")
            else:
                print(f"   ⚪ 추천 없음")
        
        # 4. 규칙 매칭 분석
        print(f"\n🎯 규칙 매칭 분석:")
        action_counts = {}
        for col, recs in recommendations['preprocessing'].items():
            for rec in recs:
                action = rec['action']
                action_counts[action] = action_counts.get(action, 0) + 1
        
        print(f"   추천된 액션 분포:")
        for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {action}: {count}회")
        
        # 5. 시스템 특성 분석
        print(f"\n🔍 시스템 특성 분석:")
        print(f"   장점:")
        print(f"   ✅ 외부 YAML 파일로 규칙 관리")
        print(f"   ✅ 순차적 규칙 매칭으로 우선순위 적용")
        print(f"   ✅ 설명 가능한 추천 (action + why)")
        print(f"   ✅ 데이터 타입별 독립적 처리")
        
        print(f"\n   한계점:")
        print(f"   ❌ 고정된 임계값 (모든 데이터에 동일 적용)")
        print(f"   ❌ 단순 조건 매칭 (복합 조건 미지원)")
        print(f"   ❌ 컨텍스트 미고려 (데이터 전체 특성 반영 부족)")
        print(f"   ❌ 기본 데이터 타입만 지원 (수치형/범주형)")
        
        # 6. 개선 방향 제시
        print(f"\n💡 개선 방향:")
        print(f"   1. 컨텍스트 기반 적응형 임계값")
        print(f"   2. 데이터 타입 세분화 (ID, 텍스트, 불린 등)")
        print(f"   3. 복합 조건 및 상호작용 지원")
        print(f"   4. 우선순위 시스템 및 코드 생성")
        
    finally:
        # 임시 파일 정리
        if os.path.exists(rules_file):
            os.unlink(rules_file)
    
    print(f"\n" + "=" * 80)
    print("✅ 테스트 완료")
    print("=" * 80)

if __name__ == "__main__":
    run_yaml_system_test() 