import pandas as pd
import numpy as np
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class RuleSet:
    numeric: Dict[str, Any] = field(default_factory=dict)
    categorical: Dict[str, Any] = field(default_factory=dict)
    datetime: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str = "improved_rules.yaml") -> "RuleSet":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

class ImprovedPreprocessingRecommender:
    """개선된 전처리 추천 시스템 - 더 세분화된 규칙 적용"""
    
    def __init__(self, rules: RuleSet):
        self.rules = rules
    
    def _apply_numeric_rules(self, col: str, s: pd.Series) -> List[Tuple[str, str]]:
        """개선된 수치형 규칙 적용"""
        rec: List[Tuple[str, str]] = []
        r = self.rules.numeric
        
        # 기본 통계 계산
        miss = s.isna().mean()
        skew, kurt = s.skew(), s.kurtosis()
        mean, std = s.mean(), s.std()
        cv = std / mean if mean and not np.isnan(mean) and mean != 0 else np.inf
        unique_count = s.nunique()
        unique_ratio = unique_count / len(s)
        
        # 1. 결측치 처리
        for cond in r["missing_ratio"]:
            if miss > cond.get("gt", -np.inf):
                rec.append((cond["action"], cond["why"]))
                break
        
        # 2. 개선된 왜도/첨도 규칙 - 더 세분화
        for cond in r["skew_kurt"]:
            skew_threshold = cond.get("skew", 0)
            kurt_threshold = cond.get("kurt", 1e9)
            
            # 왜도와 첨도 모두 체크
            if (abs(skew) > skew_threshold or kurt > kurt_threshold):
                rec.append((cond["action"], cond["why"]))
                break
        
        # 3. 분산 기반 규칙
        for cond in r["variance"]:
            if cv < cond.get("cv_lt", -np.inf):
                rec.append((cond["action"], cond["why"]))
                break
        
        # 4. 데이터 타입별 특별 규칙
        if "data_type_specific" in r:
            for cond in r["data_type_specific"]:
                # 컬럼명 기반 규칙
                if "name_contains" in cond:
                    if cond["name_contains"].lower() in col.lower():
                        # 추가 조건 체크
                        if "skew_lt" in cond and skew < cond["skew_lt"]:
                            rec.append((cond["action"], cond["why"]))
                        elif "unique_lt" in cond and unique_count < cond["unique_lt"]:
                            rec.append((cond["action"], cond["why"]))
                        elif "unique_ratio_gt" in cond and unique_ratio > cond["unique_ratio_gt"]:
                            rec.append((cond["action"], cond["why"]))
                        else:
                            rec.append((cond["action"], cond["why"]))
        
        # 5. 이상치 탐지 (항상 첫 번째 방법)
        if "outliers" in r and r["outliers"]:
            outlier_method = r["outliers"][0]
            rec.append((outlier_method["action"], outlier_method["why"]))
        
        # 6. 스케일링 (CV 기반)
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
        """개선된 범주형 규칙 적용"""
        rec: List[Tuple[str, str]] = []
        r = self.rules.categorical
        
        miss = s.isna().mean()
        unique = s.nunique(dropna=True)
        
        # 1. 결측치 처리
        for cond in r["missing_ratio"]:
            if miss > cond.get("gt", -np.inf):
                rec.append((cond["action"], cond["why"]))
                break
        
        # 2. 카디널리티 기반 인코딩
        for cond in r["cardinality"]:
            if ("lte" in cond and unique <= cond["lte"]) or ("gt" in cond and unique > cond["gt"]):
                rec.append((cond["action"], cond["why"]))
                break
        
        # 3. 텍스트 데이터 특별 규칙
        if "text_specific" in r:
            for cond in r["text_specific"]:
                if "name_contains" in cond and cond["name_contains"].lower() in col.lower():
                    rec.append((cond["action"], cond["why"]))
        
        return rec
    
    def recommend(self, df: pd.DataFrame) -> Dict[str, List[Dict[str, str]]]:
        """개선된 추천 시스템"""
        out: Dict[str, List[Dict[str, str]]] = {}
        
        numeric_cols = df.select_dtypes(include="number").columns
        categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns
        
        # 수치형 컬럼 처리
        for c in numeric_cols:
            rec = self._apply_numeric_rules(c, df[c].dropna())
            out[c] = [{"action": a, "why": w} for a, w in rec]
        
        # 범주형 컬럼 처리
        for c in categorical_cols:
            rec = self._apply_categorical_rules(c, df[c])
            out[c] = [{"action": a, "why": w} for a, w in rec]
        
        return out

class ImprovedRecommendationEngine:
    """개선된 추천 엔진"""
    
    def __init__(self, rule_path: str = "improved_rules.yaml"):
        self.rules = RuleSet.load(rule_path)
        self.preproc = ImprovedPreprocessingRecommender(self.rules)
    
    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "preprocessing": self.preproc.recommend(df),
        }

# 테스트 함수
def test_improved_recommendations():
    """개선된 추천 시스템 테스트"""
    print("=" * 80)
    print("🔧 개선된 추천 시스템 테스트")
    print("=" * 80)
    
    # 테스트 데이터 생성 (실제 데이터와 유사하게)
    np.random.seed(42)
    n_samples = 1000
    
    # 실제 데이터와 유사한 특성으로 생성
    data = {
        'MIXA_PASTEUR_STATE': np.random.exponential(2, n_samples),  # 왜도 ~3
        'MIXB_PASTEUR_STATE': np.random.exponential(2, n_samples),  # 왜도 ~3
        'MIXA_PASTEUR_TEMP': -np.random.exponential(1, n_samples),  # 음수 왜도 ~-4
        'MIXB_PASTEUR_TEMP': -np.random.exponential(1, n_samples),  # 음수 왜도 ~-4
        'STD_DT': [f"2023-{i:02d}-{j:02d}" for i, j in zip(np.random.randint(1, 13, n_samples), np.random.randint(1, 29, n_samples))],
        'INSP': np.random.choice(['PASS', 'FAIL'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # 개선된 엔진으로 테스트
    engine = ImprovedRecommendationEngine()
    recommendations = engine.run(df)
    
    print("\n📊 데이터 특성:")
    for col in df.select_dtypes(include=[np.number]).columns:
        skew = df[col].skew()
        kurt = df[col].kurtosis()
        print(f"  {col}: 왜도={skew:.2f}, 첨도={kurt:.2f}")
    
    print("\n🎯 개선된 추천 결과:")
    for col, recs in recommendations['preprocessing'].items():
        print(f"\n🔹 {col}:")
        for rec in recs:
            print(f"  ✅ {rec['action']}: {rec['why']}")
    
    return recommendations

if __name__ == "__main__":
    test_improved_recommendations() 