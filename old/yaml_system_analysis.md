# 📊 YAML 기반 규칙 시스템 분석

## 🔍 시스템 구조 분석

### 1. 전체 아키텍처
```python
# 계층적 구조
RuleSet (YAML 로더)
├── PreprocessingRecommender (규칙 적용)
├── VisualizationRecommender (시각화 추천)
└── RecommendationEngine (통합 엔진)
```

### 2. 핵심 컴포넌트 분석

#### A. RuleSet 클래스 (설정 관리)
```python
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
```

**장점:**
- ✅ **외부 설정 파일 분리**: 코드 수정 없이 규칙 변경 가능
- ✅ **구조화된 데이터**: dataclass로 타입 안전성 확보
- ✅ **확장성**: 새로운 데이터 타입 규칙 추가 용이

**한계점:**
- ❌ **기본 데이터 타입만 지원**: 수치형, 범주형, 날짜/시간만 처리
- ❌ **동적 규칙 부족**: 데이터 컨텍스트별 규칙 조정 미지원

#### B. PreprocessingRecommender 클래스 (규칙 적용)
```python
def _apply_numeric_rules(self, col: str, s: pd.Series) -> List[Tuple[str, str]]:
    rec: List[Tuple[str, str]] = []
    r = self.rules.numeric
    
    # ① Missing 처리
    miss = s.isna().mean()
    for cond in r["missing_ratio"]:
        if miss > cond.get("gt", -np.inf):
            rec.append((cond["action"], cond["why"]))
            break  # 첫 매칭만 기록
    
    # ② Skew / Kurtosis 처리
    skew, kurt = s.skew(), s.kurtosis()
    for cond in r["skew_kurt"]:
        if abs(skew) > cond.get("skew", 0) or kurt > cond.get("kurt", 1e9):
            rec.append((cond["action"], cond["why"]))
            break
    
    # ③ 변동계수 처리
    mean, std = s.mean(), s.std()
    cv = std / mean if mean else np.inf
    for cond in r["variance"]:
        if cv < cond.get("cv_lt", -np.inf):
            rec.append((cond["action"], cond["why"]))
            break
    
    return rec
```

**장점:**
- ✅ **순차적 규칙 적용**: 우선순위 있는 규칙 매칭
- ✅ **설명 가능한 추천**: action + why 쌍으로 근거 제공
- ✅ **모듈화된 구조**: 데이터 타입별 독립적 처리

**한계점:**
- ❌ **단일 조건 매칭**: 복합 조건 처리 불가
- ❌ **컨텍스트 미고려**: 데이터 전체 특성 반영 부족
- ❌ **고정된 통계**: 왜도, 첨도, CV만 활용

### 3. YAML 규칙 파일 분석
```yaml
numeric:
  missing_ratio:
    - {gt: 0.40, action: "drop", why: "결측률>40%"}
    - {gt: 0.15, action: "advanced_impute", why: "결측률 15~40% (KNN/Iterative)"}
    - {gt: 0.00, action: "simple_impute", why: "소량 결측 (mean/median)"}
  
  skew_kurt:
    - {skew: 2, kurt: 7, action: "transform", why: "심한 비대칭·첨도"}
    - {skew: 1, action: "outlier_detect", why: "약한 왜도 - 이상치 확인"}
  
  variance:
    - {cv_lt: 0.1, action: "low_variance_drop", why: "변동성↓"}

categorical:
  missing_ratio:
    - {gt: 0.30, action: "drop", why: "결측률>30%"}
    - {gt: 0.10, action: "impute_mode", why: "결측률 10~30% (mode)"}
  
  cardinality:
    - {lte: 10, action: "one_hot", why: "저카디널리티"}
    - {lte: 50, action: "ordinal_encode", why: "중간 카디널리티"}
    - {lte: 1000, action: "target_encode", why: "고카디널리티 - supervised"}
    - {gt: 1000, action: "hashing_encode", why: "초고카디널리티"}

datetime:
  extract: ["year", "quarter", "month", "weekday"]
```

**장점:**
- ✅ **직관적 규칙 정의**: 임계값과 액션을 명확히 정의
- ✅ **유연한 수정**: 코드 변경 없이 규칙 업데이트 가능
- ✅ **범위 기반 처리**: 단계별 임계값으로 세밀한 제어

**한계점:**
- ❌ **고정된 임계값**: 모든 데이터에 동일한 기준 적용
- ❌ **단순 조건**: 복합 조건이나 상호작용 미지원
- ❌ **컨텍스트 미반영**: 데이터 특성별 규칙 조정 부족

## 🚀 개선된 시스템과 비교

### 현재 YAML 시스템 vs 개선된 시스템

| 특징 | YAML 시스템 | 개선된 시스템 | 차이점 |
|------|-------------|---------------|--------|
| **규칙 관리** | YAML 파일 기반 | 적응형 임계값 + 설정 | 동적 vs 정적 |
| **데이터 타입** | 3가지 (수치/범주/날짜) | 8가지 (세분화) | 400% 증가 |
| **컨텍스트 인식** | 없음 | 6가지 상황 인식 | 완전 새로운 기능 |
| **우선순위** | 순차적 매칭 | 3단계 우선순위 | 중요도 기반 정렬 |
| **코드 생성** | 없음 | 자동 생성 | 실행 가능한 템플릿 |

### 성능 비교

```python
# YAML 시스템 - 단순 규칙
def _apply_numeric_rules(self, col: str, s: pd.Series):
    for cond in r["missing_ratio"]:
        if miss > cond.get("gt", -np.inf):
            rec.append((cond["action"], cond["why"]))
            break

# 개선된 시스템 - 적응형 규칙
def _adjust_thresholds_by_context(self):
    if self.context == DataContext.HIGH_DIMENSIONAL:
        self.thresholds['missing_high'] = 0.20  # 더 엄격
    elif self.context == DataContext.SPARSE:
        self.thresholds['missing_high'] = 0.60  # 더 관대
```

## 💡 YAML 시스템의 발전 방향

### 1. 즉시 적용 가능한 개선사항

#### A. 확장된 YAML 규칙 구조
```yaml
# 개선된 rules.yaml 예시
numeric:
  rules:
    missing_ratio:
      - context: "general"
        conditions:
          - {gt: 0.40, action: "drop", why: "결측률>40%"}
          - {gt: 0.15, action: "advanced_impute", why: "결측률 15~40%"}
      - context: "high_dimensional"
        conditions:
          - {gt: 0.20, action: "drop", why: "고차원에서 결측률>20%"}
          - {gt: 0.10, action: "advanced_impute", why: "고차원 결측 처리"}
    
    distribution:
      - {skew_gt: 2, kurt_gt: 7, action: "yeo_johnson", why: "심한 비대칭"}
      - {skew_gt: 1, action: "robust_scale", why: "약한 왜도"}
      - {outlier_ratio_gt: 0.05, action: "outlier_detect", why: "이상치 다수"}

data_types:
  classification:
    - {pattern: ".*id.*", type: "id", action: "drop"}
    - {pattern: ".*key.*", type: "id", action: "drop"}
    - {unique_ratio_gt: 0.95, type: "id", action: "consider_drop"}
    - {dtype: "bool", type: "boolean", action: "label_encode"}
    - {avg_length_gt: 50, type: "text", action: "text_vectorize"}

contexts:
  detection:
    - {n_cols_gt: 100, context: "high_dimensional"}
    - {sparsity_gt: 0.8, context: "sparse"}
    - {time_cols_gt: 0, context: "timeseries"}
```

#### B. 컨텍스트 인식 PreprocessingRecommender
```python
class EnhancedPreprocessingRecommender:
    def __init__(self, rules: RuleSet):
        self.rules = rules
        self.context = "general"
        self.data_types = {}
    
    def _detect_context(self, df: pd.DataFrame) -> str:
        """데이터 컨텍스트 자동 감지"""
        n_rows, n_cols = df.shape
        
        # 컨텍스트 감지 규칙 적용
        for rule in self.rules.contexts["detection"]:
            if "n_cols_gt" in rule and n_cols > rule["n_cols_gt"]:
                return rule["context"]
            if "sparsity_gt" in rule:
                sparsity = (df == 0).sum().sum() / (n_rows * n_cols)
                if sparsity > rule["sparsity_gt"]:
                    return rule["context"]
        
        return "general"
    
    def _classify_data_type(self, series: pd.Series) -> str:
        """데이터 타입 자동 분류"""
        col_name = series.name.lower()
        
        # 타입 분류 규칙 적용
        for rule in self.rules.data_types["classification"]:
            if "pattern" in rule and re.search(rule["pattern"], col_name):
                return rule["type"]
            if "unique_ratio_gt" in rule:
                unique_ratio = series.nunique() / len(series)
                if unique_ratio > rule["unique_ratio_gt"]:
                    return rule["type"]
        
        return "unknown"
    
    def _apply_contextual_rules(self, col: str, s: pd.Series, data_type: str):
        """컨텍스트 기반 규칙 적용"""
        rules = self.rules.numeric["rules"]
        
        # 컨텍스트별 규칙 선택
        for rule_group in rules["missing_ratio"]:
            if rule_group["context"] == self.context:
                conditions = rule_group["conditions"]
                break
        else:
            conditions = rules["missing_ratio"][0]["conditions"]  # 기본값
        
        # 규칙 적용
        miss = s.isna().mean()
        for cond in conditions:
            if miss > cond.get("gt", -np.inf):
                return (cond["action"], cond["why"])
        
        return None
```

### 2. 통합 개선 전략

#### A. 하이브리드 접근법
```python
class HybridRecommendationEngine:
    """YAML 규칙 + 동적 분석 통합"""
    
    def __init__(self, rule_path: str = "rules.yaml"):
        self.yaml_recommender = PreprocessingRecommender(RuleSet.load(rule_path))
        self.dynamic_recommender = EnhancedPreprocessingRecommender()
    
    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        # 1. 동적 분석으로 컨텍스트 감지
        context = self.dynamic_recommender._analyze_data_context(df)
        
        # 2. YAML 규칙 적용 (기본 추천)
        yaml_recs = self.yaml_recommender.recommend(df)
        
        # 3. 동적 규칙으로 보완
        dynamic_recs = self.dynamic_recommender.recommend(df)
        
        # 4. 두 추천 결과 통합
        return self._merge_recommendations(yaml_recs, dynamic_recs, context)
```

#### B. 점진적 마이그레이션
```python
# Phase 1: YAML 규칙 확장
- 컨텍스트 기반 규칙 추가
- 데이터 타입 분류 규칙 추가
- 복합 조건 지원

# Phase 2: 동적 분석 통합
- 적응형 임계값 시스템 결합
- 실시간 컨텍스트 감지
- 우선순위 기반 추천

# Phase 3: 완전 통합
- 머신러닝 기반 규칙 학습
- 사용자 피드백 반영
- 도메인 특화 규칙 적용
```

## 🎯 실무 적용 권장사항

### 1. 단기 개선 (2-4주)
```python
# 현재 YAML 시스템 기반 즉시 개선
1. 확장된 규칙 파일 작성
2. 컨텍스트 감지 함수 추가
3. 데이터 타입 분류 개선
4. 우선순위 기반 정렬
```

### 2. 중기 개선 (1-2개월)
```python
# 하이브리드 시스템 구축
1. 동적 분석 엔진 통합
2. 적응형 임계값 시스템
3. 실행 가능한 코드 생성
4. 시각화 추천 고도화
```

### 3. 장기 개선 (3-6개월)
```python
# 완전 지능형 시스템
1. 머신러닝 기반 규칙 학습
2. 사용자 피드백 반영
3. 도메인 특화 규칙
4. 자동화 파이프라인
```

## 💻 구현 예시

### 확장된 YAML 시스템 구현
```python
class ExtendedYAMLRecommender:
    def __init__(self, rule_path: str = "enhanced_rules.yaml"):
        self.rules = RuleSet.load(rule_path)
        self.context = "general"
        self.data_types = {}
    
    def recommend(self, df: pd.DataFrame) -> Dict[str, Any]:
        # 1. 컨텍스트 감지
        self.context = self._detect_context(df)
        
        # 2. 데이터 타입 분류
        for col in df.columns:
            self.data_types[col] = self._classify_data_type(df[col])
        
        # 3. 컨텍스트별 규칙 적용
        recommendations = {}
        for col in df.columns:
            recs = self._apply_contextual_rules(col, df[col], self.data_types[col])
            recommendations[col] = {
                'data_type': self.data_types[col],
                'context': self.context,
                'recommendations': recs,
                'priority': self._calculate_priority(df[col], recs)
            }
        
        return {
            'preprocessing': recommendations,
            'context': self.context,
            'data_types': self.data_types,
            'summary': self._generate_summary(recommendations)
        }
```

## 🔍 결론

현재 YAML 기반 시스템은 **체계적이고 확장 가능한 구조**를 가지고 있어 매우 좋은 기반이 됩니다. 

**주요 장점:**
- 외부 설정 파일로 규칙 관리
- 구조화된 데이터 처리
- 순차적 규칙 매칭
- 설명 가능한 추천

**개선 방향:**
1. **컨텍스트 인식** 추가로 상황별 최적화
2. **데이터 타입 세분화**로 정확도 향상
3. **적응형 임계값**으로 유연성 확보
4. **우선순위 시스템**으로 사용자 경험 개선

이 시스템을 기반으로 점진적으로 개선하면 **매우 강력한 추천 시스템**을 구축할 수 있습니다! 