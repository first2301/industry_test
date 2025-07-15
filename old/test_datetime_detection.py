import pandas as pd
import numpy as np
from research_based_engine import ResearchBasedRecommendationEngine
from datetime import datetime, timedelta

def test_datetime_detection():
    """다양한 날짜 데이터 형식 테스트"""
    print("=" * 80)
    print("📅 날짜 데이터 감지 및 처리 테스트")
    print("=" * 80)
    
    # 다양한 날짜 형식의 테스트 데이터 생성
    np.random.seed(42)
    n_samples = 100
    
    # 다양한 날짜 형식
    date_formats = {
        'ISO_DATE': [f"2023-{i:02d}-{j:02d}" for i, j in zip(np.random.randint(1, 13, n_samples), np.random.randint(1, 29, n_samples))],
        'US_DATE': [f"{i:02d}/{j:02d}/2023" for i, j in zip(np.random.randint(1, 13, n_samples), np.random.randint(1, 29, n_samples))],
        'EU_DATE': [f"{i:02d}-{j:02d}-2023" for i, j in zip(np.random.randint(1, 13, n_samples), np.random.randint(1, 29, n_samples))],
        'DATETIME': [f"2023-{i:02d}-{j:02d} {k:02d}:{l:02d}:{m:02d}" for i, j, k, l, m in zip(
            np.random.randint(1, 13, n_samples), 
            np.random.randint(1, 29, n_samples),
            np.random.randint(0, 24, n_samples),
            np.random.randint(0, 60, n_samples),
            np.random.randint(0, 60, n_samples)
        )],
        'TIMESTAMP': [f"{k:02d}:{l:02d}:{m:02d}" for k, l, m in zip(
            np.random.randint(0, 24, n_samples),
            np.random.randint(0, 60, n_samples),
            np.random.randint(0, 60, n_samples)
        )],
        'NON_DATE': ['PASS', 'FAIL', 'PENDING'] * (n_samples // 3 + 1),
        'MIXED_DATE': ['2023-01-01', 'PASS', '2023-02-15', 'FAIL', '2023-03-20'] * (n_samples // 5 + 1)
    }
    
    engine = ResearchBasedRecommendationEngine()
    
    for format_name, date_data in date_formats.items():
        print(f"\n🔹 {format_name} 형식 테스트:")
        print(f"  샘플 데이터: {date_data[:3]}")
        
        # 데이터 길이 맞추기
        actual_length = len(date_data)
        numeric_data = np.random.normal(0, 1, actual_length)
        
        # 테스트 데이터프레임 생성
        test_df = pd.DataFrame({
            'date_column': date_data,
            'numeric_col': numeric_data
        })
        
        # 날짜 감지 테스트
        recommender = engine.preproc
        is_datetime = recommender._is_datetime_column(test_df['date_column'], 'date_column')
        print(f"  날짜 감지 결과: {is_datetime}")
        
        if is_datetime:
            # 날짜 특성 추출
            datetime_features = recommender._extract_datetime_features(test_df['date_column'])
            print(f"  날짜 특성: {datetime_features}")
            
            # 추천 결과
            recommendations = engine.run(test_df)
            for col, recs in recommendations['preprocessing'].items():
                if col == 'date_column':
                    print(f"  추천사항:")
                    for rec in recs:
                        print(f"    ✅ {rec['action']}: {rec['why']}")
        else:
            print(f"  일반 범주형 데이터로 처리됨")

def test_real_data_with_dates():
    """실제 데이터의 날짜 처리 테스트"""
    print("\n" + "=" * 80)
    print("📊 실제 데이터 날짜 처리 테스트")
    print("=" * 80)
    
    # 실제 데이터 로드
    df1 = pd.read_csv('./data/Test_01.csv')
    
    engine = ResearchBasedRecommendationEngine()
    
    print(f"📈 Test_01.csv 데이터 분석:")
    print(f"  - 데이터 형태: {df1.shape}")
    print(f"  - 컬럼 타입:")
    for col, dtype in df1.dtypes.items():
        print(f"    {col}: {dtype}")
    
    # STD_DT 컬럼 특별 분석
    std_dt_col = df1['STD_DT']
    print(f"\n📅 STD_DT 컬럼 분석:")
    print(f"  - 데이터 타입: {std_dt_col.dtype}")
    print(f"  - 샘플 데이터: {std_dt_col.head().tolist()}")
    print(f"  - 고유값 개수: {std_dt_col.nunique()}")
    
    # 날짜 감지 테스트
    recommender = engine.preproc
    is_datetime = recommender._is_datetime_column(std_dt_col, 'STD_DT')
    print(f"  - 날짜 감지 결과: {is_datetime}")
    
    if is_datetime:
        datetime_features = recommender._extract_datetime_features(std_dt_col)
        print(f"  - 날짜 특성: {datetime_features}")
    
    # 전체 추천 결과
    print(f"\n🎯 전체 추천 결과:")
    recommendations = engine.run(df1)
    for col, recs in recommendations['preprocessing'].items():
        if col == 'STD_DT':
            print(f"\n🔹 {col} (날짜 데이터):")
            for rec in recs:
                print(f"  ✅ {rec['action']}: {rec['why']}")

def test_edge_cases():
    """엣지 케이스 테스트"""
    print("\n" + "=" * 80)
    print("🔍 엣지 케이스 테스트")
    print("=" * 80)
    
    edge_cases = {
        'EMPTY_DATES': pd.Series([None, None, None], name='empty_dates'),
        'INVALID_DATES': pd.Series(['invalid', 'not-a-date', '2023-13-45'], name='invalid_dates'),
        'MIXED_VALID_INVALID': pd.Series(['2023-01-01', 'invalid', '2023-02-15', 'not-date'], name='mixed_dates'),
        'HIGH_CARDINALITY_DATES': pd.Series([f"2023-{i:02d}-{j:02d}" for i in range(1, 13) for j in range(1, 29)], name='high_card_dates'),
        'DATE_LIKE_NAMES': pd.Series(['PASS', 'FAIL', 'PENDING'], name='date_column'),  # 컬럼명은 날짜 같지만 데이터는 아님
    }
    
    engine = ResearchBasedRecommendationEngine()
    
    for case_name, test_series in edge_cases.items():
        print(f"\n🔹 {case_name}:")
        print(f"  샘플 데이터: {test_series.head().tolist()}")
        
        recommender = engine.preproc
        is_datetime = recommender._is_datetime_column(test_series, test_series.name)
        print(f"  날짜 감지 결과: {is_datetime}")
        
        if is_datetime:
            datetime_features = recommender._extract_datetime_features(test_series)
            print(f"  날짜 특성: {datetime_features}")

if __name__ == "__main__":
    test_datetime_detection()
    test_real_data_with_dates()
    test_edge_cases() 