import pandas as pd
import numpy as np
from research_based_engine import ResearchBasedRecommendationEngine

# 실제 데이터 로드
df1 = pd.read_csv('./data/Test_01.csv')
df2 = pd.read_csv('./data/Test_02.csv')

def test_real_data():
    """실제 데이터로 연구 기반 엔진 테스트"""
    print("=" * 80)
    print("🔬 실제 데이터 연구 기반 추천 시스템 테스트")
    print("=" * 80)
    
    engine = ResearchBasedRecommendationEngine()
    
    # Test_01.csv 테스트
    print("\n📊 Test_01.csv 데이터 분석:")
    print(f"  - 데이터 형태: {df1.shape}")
    print(f"  - 수치형 컬럼: {len(df1.select_dtypes(include='number').columns)}개")
    print(f"  - 범주형 컬럼: {len(df1.select_dtypes(include=['object']).columns)}개")
    
    print("\n📈 수치형 컬럼 특성:")
    for col in df1.select_dtypes(include=[np.number]).columns:
        skew = df1[col].skew()
        kurt = df1[col].kurtosis()
        cv = df1[col].std() / df1[col].mean() if df1[col].mean() != 0 else np.inf
        print(f"  {col}: 왜도={skew:.2f}, 첨도={kurt:.2f}, CV={cv:.2f}")
    
    print("\n🎯 Test_01.csv 연구 기반 추천:")
    rec1 = engine.run(df1)
    for col, recs in rec1['preprocessing'].items():
        if col.startswith('_'):
            print(f"\n🔹 {col}:")
        else:
            print(f"\n🔹 {col}:")
        for rec in recs:
            print(f"  ✅ {rec['action']}: {rec['why']}")
    
    # Test_02.csv 테스트
    print("\n" + "="*80)
    print("📊 Test_02.csv 데이터 분석:")
    print(f"  - 데이터 형태: {df2.shape}")
    print(f"  - 수치형 컬럼: {len(df2.select_dtypes(include='number').columns)}개")
    print(f"  - 범주형 컬럼: {len(df2.select_dtypes(include=['object']).columns)}개")
    
    print("\n📈 수치형 컬럼 특성:")
    for col in df2.select_dtypes(include=[np.number]).columns:
        skew = df2[col].skew()
        kurt = df2[col].kurtosis()
        cv = df2[col].std() / df2[col].mean() if df2[col].mean() != 0 else np.inf
        print(f"  {col}: 왜도={skew:.2f}, 첨도={kurt:.2f}, CV={cv:.2f}")
    
    print("\n🎯 Test_02.csv 연구 기반 추천:")
    rec2 = engine.run(df2)
    for col, recs in rec2['preprocessing'].items():
        if col.startswith('_'):
            print(f"\n🔹 {col}:")
        else:
            print(f"\n🔹 {col}:")
        for rec in recs:
            print(f"  ✅ {rec['action']}: {rec['why']}")

if __name__ == "__main__":
    test_real_data() 