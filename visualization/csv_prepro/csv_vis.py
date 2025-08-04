import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import os

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# lib 모듈에서 추상화된 클래스들 임포트
from lib import DataAugmenter, DataVisualizer, DataUtils

# 페이지 설정
st.set_page_config(layout='wide')
st.title("1. CSV 정형 데이터 증강 및 시각화 도구")

# 클래스 인스턴스 생성
augmenter = DataAugmenter()
visualizer = DataVisualizer()

# 임시로 메서드를 직접 정의 (모듈 캐싱 문제 해결)
def setup_augmentation_parameters(categorical_cols, numeric_cols, df):
    """사이드바에서 증강 파라미터를 설정하고 반환합니다."""
    with st.sidebar:
        st.markdown("---")
        st.markdown("**1. 증강 파라미터 설정**")
        
        # SMOTE 관련 설정
        st.markdown("**2. SMOTE 설정**")
        use_smote = st.checkbox("SMOTE 사용", value=False, help="불균형 데이터 증강을 위해 SMOTE를 사용합니다.")
        
        target_col = None
        imb_method = None
        
        if use_smote:
            all_cols = categorical_cols + numeric_cols
            target_col = st.selectbox("타겟(레이블) 컬럼 선택", all_cols, key="target_select")
            
            if target_col:
                if target_col in numeric_cols:
                    unique_count = df[target_col].nunique()
                    if unique_count > 20:
                        st.warning("⚠️ 연속형 데이터로 보입니다.")
                    else:
                        st.success("✅ 범주형으로 처리 가능")
                else:
                    st.success(f"✅ 범주형 데이터")
            
            imb_method = st.selectbox("불균형 증강 방법", ["SMOTE", "RandomOverSampler", "RandomUnderSampler"], key="imb_method_select")
        
        # 노이즈 설정
        st.markdown("**3. 노이즈 설정**")
        
        # 노이즈 레벨 통합 설명
        with st.expander("노이즈 레벨 설명"):
            st.markdown("""
            **권장 설정:**
            - **낮은 노이즈 (0.01~0.05)**: 데이터의 원래 특성을 최대한 유지
            - **중간 노이즈 (0.05~0.1)**: 적절한 다양성 추가
            - **높은 노이즈 (0.1~0.2)**: 강한 다양성 추가 (주의 필요)
            """)
        
        noise_level = st.slider(
            "노이즈 레벨", 
            0.01, 0.2, 0.03, 
            step=0.01, 
            help="수치형 컬럼에 추가할 노이즈의 강도 (모든 증강 방법에서 공통 사용)"
        )
        
        # 증강 비율 설정
        st.markdown("**4. 증강 비율 설정**")
        
        # 통합된 증강 비율
        augmentation_ratio = st.slider(
            "증강 비율", 
            0.1, 2.0, 0.5, 
            step=0.1, 
            help="원본 데이터 대비 증강할 비율 (모든 증강 방법에서 공통 사용)"
        )
        
        # 중복 설정
        dup_count = st.slider(
            "중복 횟수", 
            2, 10, 2, 
            help="전체 데이터를 몇 번 복제할지 설정"
        )
        

        
        # 데이터 삭제 설정
        st.markdown("**5. 데이터 삭제 설정**")
        use_drop = st.checkbox("데이터 삭제 사용", value=False, help="과적합 방지를 위해 일부 데이터를 삭제합니다.")
        drop_rate = None
        if use_drop:
            drop_rate = st.slider("삭제 비율", 0.01, 0.5, 0.1, step=0.01, help="랜덤하게 삭제할 데이터의 비율")
    
    # 기본 증강 방법 설정
    selected_methods = ['noise', 'duplicate', 'feature']
    if use_smote and target_col:
        selected_methods.append('smote')
    if use_drop:
        selected_methods.append('drop')
    selected_methods.append('general')
    
    # 파라미터 딕셔너리 생성
    params = {
        'noise_level': noise_level,
        'dup_count': dup_count,
        'augmentation_ratio': augmentation_ratio
    }
    
    if use_drop and drop_rate is not None:
        params['drop_rate'] = drop_rate
    if use_smote and target_col:
        params['target_col'] = target_col
        params['imb_method'] = imb_method
    
    return params, selected_methods

# 파일 업로드 섹션
uploaded_file = st.sidebar.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file is not None:
    # ===== 데이터 로드 및 검증 =====
    df = DataUtils.load_csv_file(uploaded_file)
    
    if df is not None and DataUtils.validate_data(df):
        
        # ===== 컬럼 타입 분석 =====
        numeric_cols = visualizer.get_numeric_columns(df)
        categorical_cols = visualizer.get_categorical_columns(df)

        # ===== 증강 파라미터 설정 =====
        # 사이드바에서 파라미터 설정 (임시로 직접 정의된 메서드 사용)
        params, selected_methods = setup_augmentation_parameters(categorical_cols, numeric_cols, df)
        
        # ===== 데이터 증강 실행 =====
        df_aug = augmenter._combined_augmentation(df, methods=selected_methods, **params)
        
        # ===== 증강 전후 비교 섹션 =====
        st.markdown("---")
        st.subheader("1. 증강 전후 비교")
        
        # ===== 클래스 분포 비교 (SMOTE 사용 시) =====
        if 'smote' in selected_methods and 'target_col' in params and params['target_col'] in categorical_cols:
            st.markdown("### 1-1. 클래스 분포 비교")
            visualizer.compare_distributions(df, df_aug, params['target_col'])

        # ===== 수치형 데이터 시각화 =====
        if numeric_cols:
            selected_compare = st.selectbox("비교할 수치형 컬럼 선택", numeric_cols, key="compare_select")
            
            # 히스토그램 비교
            st.markdown("### 1-2. 히스토그램 분포 비교")
            fig_overlap_hist = visualizer.create_overlapping_histogram(df, df_aug, selected_compare)
            st.plotly_chart(fig_overlap_hist, use_container_width=True, key="overlap_hist")
            
            # 박스플롯 비교
            st.markdown("### 1-3. 박스플롯 분포 비교")
            fig_overlap_box = visualizer.create_overlapping_boxplot(df, df_aug, selected_compare)
            st.plotly_chart(fig_overlap_box, use_container_width=True, key="overlap_box")
            
            # 통계 요약 (히스토그램과 박스플롯 모두에 대한 요약)
            st.markdown("### 1-4. 통계 요약 (히스토그램 & 박스플롯)")
            visualizer.display_comparison_summary(df, df_aug, numeric_cols)
            
            # ===== 산점도 비교 (수치형 컬럼이 2개 이상인 경우) =====
            if len(numeric_cols) >= 2:
                st.markdown("### 1-5. 산점도 비교")
                x_col_overlap = st.selectbox("X축 컬럼", numeric_cols, key="scatter_x_overlap")
                y_col_overlap = st.selectbox("Y축 컬럼", [col for col in numeric_cols if col != x_col_overlap], key="scatter_y_overlap")
                
                if x_col_overlap and y_col_overlap:
                    fig_overlap_scatter = visualizer.create_overlapping_scatter(df, df_aug, x_col_overlap, y_col_overlap)
                    st.plotly_chart(fig_overlap_scatter, use_container_width=True, key="overlap_scatter")
                    
                    # 산점도 통계 요약 (임시로 직접 정의)
                    st.markdown("### 1-6. 산점도 통계 요약")
                    
                    # 상관계수 계산
                    orig_corr = df[x_col_overlap].corr(df[y_col_overlap])
                    aug_corr = df_aug[x_col_overlap].corr(df_aug[y_col_overlap])
                    corr_change = aug_corr - orig_corr
                    
                    # 데이터 포인트 수
                    orig_points = len(df)
                    aug_points = len(df_aug)
                    points_increase = aug_points - orig_points
                    points_increase_pct = (points_increase / orig_points) * 100
                    
                    # 통계 요약표 생성
                    summary_data = {
                        '지표': ['상관계수', '데이터 포인트'],
                        '원본': [f"{orig_corr:.3f}", f"{orig_points:,}개"],
                        '증강': [f"{aug_corr:.3f}", f"{aug_points:,}개"],
                        '변화량': [f"{corr_change:.3f}", f"{points_increase:,}개"],
                        '변화율': [f"{corr_change/orig_corr*100:.1f}%" if orig_corr != 0 else "N/A", f"{points_increase_pct:.1f}%"]
                    }
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
                    
        
        # ===== 범주형 데이터 비교 =====
        filtered_categorical_cols = DataUtils.filter_categorical_columns(categorical_cols, df)

        if filtered_categorical_cols:
            st.markdown("### 2. 범주형 데이터 비교")
            selected_cat_compare = st.selectbox("비교할 범주형 컬럼 선택", filtered_categorical_cols, key="cat_compare_select")
            
            # 원본 데이터 카운트
            orig_counts = df[selected_cat_compare].value_counts().sort_index()
            aug_counts = df_aug[selected_cat_compare].value_counts().sort_index()
            
            # 모든 카테고리 통합
            all_categories = sorted(set(orig_counts.index) | set(aug_counts.index))
            
            # 데이터프레임 생성
            comparison_data = []
            for cat in all_categories:
                orig_count = orig_counts.get(cat, 0)
                aug_count = aug_counts.get(cat, 0)
                comparison_data.append({
                    '카테고리': cat,
                    '원본': orig_count,
                    '증강': aug_count,
                    '증가량': aug_count - orig_count,
                    '증가율(%)': ((aug_count - orig_count) / orig_count * 100) if orig_count > 0 else float('inf')
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # 겹쳐서 보여주는 막대그래프 생성
            fig_overlap = px.bar(
                comparison_df,
                x='카테고리',
                y=['원본', '증강'],
                title=f'{selected_cat_compare} 분포 비교 (원본 vs 증강)',
                barmode='group',
                color_discrete_map={'원본': '#1f77b4', '증강': '#ff7f0e'}
            )
            
            fig_overlap.update_layout(
                xaxis_title="카테고리",
                yaxis_title="개수",
                legend_title="데이터",
                height=500
            )
            
            st.plotly_chart(fig_overlap, use_container_width=True, key="overlap_cat")
            
            # 통계 요약표
            st.markdown("**통계 요약**")
            st.dataframe(comparison_df, use_container_width=True)
        

        # ===== 증강 결과 리포트 =====
        report_params = params.copy()
        report_params['methods'] = selected_methods
        visualizer.create_augmentation_report(df, df_aug, report_params)

        # ===== 데이터 다운로드 =====
        st.markdown("---")
        st.subheader("증강 데이터 다운로드")
        DataUtils.create_download_button(df_aug, "augmented_csv.csv")
        
else:
    # ===== 초기 안내 메시지 =====
    with st.expander("지원되는 데이터 형식"):
        st.markdown("""
        - **수치형 데이터**: 히스토그램, 박스플롯에 적합
        - **범주형 데이터**: 막대그래프, 파이차트에 적합
        - **CSV 파일 형식**만 지원됩니다
        """)
