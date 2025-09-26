import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="증강 데이터 분석 시각화", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px dashed #dee2e6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📊 증강 데이터 분석 및 시각화 대시보드</h1>', unsafe_allow_html=True)

# ------------------------------
# 1. 데이터 업로드 (원본 + 증강)
# ------------------------------
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.subheader("📁 데이터 업로드")

col1, col2 = st.columns(2)

with col1:
    st.write("**원본 데이터**")
    original_file = st.file_uploader(
        "원본 CSV 또는 Excel 파일", 
        type=["csv", "xlsx"], 
        key="original",
        help="증강 전 원본 데이터 파일을 업로드하세요"
    )

with col2:
    st.write("**증강 데이터**")
    augmented_file = st.file_uploader(
        "증강 CSV 또는 Excel 파일", 
        type=["csv", "xlsx"], 
        key="augmented",
        help="증강 후 데이터 파일을 업로드하세요"
    )
st.markdown('</div>', unsafe_allow_html=True)

# 데이터 로드 함수
def load_data(file):
    """파일을 읽어서 DataFrame으로 반환"""
    if file is None:
        return None
    
    try:
        if file.name.endswith(".csv"):
            # CSV 파일 읽기 - 여러 인코딩 시도
            encodings = ['utf-8', 'cp949', 'euc-kr', 'iso-8859-1', 'latin-1']
            for encoding in encodings:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, encoding=encoding)
                    return df
                except UnicodeDecodeError:
                    continue
            return None
        else:
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"파일 읽기 오류: {str(e)}")
        return None

# 데이터 로드
original_df = load_data(original_file)
augmented_df = load_data(augmented_file)

if original_df is not None and augmented_df is not None:
    st.success("✅ 원본 데이터와 증강 데이터가 모두 업로드되었습니다!")
    
    # 데이터 타입 분석
    numeric_cols = original_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = original_df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # ------------------------------
    # 2. 핵심 지표 카드
    # ------------------------------
    st.markdown("---")
    st.subheader("🎯 핵심 지표")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📈 원본 데이터</h4>
            <h2>{len(original_df):,}</h2>
            <p>행 × {len(original_df.columns):,}열</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🚀 증강 데이터</h4>
            <h2>{len(augmented_df):,}</h2>
            <p>행 × {len(augmented_df.columns):,}열</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        augmentation_ratio = len(augmented_df) / len(original_df)
        growth_rate = ((len(augmented_df) - len(original_df)) / len(original_df)) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 증강 비율</h4>
            <h2>{augmentation_ratio:.2f}x</h2>
            <p>{growth_rate:.1f}% 증가</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        increase_count = len(augmented_df) - len(original_df)
        st.markdown(f"""
        <div class="metric-card">
            <h4>➕ 증가량</h4>
            <h2>{increase_count:,}</h2>
            <p>새로운 데이터</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ------------------------------
    # 3. 시각화 분석 섹션
    # ------------------------------
    st.markdown("---")
    
    # ------------------------------
    # 데이터 미리보기
    # ------------------------------
    st.subheader("👀 데이터 미리보기")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**원본 데이터**")
        st.dataframe(original_df.head(10), use_container_width=True)
        st.info(f"원본 데이터 크기: {original_df.shape[0]:,}행 × {original_df.shape[1]:,}열")
    
    with col2:
        st.write("**증강 데이터**")
        st.dataframe(augmented_df.head(10), use_container_width=True)
        st.info(f"증강 데이터 크기: {augmented_df.shape[0]:,}행 × {augmented_df.shape[1]:,}열")

    # ------------------------------
    # 통계 비교
    # ------------------------------
    st.markdown("---")
    st.subheader("📊 통계 비교")
    
    if numeric_cols:
        # 통계 요약 비교
        original_stats = original_df[numeric_cols].describe()
        augmented_stats = augmented_df[numeric_cols].describe()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**원본 데이터 통계**")
            st.dataframe(original_stats.round(3), use_container_width=True)
        
        with col2:
            st.write("**증강 데이터 통계**")
            st.dataframe(augmented_stats.round(3), use_container_width=True)
    else:
        st.info("수치형 컬럼이 없어 통계 비교를 수행할 수 없습니다.")

    # ------------------------------
    # 분포 비교 시각화
    # ------------------------------
    st.markdown("---")
    st.subheader("🎨 분포 비교 시각화")
    
    if numeric_cols:
        selected_col = st.selectbox("비교할 변수 선택", numeric_cols, key="dist_select")
        
        if selected_col:
            # 겹쳐진 히스토그램
            fig = go.Figure()
            
            # 증강 데이터 히스토그램 (뒤에 배치)
            fig.add_trace(go.Histogram(
                x=augmented_df[selected_col].dropna(),
                name='증강 데이터',
                opacity=0.5,
                marker_color='lightcoral'
            ))
            
            # 원본 데이터 히스토그램 (앞에 배치)
            fig.add_trace(go.Histogram(
                x=original_df[selected_col].dropna(),
                name='원본 데이터',
                opacity=0.8,
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title=f'{selected_col} 분포 비교',
                xaxis_title=selected_col,
                yaxis_title='빈도',
                barmode='overlay',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 박스플롯 비교
            fig_box = go.Figure()
            
            # 증강 데이터 박스플롯 (뒤에 배치)
            fig_box.add_trace(go.Box(
                y=augmented_df[selected_col].dropna(),
                name='증강 데이터',
                marker_color='lightcoral',
                opacity=0.7
            ))
            
            # 원본 데이터 박스플롯 (앞에 배치)
            fig_box.add_trace(go.Box(
                y=original_df[selected_col].dropna(),
                name='원본 데이터',
                marker_color='lightblue',
                opacity=0.9
            ))
            
            fig_box.update_layout(
                title=f'{selected_col} 박스플롯 비교',
                yaxis_title=selected_col,
                height=400
            )
            
            st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("수치형 컬럼이 없어 분포 비교를 수행할 수 없습니다.")

    # ------------------------------
    # 상관관계 분석
    # ------------------------------
    st.markdown("---")
    st.subheader("🔗 상관관계 분석")
    
    if len(numeric_cols) > 1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**원본 데이터 상관관계**")
            original_corr = original_df[numeric_cols].corr()
            fig_orig = px.imshow(
                original_corr,
                text_auto=True,
                aspect="auto",
                title="원본 데이터 상관관계",
                color_continuous_scale='RdBu_r'
            )
            fig_orig.update_layout(height=400)
            st.plotly_chart(fig_orig, use_container_width=True)
        
        with col2:
            st.write("**증강 데이터 상관관계**")
            augmented_corr = augmented_df[numeric_cols].corr()
            fig_aug = px.imshow(
                augmented_corr,
                text_auto=True,
                aspect="auto",
                title="증강 데이터 상관관계",
                color_continuous_scale='RdBu_r'
            )
            fig_aug.update_layout(height=400)
            st.plotly_chart(fig_aug, use_container_width=True)
    else:
        st.info("상관관계 분석을 위해서는 최소 2개의 수치형 컬럼이 필요합니다.")

    # ------------------------------
    # 맞춤 시각화
    # ------------------------------
    st.markdown("---")
    st.subheader("🎯 맞춤 시각화")
    
    # 두 개의 컬럼으로 차트 유형 분리
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📊 산점도 & 라인 차트**")
        
        if len(numeric_cols) >= 2:
            x_axis = st.selectbox("X축 선택", numeric_cols, key="custom_x")
            y_axis = st.selectbox("Y축 선택", [col for col in numeric_cols if col != x_axis], key="custom_y")
            
            # 산점도
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=augmented_df[x_axis],
                y=augmented_df[y_axis],
                mode='markers',
                name='증강 데이터',
                marker=dict(color='lightcoral', opacity=0.5, size=6)
            ))
            fig_scatter.add_trace(go.Scatter(
                x=original_df[x_axis],
                y=original_df[y_axis],
                mode='markers',
                name='원본 데이터',
                marker=dict(color='lightblue', opacity=0.8, size=8)
            ))
            fig_scatter.update_layout(
                title=f"산점도: {x_axis} vs {y_axis}",
                xaxis_title=x_axis,
                yaxis_title=y_axis,
                height=400
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # 라인 차트
            fig_line = go.Figure()
            aug_sorted = augmented_df.sort_values(x_axis)
            fig_line.add_trace(go.Scatter(
                x=aug_sorted[x_axis],
                y=aug_sorted[y_axis],
                mode='lines+markers',
                name='증강 데이터',
                line=dict(color='lightcoral', width=2),
                marker=dict(size=6, opacity=0.7)
            ))
            orig_sorted = original_df.sort_values(x_axis)
            fig_line.add_trace(go.Scatter(
                x=orig_sorted[x_axis],
                y=orig_sorted[y_axis],
                mode='lines+markers',
                name='원본 데이터',
                line=dict(color='lightblue', width=3),
                marker=dict(size=8, opacity=0.9)
            ))
            fig_line.update_layout(
                title=f"라인 차트: {x_axis} vs {y_axis}",
                xaxis_title=x_axis,
                yaxis_title=y_axis,
                height=400
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.warning("산점도와 라인 차트를 위해서는 최소 2개의 수치형 컬럼이 필요합니다.")
    
    with col2:
        st.write("**📈 막대그래프 & 박스플롯**")
        
        # 막대그래프
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            cat_col = st.selectbox("범주형 컬럼 선택", categorical_cols, key="bar_cat")
            num_col = st.selectbox("수치형 컬럼 선택", numeric_cols, key="bar_num")
            
            # 원본 데이터 막대그래프
            orig_grouped = original_df.groupby(cat_col)[num_col].mean().reset_index()
            fig_orig = px.bar(
                orig_grouped, 
                x=cat_col, 
                y=num_col,
                title=f"원본 데이터 - {cat_col}별 {num_col} 평균"
            )
            fig_orig.update_layout(height=300)
            st.plotly_chart(fig_orig, use_container_width=True)
            
            # 증강 데이터 막대그래프
            aug_grouped = augmented_df.groupby(cat_col)[num_col].mean().reset_index()
            fig_aug = px.bar(
                aug_grouped, 
                x=cat_col, 
                y=num_col,
                title=f"증강 데이터 - {cat_col}별 {num_col} 평균"
            )
            fig_aug.update_layout(height=300)
            st.plotly_chart(fig_aug, use_container_width=True)
        else:
            st.warning("막대그래프를 위해서는 범주형과 수치형 컬럼이 각각 최소 1개씩 필요합니다.")
        
        # 박스플롯
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("박스플롯 컬럼 선택", numeric_cols, key="box_col")
            
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(
                y=augmented_df[selected_col].dropna(),
                name='증강 데이터',
                marker_color='lightcoral',
                opacity=0.7
            ))
            fig_box.add_trace(go.Box(
                y=original_df[selected_col].dropna(),
                name='원본 데이터',
                marker_color='lightblue',
                opacity=0.9
            ))
            fig_box.update_layout(
                title=f'{selected_col} 박스플롯 비교',
                yaxis_title=selected_col,
                height=400
            )
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning("박스플롯을 위해서는 최소 1개의 수치형 컬럼이 필요합니다.")

elif original_df is not None:
    st.warning("⚠️ 증강 데이터를 업로드해주세요.")
elif augmented_df is not None:
    st.warning("⚠️ 원본 데이터를 업로드해주세요.")
else:
    st.info("📁 원본 데이터와 증강 데이터를 모두 업로드하세요.")
    
    # 사용 가이드
    with st.expander("📋 사용 가이드", expanded=True):
        st.markdown("""
        ### 🚀 증강 데이터 분석 대시보드 사용법
        
        **1. 데이터 업로드**
        - 원본 데이터와 증강 데이터를 각각 업로드하세요
        - CSV 또는 Excel 파일 형식을 지원합니다
        
        **2. 분석 기능**
        - **핵심 지표**: 데이터 크기, 증강 비율 등 주요 메트릭
        - **통계 비교**: 원본과 증강 데이터의 통계적 특성 비교
        - **분포 비교**: 히스토그램과 박스플롯으로 분포 변화 분석
        - **상관관계 분석**: 변수 간 상관관계 변화 분석
        - **맞춤 시각화**: 사용자가 선택한 차트로 데이터 비교
        
        **3. 시각화 유형**
        - **산점도**: 두 수치형 변수 간의 관계 비교
        - **라인 차트**: 시간에 따른 변화 추이 분석
        - **막대그래프**: 범주별 수치 비교
        - **박스플롯**: 분포와 이상치 비교
        
        **4. 해석 가이드**
        - 양수 변화량: 증강 후 증가
        - 음수 변화량: 증강 후 감소
        - 상관관계 변화: 변수 간 관계의 변화
        """)
