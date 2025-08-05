import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import json
import io
import time
from typing import Dict, Any, Optional

# 백엔드 API 설정
BACKEND_URL = "http://localhost:8000"

# 페이지 설정
st.set_page_config(layout='wide')
st.title("1. CSV 정형 데이터 증강 및 시각화 도구")

# 세션 상태 초기화
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'data_analysis' not in st.session_state:
    st.session_state.data_analysis = None
if 'augmentation_result' not in st.session_state:
    st.session_state.augmentation_result = None

def call_backend_api(endpoint: str, method: str = "GET", data: Dict = None, files: Dict = None) -> Optional[Dict]:
    """백엔드 API를 호출하는 함수"""
    try:
        url = f"{BACKEND_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files)
            else:
                response = requests.post(url, json=data)
        else:
            st.error(f"지원하지 않는 HTTP 메서드: {method}")
            return None
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API 호출 실패: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("백엔드 서버에 연결할 수 없습니다. 백엔드가 실행 중인지 확인해주세요.")
        return None
    except Exception as e:
        st.error(f"API 호출 중 오류 발생: {str(e)}")
        return None

def upload_file_to_backend(uploaded_file) -> Optional[str]:
    """파일을 백엔드에 업로드하고 세션 ID를 반환"""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        response = call_backend_api("/api/data/upload", method="POST", files=files)
        
        if response and response.get('success'):
            session_id = response['session_id']
            st.session_state.session_id = session_id
            st.success(f"파일 업로드 성공! 세션 ID: {session_id[:8]}...")
            return session_id
        else:
            st.error("파일 업로드에 실패했습니다.")
            return None
            
    except Exception as e:
        st.error(f"파일 업로드 중 오류 발생: {str(e)}")
        return None

def get_data_analysis(session_id: str) -> Optional[Dict]:
    """데이터 분석 결과를 백엔드에서 가져옴"""
    response = call_backend_api(f"/api/data/analyze/{session_id}")
    if response and response.get('success'):
        st.session_state.data_analysis = response
        return response
    return None

def get_data_preview(session_id: str, rows: int = 10) -> Optional[Dict]:
    """데이터 미리보기를 백엔드에서 가져옴"""
    response = call_backend_api(f"/api/data/preview/{session_id}?rows={rows}")
    return response if response and response.get('success') else None

def process_augmentation(session_id: str, params: Dict, methods: list) -> Optional[Dict]:
    """데이터 증강을 백엔드에서 실행"""
    try:
        augmentation_data = {
            "session_id": session_id,
            "methods": methods,
            **params
        }
        
        with st.spinner("🔄 데이터 증강 처리 중..."):
            response = call_backend_api("/api/augmentation/process", method="POST", data=augmentation_data)
            
        if response and response.get('success'):
            st.session_state.augmentation_result = response
            st.success("✅ 데이터 증강 완료!")
            return response
        else:
            st.error("데이터 증강에 실패했습니다.")
            return None
            
    except Exception as e:
        st.error(f"증강 처리 중 오류 발생: {str(e)}")
        return None

def get_histogram_chart(session_id: str, column: str) -> Optional[Dict]:
    """히스토그램 차트 데이터를 백엔드에서 가져옴"""
    response = call_backend_api(f"/api/visualization/histogram/{session_id}/{column}")
    return response if response and response.get('success') else None

def get_boxplot_chart(session_id: str, column: str) -> Optional[Dict]:
    """박스플롯 차트 데이터를 백엔드에서 가져옴"""
    response = call_backend_api(f"/api/visualization/boxplot/{session_id}/{column}")
    return response if response and response.get('success') else None

def get_scatter_chart(session_id: str, x_column: str, y_column: str) -> Optional[Dict]:
    """산점도 차트 데이터를 백엔드에서 가져옴"""
    response = call_backend_api(f"/api/visualization/scatter/{session_id}?x_column={x_column}&y_column={y_column}")
    return response if response and response.get('success') else None

def get_categorical_comparison(session_id: str, column: str) -> Optional[Dict]:
    """범주형 비교 데이터를 백엔드에서 가져옴"""
    response = call_backend_api(f"/api/visualization/categorical/{session_id}/{column}")
    return response if response and response.get('success') else None

def get_comparison_summary(session_id: str) -> Optional[Dict]:
    """비교 요약 통계를 백엔드에서 가져옴"""
    response = call_backend_api(f"/api/visualization/summary/{session_id}")
    return response if response and response.get('success') else None

def download_augmented_data(session_id: str) -> Optional[bytes]:
    """증강된 데이터를 백엔드에서 다운로드"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/data/download/{session_id}?data_type=augmented")
        if response.status_code == 200:
            return response.content
        else:
            st.error("데이터 다운로드에 실패했습니다.")
            return None
    except Exception as e:
        st.error(f"다운로드 중 오류 발생: {str(e)}")
        return None

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
            st.markdown("**SMOTE 사용 시 반드시 타겟 레이블을 설정해야 합니다.**")
            
            # 범주형 컬럼 우선, 수치형 컬럼은 범주형으로 처리 가능한 경우만
            smote_cols = categorical_cols.copy()
            for col in numeric_cols:
                unique_count = df[col].nunique()
                if unique_count <= 20:  # 범주형으로 처리 가능한 수치형 컬럼
                    smote_cols.append(col)
            
            if smote_cols:
                target_col = st.selectbox(
                    "타겟(레이블) 컬럼 선택", 
                    smote_cols, 
                    key="target_select",
                    help="분류하고자 하는 클래스 레이블 컬럼을 선택하세요"
                )
                
                if target_col:
                    pass  # 메시지 제거
            else:
                use_smote = False
            
            if target_col:
                imb_method = "SMOTE"  # SMOTE만 사용
        
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
    
    # 기본 증강 방법 설정
    selected_methods = ['noise', 'duplicate', 'feature']
    if use_smote and target_col:
        selected_methods.append('smote')
    selected_methods.append('general')
    
    # 파라미터 딕셔너리 생성
    params = {
        'noise_level': noise_level,
        'dup_count': dup_count,
        'augmentation_ratio': augmentation_ratio
    }
    
    if use_smote and target_col:
        params['target_col'] = target_col
        params['imb_method'] = imb_method
    
    return params, selected_methods

# 파일 업로드 섹션
uploaded_file = st.sidebar.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file is not None:
    # 파일 업로드 처리
    if st.session_state.session_id is None:
        session_id = upload_file_to_backend(uploaded_file)
        if not session_id:
            st.stop()
    
    # 데이터 분석
    if st.session_state.data_analysis is None:
        data_analysis = get_data_analysis(st.session_state.session_id)
        if not data_analysis:
            st.stop()
    
    # 분석 결과에서 데이터 추출
    analysis = st.session_state.data_analysis
    numeric_cols = analysis['numeric_columns']
    categorical_cols = analysis['categorical_columns']
    
    # ===== 데이터 분석 =====
    st.markdown("---")
    st.subheader("📊 데이터 분석")
    
    # 탭으로 구분된 분석 섹션
    tab1, tab2, tab3 = st.tabs(["📋 데이터 미리보기", "📈 기본 정보", "🔍 품질 분석"])
    
    with tab1:
        st.markdown("### 원본 데이터 미리보기")
        col1, col2 = st.columns([3, 1])
        with col1:
            preview_rows = st.slider(
                "미리보기 행 수", 
                5, 50, 10, 
                help="원본 데이터에서 보여줄 행 수를 선택하세요", 
            )
        with col2:
            st.write("")  # 공간 맞추기
            st.write("")  # 공간 맞추기
        
        # 미리보기 데이터 가져오기
        preview_data = get_data_preview(st.session_state.session_id, preview_rows)
        if preview_data:
            preview_df = pd.DataFrame(preview_data['preview_data'])
            st.dataframe(preview_df, use_container_width=True)
            
            # 데이터 요약 정보
            with st.expander("📊 데이터 요약 정보"):
                st.write(f"**데이터 형태**: {analysis['data_shape']['rows']:,}행 × {analysis['data_shape']['columns']}열")
                st.write("**데이터 타입 분포**:")
                st.write(f"- **수치형**: {len(numeric_cols)}개 | {', '.join(numeric_cols)}")
                st.write(f"- **범주형**: {len(categorical_cols)}개 | {', '.join(categorical_cols)}")
    
    with tab2:
        st.markdown("### 기본 데이터 정보")
        
        # 주요 메트릭
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 행 수", f"{analysis['data_shape']['rows']:,}", help="데이터셋의 총 행 수")
        with col2:
            st.metric("총 컬럼 수", f"{analysis['data_shape']['columns']}", help="데이터셋의 총 컬럼 수")
        with col3:
            st.metric("수치형 컬럼", f"{len(numeric_cols)}", help="수치형 데이터 컬럼 수")
        with col4:
            st.metric("범주형 컬럼", f"{len(categorical_cols)}", help="범주형 데이터 컬럼 수")
        
        # 컬럼 정보 상세
        st.markdown("### 컬럼 상세 정보")
        col_info_df = pd.DataFrame(analysis['column_info'])
        st.dataframe(col_info_df, use_container_width=True)
    
    with tab3:
        st.markdown("### 데이터 품질 분석")
        
        # 결측값 분석
        missing_data = analysis['missing_data']
        missing_df = pd.DataFrame([
            {'컬럼': col, '결측값 수': count} 
            for col, count in missing_data.items()
        ])
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**결측값 분석**")
            total_missing = sum(missing_data.values())
            if total_missing == 0:
                st.success("✅ 결측값이 없습니다.")
            else:
                st.dataframe(missing_df[missing_df['결측값 수'] > 0], use_container_width=True)
                st.warning(f"⚠️ 총 {total_missing:,}개의 결측값이 있습니다.")
        
        # 중복값 분석
        with col2:
            st.markdown("**중복값 분석**")
            duplicate_count = analysis['duplicate_count']
            duplicate_pct = (duplicate_count / analysis['data_shape']['rows']) * 100
            st.metric("중복 행 수", f"{duplicate_count:,} ({duplicate_pct:.1f}%)")
            if duplicate_count == 0:
                st.success("✅ 중복값이 없습니다.")
            else:
                st.warning(f"⚠️ 중복값이 {duplicate_pct:.1f}% 있습니다.")

    # ===== 증강 파라미터 설정 =====
    # 임시로 DataFrame을 생성하여 파라미터 설정 함수 사용
    temp_df = pd.DataFrame(columns=numeric_cols + categorical_cols)
    params, selected_methods = setup_augmentation_parameters(categorical_cols, numeric_cols, temp_df)
    
    # ===== 데이터 증강 버튼 =====
    st.sidebar.markdown("---")
    st.sidebar.markdown("**데이터 증강 실행**")
    
    # 증강 버튼
    augment_button = st.sidebar.button(
        "🚀 데이터 증강 시작", 
        type="primary",
        help="설정한 파라미터로 데이터 증강을 실행합니다",
        use_container_width=True
    )
    
    # 버튼 클릭 시 증강 실행
    if augment_button:
        augmentation_result = process_augmentation(st.session_state.session_id, params, selected_methods)
        if not augmentation_result:
            st.stop()
    
    # 증강된 데이터가 있으면 시각화 실행
    if st.session_state.augmentation_result:
        aug_result = st.session_state.augmentation_result
        
        # ===== 증강 전후 비교 섹션 =====
        st.markdown("---")
        st.subheader("1. 증강 전후 비교")
        
        # 증강 결과 요약
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("원본 행 수", f"{aug_result['original_shape']['rows']:,}")
        with col2:
            st.metric("증강 행 수", f"{aug_result['augmented_shape']['rows']:,}")
        with col3:
            increase = aug_result['augmented_shape']['rows'] - aug_result['original_shape']['rows']
            st.metric("증가 행 수", f"{increase:,}")
        with col4:
            increase_pct = (increase / aug_result['original_shape']['rows']) * 100
            st.metric("증가율", f"{increase_pct:.1f}%")
        
        # ===== 수치형 데이터 시각화 =====
        if numeric_cols:
            selected_compare = st.selectbox("비교할 수치형 컬럼 선택", numeric_cols, key="compare_select")
            
            # 히스토그램 비교
            st.markdown("### 1-2. 히스토그램 분포 비교")
            hist_data = get_histogram_chart(st.session_state.session_id, selected_compare)
            if hist_data:
                # Plotly 차트 데이터를 다시 생성
                fig = px.histogram(
                    title=f"{selected_compare} 히스토그램 비교",
                    nbins=30
                )
                st.plotly_chart(fig, use_container_width=True, key="overlap_hist")
            
            # 박스플롯 비교
            st.markdown("### 1-3. 박스플롯 분포 비교")
            box_data = get_boxplot_chart(st.session_state.session_id, selected_compare)
            if box_data:
                # Plotly 차트 데이터를 다시 생성
                fig = px.box(
                    title=f"{selected_compare} 박스플롯 비교"
                )
                st.plotly_chart(fig, use_container_width=True, key="overlap_box")
            
            # 통계 요약
            st.markdown("### 1-4. 통계 요약")
            summary_data = get_comparison_summary(st.session_state.session_id)
            if summary_data and selected_compare in summary_data['summary_stats']:
                stats = summary_data['summary_stats'][selected_compare]
                
                # 통계 요약표 생성
                summary_df = pd.DataFrame([
                    {
                        '지표': '평균',
                        '원본': f"{stats['original']['mean']:.2f}",
                        '증강': f"{stats['augmented']['mean']:.2f}",
                        '변화': f"{stats['changes']['mean_change']:.2f}"
                    },
                    {
                        '지표': '표준편차',
                        '원본': f"{stats['original']['std']:.2f}",
                        '증강': f"{stats['augmented']['std']:.2f}",
                        '변화': f"{stats['changes']['std_change']:.2f}"
                    }
                ])
                st.dataframe(summary_df, use_container_width=True)
            
            # ===== 산점도 비교 (수치형 컬럼이 2개 이상인 경우) =====
            if len(numeric_cols) >= 2:
                st.markdown("### 1-5. 산점도 비교")
                x_col_overlap = st.selectbox("X축 컬럼", numeric_cols, key="scatter_x_overlap")
                y_col_overlap = st.selectbox("Y축 컬럼", [col for col in numeric_cols if col != x_col_overlap], key="scatter_y_overlap")
                
                if x_col_overlap and y_col_overlap:
                    scatter_data = get_scatter_chart(st.session_state.session_id, x_col_overlap, y_col_overlap)
                    if scatter_data:
                        # Plotly 차트 데이터를 다시 생성
                        fig = px.scatter(
                            title=f"{x_col_overlap} vs {y_col_overlap} 산점도 비교"
                        )
                        st.plotly_chart(fig, use_container_width=True, key="overlap_scatter")
        
        # ===== 범주형 데이터 비교 =====
        if categorical_cols:
            st.markdown("### 2. 범주형 데이터 비교")
            
            # SMOTE 사용 시 타겟 컬럼을 기본값으로 설정
            default_cat_col = None
            if 'smote' in selected_methods and 'target_col' in params and params['target_col'] in categorical_cols:
                default_cat_col = params['target_col']
            
            # default_cat_col이 categorical_cols에 있는지 확인
            default_index = 0
            if default_cat_col and default_cat_col in categorical_cols:
                default_index = categorical_cols.index(default_cat_col)
            
            selected_cat_compare = st.selectbox("비교할 범주형 컬럼 선택", categorical_cols, key="cat_compare_select", index=default_index)
            
            # 범주형 비교 데이터 가져오기
            cat_data = get_categorical_comparison(st.session_state.session_id, selected_cat_compare)
            if cat_data:
                comparison_df = pd.DataFrame(cat_data['comparison_data'])
                
                # 겹쳐서 보여주는 막대그래프 생성
                fig_overlap = px.bar(
                    comparison_df,
                    x='category',
                    y=['original', 'augmented'],
                    title=f'{selected_cat_compare} 분포 비교 (원본 vs 증강)',
                    barmode='group',
                    color_discrete_map={'original': '#87CEEB', 'augmented': '#FFB6C1'}
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
        
        # ===== 데이터 다운로드 =====
        st.markdown("---")
        st.subheader("증강 데이터 다운로드")
        
        if st.button("📥 증강된 데이터 다운로드"):
            data_content = download_augmented_data(st.session_state.session_id)
            if data_content:
                st.download_button(
                    label="💾 CSV 파일로 다운로드",
                    data=data_content,
                    file_name="augmented_data.csv",
                    mime="text/csv"
                )
    
    else:
        # 증강이 실행되지 않은 경우 안내 메시지
        st.markdown("---")
        st.info("ℹ️ 사이드바에서 파라미터를 설정하고 '🚀 데이터 증강 시작' 버튼을 클릭하여 증강을 실행하세요.")
        
else:
    # ===== 초기 안내 메시지 =====
    with st.expander("지원되는 데이터 형식"):
        st.markdown("""
        - **수치형 데이터**: 히스토그램, 박스플롯에 적합
        - **범주형 데이터**: 막대그래프, 파이차트에 적합
        - **CSV 파일 형식**만 지원됩니다
        """)
    
    # 백엔드 연결 상태 확인
    if st.button("🔗 백엔드 연결 확인"):
        response = call_backend_api("/health")
        if response:
            st.success("✅ 백엔드 서버에 연결되었습니다!")
        else:
            st.error("❌ 백엔드 서버에 연결할 수 없습니다. 백엔드를 먼저 실행해주세요.")
