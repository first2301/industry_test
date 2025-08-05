import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import io
import time
from typing import Dict, Any, Optional, List

# 백엔드 API 설정
BACKEND_URL = "http://localhost:8000"

# 페이지 설정
st.set_page_config(
    page_title="데이터 시각화 및 증강 도구",
    page_icon="📊",
    layout='wide',
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'augmented_data' not in st.session_state:
    st.session_state.augmented_data = None
if 'column_types' not in st.session_state:
    st.session_state.column_types = None

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

def upload_data_to_backend(uploaded_file) -> Optional[Dict]:
    """파일을 백엔드에 업로드하고 데이터 정보를 반환"""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        response = call_backend_api("/visualization/upload-data", method="POST", files=files)
        
        if response and response.get('success'):
            st.success("파일 업로드 성공!")
            return response
        else:
            st.error("파일 업로드에 실패했습니다.")
            return None
            
    except Exception as e:
        st.error(f"파일 업로드 중 오류 발생: {str(e)}")
        return None

def get_column_types_from_backend(data: List[Dict]) -> Optional[Dict]:
    """백엔드에서 컬럼 타입 분석 결과를 가져옴"""
    response = call_backend_api("/visualization/get-column-types", method="POST", data=data)
    if response and response.get('success'):
        return response
    return None

def get_augmentation_methods() -> Optional[List[str]]:
    """백엔드에서 사용 가능한 증강 방법 목록을 가져옴"""
    response = call_backend_api("/augmentation/methods")
    if response and response.get('success'):
        return response.get('available_methods', [])
    return []

def augment_data(data: List[Dict], method: str, parameters: Dict) -> Optional[Dict]:
    """백엔드에서 데이터 증강을 수행"""
    augmentation_data = {
        "data": data,
        "method": method,
        "parameters": parameters
    }
    
    with st.spinner("🔄 데이터 증강 처리 중..."):
        response = call_backend_api("/augmentation/augment", method="POST", data=augmentation_data)
    
    if response and response.get('success'):
        return response.get('augmentation_result')
    return None

def create_visualization_chart(chart_type: str, original_data: List[Dict], 
                             augmented_data: List[Dict], **kwargs) -> Optional[Dict]:
    """백엔드에서 시각화 차트를 생성"""
    if chart_type == "histogram":
        data = {
            "original_data": original_data,
            "augmented_data": augmented_data,
            "column": kwargs.get('column')
        }
        response = call_backend_api("/visualization/create-histogram-comparison", method="POST", data=data)
    elif chart_type == "boxplot":
        data = {
            "original_data": original_data,
            "augmented_data": augmented_data,
            "column": kwargs.get('column')
        }
        response = call_backend_api("/visualization/create-boxplot-comparison", method="POST", data=data)
    elif chart_type == "scatter":
        data = {
            "original_data": original_data,
            "augmented_data": augmented_data,
            "x_column": kwargs.get('x_column'),
            "y_column": kwargs.get('y_column')
        }
        response = call_backend_api("/visualization/create-scatter-comparison", method="POST", data=data)
    elif chart_type == "categorical":
        data = {
            "data": original_data,
            "column": kwargs.get('column'),
            "chart_type": kwargs.get('chart_style', '막대그래프')
        }
        response = call_backend_api("/visualization/create-categorical-chart", method="POST", data=data)
    else:
        st.error(f"지원하지 않는 차트 타입: {chart_type}")
        return None
    
    if response and response.get('success'):
        return response.get('result')
    return None

def display_plotly_chart(chart_data: Dict):
    """Plotly 차트 데이터를 표시"""
    if not chart_data:
        st.error("차트 데이터가 없습니다.")
        return
    
    try:
        # Plotly Figure 객체로 변환
        fig = go.Figure(chart_data.get('figure', {}))
        
        # 차트 표시
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"차트 표시 중 오류 발생: {str(e)}")

def setup_augmentation_parameters(numeric_cols: List[str], categorical_cols: List[str]) -> tuple:
    """사이드바에서 증강 파라미터를 설정"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("**🔧 증강 파라미터 설정**")
        
        # 증강 방법 선택
        available_methods = get_augmentation_methods()
        if available_methods:
            selected_method = st.selectbox(
                "증강 방법 선택",
                available_methods,
                help="사용할 데이터 증강 방법을 선택하세요"
            )
        else:
            selected_method = "조합 증강"
            st.warning("백엔드에서 증강 방법을 가져올 수 없습니다. 기본값을 사용합니다.")
        
        # 노이즈 설정
        st.markdown("**📊 노이즈 설정**")
        noise_level = st.slider(
            "노이즈 레벨", 
            0.01, 0.2, 0.05, 
            step=0.01, 
            help="수치형 컬럼에 추가할 노이즈의 강도"
        )
        
        # 증강 비율 설정
        st.markdown("**📈 증강 비율 설정**")
        augmentation_ratio = st.slider(
            "증강 비율", 
            0.1, 2.0, 0.5, 
            step=0.1, 
            help="원본 데이터 대비 증강할 비율"
        )
        
        # 중복 설정
        dup_count = st.slider(
            "중복 횟수", 
            2, 10, 2, 
            help="전체 데이터를 몇 번 복제할지 설정"
        )
        
        # SMOTE 설정
        st.markdown("**⚖️ SMOTE 설정**")
        use_smote = st.checkbox("SMOTE 사용", value=False, help="불균형 데이터 증강을 위해 SMOTE를 사용합니다.")
        
        target_col = None
        if use_smote and categorical_cols:
            target_col = st.selectbox(
                "타겟(레이블) 컬럼 선택", 
                categorical_cols, 
                help="분류하고자 하는 클래스 레이블 컬럼을 선택하세요"
            )
    
    # 파라미터 딕셔너리 생성
    parameters = {
        'noise_level': noise_level,
        'dup_count': dup_count,
        'augmentation_ratio': augmentation_ratio
    }
    
    if use_smote and target_col:
        parameters['target_col'] = target_col
        parameters['imb_method'] = 'SMOTE'
    
    return selected_method, parameters

# 메인 애플리케이션
st.title("📊 데이터 시각화 및 증강 도구")
st.markdown("---")

# 사이드바 - 파일 업로드
st.sidebar.title("📁 데이터 업로드")
uploaded_file = st.sidebar.file_uploader(
    "CSV 파일을 업로드하세요", 
    type=["csv"],
    help="분석할 CSV 파일을 선택하세요"
)

if uploaded_file is not None:
    # 파일 업로드 처리
    if st.session_state.original_data is None:
        upload_result = upload_data_to_backend(uploaded_file)
        if upload_result:
            st.session_state.original_data = upload_result.get('data', [])
            st.session_state.column_types = upload_result.get('data_info', {})
    
    # 데이터 정보 표시
    if st.session_state.original_data:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("행 수", len(st.session_state.original_data))
        with col2:
            st.metric("열 수", st.session_state.column_types.get('columns', 0))
        with col3:
            st.metric("파일명", uploaded_file.name)
        with col4:
            st.metric("상태", "✅ 업로드 완료")
        
        # 컬럼 타입 분석
        if st.session_state.column_types:
            numeric_cols = st.session_state.column_types.get('numeric_columns', [])
            categorical_cols = st.session_state.column_types.get('categorical_columns', [])
            
            # 증강 파라미터 설정
            selected_method, parameters = setup_augmentation_parameters(numeric_cols, categorical_cols)
            
            # 증강 실행 버튼
            if st.sidebar.button("🚀 데이터 증강 실행", type="primary"):
                if st.session_state.original_data:
                    augmented_result = augment_data(
                        st.session_state.original_data, 
                        selected_method, 
                        parameters
                    )
                    
                    if augmented_result:
                        st.session_state.augmented_data = augmented_result.get('augmented_data', [])
                        st.success("데이터 증강이 완료되었습니다!")
                        
                        # 증강 통계 표시
                        stats = augmented_result.get('augmentation_stats', {})
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("원본 행 수", stats.get('original_rows', 0))
                        with col2:
                            st.metric("증강 후 행 수", stats.get('augmented_rows', 0))
                        with col3:
                            st.metric("증가율", f"{stats.get('row_increase_ratio', 0):.1f}%")
        
        # 데이터 미리보기
        st.markdown("### 📋 데이터 미리보기")
        if st.session_state.original_data:
            df_original = pd.DataFrame(st.session_state.original_data)
            st.dataframe(df_original.head(10), use_container_width=True)
        
        # 시각화 섹션
        if st.session_state.augmented_data:
            st.markdown("---")
            st.markdown("### 📊 시각화")
            
            # 탭으로 시각화 종류 선택
            viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(["히스토그램", "박스플롯", "산점도", "범주형"])
            
            with viz_tab1:
                st.markdown("#### 히스토그램 비교")
                if numeric_cols:
                    selected_col = st.selectbox("컬럼 선택", numeric_cols, key="hist_col")
                    if st.button("히스토그램 생성", key="hist_btn"):
                        chart_data = create_visualization_chart(
                            "histogram",
                            st.session_state.original_data,
                            st.session_state.augmented_data,
                            column=selected_col
                        )
                        display_plotly_chart(chart_data)
            
            with viz_tab2:
                st.markdown("#### 박스플롯 비교")
                if numeric_cols:
                    selected_col = st.selectbox("컬럼 선택", numeric_cols, key="box_col")
                    if st.button("박스플롯 생성", key="box_btn"):
                        chart_data = create_visualization_chart(
                            "boxplot",
                            st.session_state.original_data,
                            st.session_state.augmented_data,
                            column=selected_col
                        )
                        display_plotly_chart(chart_data)
            
            with viz_tab3:
                st.markdown("#### 산점도 비교")
                if len(numeric_cols) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        x_col = st.selectbox("X축 컬럼", numeric_cols, key="scatter_x")
                    with col2:
                        y_col = st.selectbox("Y축 컬럼", numeric_cols, key="scatter_y")
                    
                    if st.button("산점도 생성", key="scatter_btn"):
                        chart_data = create_visualization_chart(
                            "scatter",
                            st.session_state.original_data,
                            st.session_state.augmented_data,
                            x_column=x_col,
                            y_column=y_col
                        )
                        display_plotly_chart(chart_data)
            
            with viz_tab4:
                st.markdown("#### 범주형 차트")
                if categorical_cols:
                    selected_col = st.selectbox("컬럼 선택", categorical_cols, key="cat_col")
                    chart_style = st.selectbox("차트 스타일", ["막대그래프", "파이차트"], key="cat_style")
                    
                    if st.button("범주형 차트 생성", key="cat_btn"):
                        chart_data = create_visualization_chart(
                            "categorical",
                            st.session_state.original_data,
                            st.session_state.augmented_data,
                            column=selected_col,
                            chart_style=chart_style
                        )
                        display_plotly_chart(chart_data)
            
            # 증강된 데이터 다운로드
            st.markdown("---")
            st.markdown("### 💾 증강된 데이터 다운로드")
            if st.button("📥 증강된 데이터 다운로드"):
                if st.session_state.augmented_data:
                    df_augmented = pd.DataFrame(st.session_state.augmented_data)
                    csv = df_augmented.to_csv(index=False)
                    st.download_button(
                        label="CSV 파일 다운로드",
                        data=csv,
                        file_name=f"augmented_{uploaded_file.name}",
                        mime="text/csv"
                    )

else:
    st.info("👈 사이드바에서 CSV 파일을 업로드해주세요.")
    
    # 백엔드 연결 상태 확인
    st.markdown("---")
    st.markdown("### 🔗 백엔드 연결 상태")
    health_response = call_backend_api("/health")
    if health_response:
        st.success("✅ 백엔드 서버에 연결되었습니다.")
    else:
        st.error("❌ 백엔드 서버에 연결할 수 없습니다.")
        st.info("백엔드 서버를 실행하려면: `cd structure_vis/backend && python main.py`")

# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>데이터 시각화 및 증강 도구 | FastAPI + Streamlit</p>
</div>
""", unsafe_allow_html=True) 