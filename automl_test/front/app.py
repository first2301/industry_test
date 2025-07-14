# 대시보드 UI
# https://docs.streamlit.io/knowledge-base/deploy/increase-file-uploader-limit-streamlit-cloud
# https://docs.streamlit.io/library/advanced-features/configuration#set-configuration-options
# 실행방법1: streamlit run recomendation.py
# 실행방법2: streamlit run recomendation.py --server.maxUploadSize 500 --server.maxMessageSize 500 (업로드 파일 용량 증대할 경우)
# import time # 코드 실행 시간 측정 시 사용
# sqlite:///./database/database.db
# AxiosError: Request failed with status code 403 in streamlit 발생 시, enableXsrfProtection 입력하여 실행
# streamlit run recomendation.py --server.enableXsrfProtection false

# streamlit 1.24.0 이상 버전에서 파일 업로드할 경우 AxiosError: Request failed with status code 403 발생할 수 있음
# AxiosError 403 에러 발생 시 streamlit==1.24.0 버전으로 변경 
# pip install streamlit==1.24.0

# import ray
import json
import requests
import pandas as pd
import streamlit as st
from lib.template import Template
from lib.prepro import Preprocessing
# from database.connector import Database # , SelectTB
from io import StringIO
import time
import numpy as np
import plotly.express as px


# import matplotlib.pyplot as plt
# import seaborn as sns


st.set_page_config(layout="wide")
st.sidebar.title("Details")

# 분류, 이상 탐지 등 추천받을 머신러닝 모델 선택
option = st.sidebar.selectbox(
    '머신러닝 유형 선택', ('분류', '군집', '회귀'))
connecton_option = st.sidebar.selectbox(
    'Select how to upload data', ('File_upload'))

uploaded_file = None
df = None

if connecton_option == 'File_upload':
    uploaded_file = st.sidebar.file_uploader("csv file upload", type="csv") # 파일 업로드
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

with st.spinner('Wait for it...'):
    updated_df = None
    # Uploaded data Dashboard
    if uploaded_file is not None or df is not None:
        template = Template(df)
        st.subheader('데이터 분석')
        col_list = df.columns.tolist() # 데이터 전처리 옵션 설정 리스트
        target_feture = ""
        if option == '분류' or option == '회귀':
            target_feture = st.sidebar.multiselect('Select Target Column', options=col_list)
        elif option == '군집':
            st.sidebar.info("군집 분석은 비지도 학습이므로 타겟 컬럼이 필요하지 않습니다.")
        data_to_drop = st.sidebar.multiselect('Drop Cloumns', options=col_list)
        data_for_labelencoding = st.sidebar.multiselect('Choose LabelEncoding column name', options=col_list)
        
        tab_eda_df, tab_eda_info, tab_Label_counts = st.tabs(['Original data', 'Null information', 'Target Data Counts']) # tab_Label_counts Labels counts
        # tab_eda_df, tab_eda_info tab UI Template
        template.eda_df(tab_eda_df=tab_eda_df, tab_eda_info=tab_eda_info)
        label_to_drop = ""
        with tab_Label_counts: # Target Data 정보 출력 및 시각화
            if target_feture:            
                label_to_drop = template.label_to_drop(target_feture) # 제거할 Target 데이터 선택
            else:
                template.sample_df()

  
        if data_for_labelencoding:
            prepro = Preprocessing()
            if updated_df is None:
                # st.write(type(df[data_for_labelencoding]))
                df = prepro.encoded_df(df, data_for_labelencoding[0])
                updated_df = df
            if updated_df is not None:
                updated_df = prepro.encoded_df(updated_df, data_for_labelencoding[0])

       
        if data_to_drop:
            for data in data_to_drop:
                updated_df = df.drop(data_to_drop, axis=1)

     
        try:
            if label_to_drop:
                target_feture = target_feture[0]
                label_to_drop = label_to_drop[0]
                updated_df = df[df[target_feture] != label_to_drop]
        except ValueError:
            st.write('1개 이상 데이터가 남아있어야 합니다.')

        # 데이터 전처리된 데이터 출력
        if updated_df is not None: 
            st.subheader('데이터 전처리')
            st.dataframe(updated_df, use_container_width=True)
        
        if st.sidebar.button("초기화"):
            st.cache_resource.clear()


#################### Starting ML traning
        button_for_training = st.sidebar.button("머신러닝 테스트 실행", key="button1") 
        if button_for_training: # 분류, 이상탐지 옵션에 따라 머신러닝 학습 진행
            start_time = time.time()
            # start_time = time.time() # 학습 시간 체크 시 설정

            if option == '분류' or option == '군집' or option == '회귀':
                st.subheader('머신러닝 학습 결과')
                with st.spinner('Wait for it...'):
                    if updated_df is None:
                        updated_df = df   

                    # 데이터 검증
                    if option != '군집':  # 군집이 아닌 경우에만 타겟 컬럼 검증
                        if target_feture is None or len(target_feture) == 0:
                            st.error("타겟 컬럼을 선택해주세요.")
                            st.stop()
                    
                    # 수치형 데이터만 선택
                    numeric_df = updated_df.select_dtypes(include=[np.number])
                    if len(numeric_df.columns) == 0:
                        st.error("수치형 데이터가 없습니다.")
                        st.stop()
                    
                    # 각 머신러닝 유형별 검증
                    if option == '분류':
                        # 분류: 타겟이 범주형이어야 함
                        if target_feture[0] not in updated_df.columns:
                            st.error(f"타겟 컬럼 '{target_feture[0]}'이 데이터에 존재하지 않습니다.")
                            st.stop()
                        
                        # 분류의 경우 타겟이 범주형이어야 하므로 수치형이 아닌 컬럼도 허용
                        target_unique_count = updated_df[target_feture[0]].nunique()
                        if target_unique_count < 2:
                            st.error("분류를 위해서는 타겟 컬럼에 최소 2개 이상의 클래스가 필요합니다.")
                            st.stop()
                        elif target_unique_count > 50:
                            st.warning(f"타겟 컬럼에 {target_unique_count}개의 클래스가 있습니다. 다중 분류가 될 수 있습니다.")
                    
                    elif option == '회귀':
                        # 회귀: 타겟이 수치형이어야 함
                        if target_feture[0] not in numeric_df.columns:
                            st.error(f"회귀를 위해서는 타겟 컬럼 '{target_feture[0]}'이 수치형이어야 합니다.")
                            st.stop()
                        
                        # 회귀의 경우 타겟에서 수치형 컬럼 제외
                        feature_cols = [col for col in numeric_df.columns if col != target_feture[0]]
                        if len(feature_cols) == 0:
                            st.error("회귀를 위한 특성 컬럼이 없습니다.")
                            st.stop()
                        numeric_df = updated_df[feature_cols + [target_feture[0]]]
                    
                    elif option == '군집':
                        # 군집: 타겟 컬럼이 없어야 함 (비지도 학습)
                        # st.warning("군집 분석은 비지도 학습이므로 타겟 컬럼을 사용하지 않습니다.")
                        # 군집의 경우 모든 수치형 데이터를 특성으로 사용
                        feature_cols = numeric_df.columns.tolist()
                        if len(feature_cols) == 0:
                            st.error("군집 분석을 위한 특성 컬럼이 없습니다.")
                            st.stop()
                        numeric_df = updated_df[feature_cols]
                        # 군집의 경우 타겟 정보를 빈 문자열로 설정 (백엔드에서 무시됨)
                        target_feture = ""

                    
                    json_data = numeric_df.to_json() # pandas DataFrame를 json 형태로 변환
                    data_dump = json.dumps({'json_data':json_data, 'target': target_feture}) # 학습 데이터, Target Data 객체를 문자열로 직렬화(serialize)
                    data = json.loads(data_dump) # json을 파이썬 객체로 변환
                    
                    try:
                        if option == '분류':
                            response = requests.post('http://127.0.0.1:8000/automl/classification', json=data, timeout=300)
                        elif option == '군집':
                            response = requests.post('http://127.0.0.1:8000/automl/clustering', json=data, timeout=300)
                        elif option == '회귀':
                            response = requests.post('http://127.0.0.1:8000/automl/regression', json=data, timeout=300)
                        
                        if response.status_code == 200: 
                            json_data = response.json() 
                            data = json.loads(json_data['result'])
                             # 분류 모델
                            if option == '분류':
                                # 모든 평가지표를 통합하여 보여주기
                                clf_results = {}
                                scoring_methods = ['accuracy', 'recall', 'precision', 'f1_weighted']
                                
                                # 각 지표별 결과 수집
                                for i, scoring_method in enumerate(scoring_methods):
                                    if str(i) in data:
                                        best_json = data[str(i)]["best"]
                                        best_data = json.loads(best_json)
                                        clf_results[scoring_method] = best_data.get('models', {})
                                
                                # 모든 모델명 수집
                                all_models = set()
                                for models_dict in clf_results.values():
                                    all_models.update(models_dict.keys())
                                
                                # 통합 DataFrame 생성
                                integrated_data = []
                                for model_name in all_models:
                                    model_scores = {'Model': model_name}
                                    for scoring_method in scoring_methods:
                                        if scoring_method in clf_results:
                                            model_scores[scoring_method] = clf_results[scoring_method].get(model_name, 0.0)
                                        else:
                                            model_scores[scoring_method] = 0.0
                                    integrated_data.append(model_scores)
                                
                                # DataFrame 생성 및 정렬
                                if integrated_data:
                                    sorted_df = pd.DataFrame(integrated_data).set_index('Model')
                                    # f1_weighted 기준으로 정렬 (없으면 accuracy)
                                    if 'f1_weighted' in sorted_df.columns:
                                        sorted_df = sorted_df.sort_values(by='f1_weighted', ascending=False)
                                    elif 'accuracy' in sorted_df.columns:
                                        sorted_df = sorted_df.sort_values(by='accuracy', ascending=False)
                                    else:
                                        sorted_df = sorted_df.sort_values(by=sorted_df.columns[0], ascending=False)
                                else:
                                    sorted_df = pd.DataFrame({'Score': [0.0]}, index=['No Models'])
                            if option == '회귀':
                                # 모든 평가지표를 통합하여 보여주기
                                reg_results = {}
                                scoring_methods = ['neg_mean_squared_error', 'neg_mean_absolute_error']
                                
                                # 각 지표별 결과 수집
                                for i, scoring_method in enumerate(scoring_methods):
                                    if str(i) in data:
                                        best_json = data[str(i)]["best"]
                                        best_data = json.loads(best_json)
                                        reg_results[scoring_method] = best_data.get('models', {})
                                
                                # 모든 모델명 수집
                                all_models = set()
                                for models_dict in reg_results.values():
                                    all_models.update(models_dict.keys())
                                
                                # 통합 DataFrame 생성
                                integrated_data = []
                                for model_name in all_models:
                                    model_scores = {'Model': model_name}
                                    for scoring_method in scoring_methods:
                                        if scoring_method in reg_results:
                                            model_scores[scoring_method] = reg_results[scoring_method].get(model_name, 0.0)
                                        else:
                                            model_scores[scoring_method] = 0.0
                                    integrated_data.append(model_scores)
                                
                                # DataFrame 생성 및 정렬
                                if integrated_data:
                                    sorted_df = pd.DataFrame(integrated_data).set_index('Model')
                                    # neg_mean_squared_error 기준으로 정렬 (음수값이므로 내림차순)
                                    if 'neg_mean_squared_error' in sorted_df.columns:
                                        sorted_df = sorted_df.sort_values(by='neg_mean_squared_error', ascending=False)
                                    else:
                                        sorted_df = sorted_df.sort_values(by=sorted_df.columns[0], ascending=False)
                                else:
                                    sorted_df = pd.DataFrame({'Score': [0.0]}, index=['No Models'])
                            if option == '군집':
                                # 모든 평가지표를 통합하여 보여주기
                                cluster_results = {}
                                scoring_methods = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
                                
                                # 각 지표별 결과 수집
                                for i, scoring_method in enumerate(scoring_methods):
                                    if str(i) in data:
                                        best_json = data[str(i)]["best"]
                                        best_data = json.loads(best_json)
                                        cluster_results[scoring_method] = best_data.get('models', {})
                                
                                # 모든 모델명 수집
                                all_models = set()
                                for models_dict in cluster_results.values():
                                    all_models.update(models_dict.keys())
                                
                                # 통합 DataFrame 생성
                                integrated_data = []
                                for model_name in all_models:
                                    model_scores = {'Model': model_name}
                                    for scoring_method in scoring_methods:
                                        if scoring_method in cluster_results:
                                            model_scores[scoring_method] = cluster_results[scoring_method].get(model_name, 0.0)
                                        else:
                                            model_scores[scoring_method] = 0.0
                                    integrated_data.append(model_scores)
                                
                                # DataFrame 생성 및 정렬
                                if integrated_data:
                                    sorted_df = pd.DataFrame(integrated_data).set_index('Model')
                                    # silhouette 기준으로 정렬 (높을수록 좋음)
                                    if 'silhouette' in sorted_df.columns:
                                        sorted_df = sorted_df.sort_values(by='silhouette', ascending=False)
                                    elif 'calinski_harabasz' in sorted_df.columns:
                                        sorted_df = sorted_df.sort_values(by='calinski_harabasz', ascending=False)
                                    elif 'davies_bouldin' in sorted_df.columns:
                                        sorted_df = sorted_df.sort_values(by='davies_bouldin', ascending=True)  # 낮을수록 좋음
                                    else:
                                        sorted_df = sorted_df.sort_values(by=sorted_df.columns[0], ascending=False)
                                else:
                                    sorted_df = pd.DataFrame({'Score': [0.0]}, index=['No Models'])
                                
                                # 군집 분석 결과 정렬 로직 제거 (이미 정렬됨)
                            # sorted_concat_df = concat_df.sort_values(by='f1_weighted', ascending=False)

                            # 모델 추천
                            # 모든 평가지표를 포함한 우상향 라인 차트
                            chart_df = sorted_df.reset_index()
                            # 우상향 시각화를 위해 순서를 뒤집기 (낮은 성능부터 높은 성능 순서)
                            chart_df = chart_df.iloc[::-1].reset_index(drop=True)
                            
                            # 모든 평가지표를 long format으로 변환
                            chart_df_long = chart_df.melt(
                                id_vars=['Model'], 
                                var_name='Metric', 
                                value_name='Score'
                            )
                            
                            fig = px.line(
                                chart_df_long,
                                x='Model',
                                y='Score',
                                color='Metric',
                                # title='모든 평가지표 성능 비교 - 우상향 성능 트렌드',
                                labels={'Model': '모델', 'Score': '성능 점수', 'Metric': '평가지표'},
                                markers=True
                            )
                            
                            # x축 순서 고정 (낮은 성능부터 높은 성능으로 우상향)
                            fig.update_xaxes(categoryorder='array', categoryarray=chart_df['Model'].tolist())
                            fig.update_layout(
                                height=500,
                                xaxis_title="모델",
                                yaxis_title="성능 점수",
                                legend_title="평가지표"
                            )
                            
                            # 라인 및 마커 스타일 설정
                            fig.update_traces(
                                line=dict(width=3),
                                marker=dict(size=8)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # 평가지표 설명 추가
                            if option == '분류':
                                st.subheader('📊 분류 모델 성능 평가')
                                st.info("""
                                **평가지표 설명:**
                                - **accuracy**: 정확도 (높을수록 좋음)
                                - **recall**: 재현율 (높을수록 좋음)
                                - **precision**: 정밀도 (높을수록 좋음)  
                                - **f1_weighted**: 가중 F1 스코어 (높을수록 좋음)
                                """)
                            elif option == '회귀':
                                st.subheader('📊 회귀 모델 성능 평가')
                                st.info("""
                                **평가지표 설명:**
                                - **neg_mean_squared_error**: 음의 평균 제곱 오차 (0에 가까울수록 좋음)
                                - **neg_mean_absolute_error**: 음의 평균 절대 오차 (0에 가까울수록 좋음)
                                """)
                            elif option == '군집':
                                st.subheader('📊 군집 모델 성능 평가')
                                st.info("""
                                **평가지표 설명:**
                                - **silhouette**: 실루엣 스코어 (높을수록 좋음, -1~1)
                                - **calinski_harabasz**: 칼린스키-하라바즈 지수 (높을수록 좋음)
                                - **davies_bouldin**: 데이비스-볼딘 지수 (낮을수록 좋음)
                                """)
                            
                            st.subheader('🎯 모델 비교')
                            
                            # 소수점 자리수 형식 지정
                            if not sorted_df.empty and 'Score' not in sorted_df.columns:
                                # 수치형 컬럼에 대해 소수점 4자리로 표시
                                formatted_df = sorted_df.round(4)
                                st.dataframe(formatted_df, use_container_width=True)
                                
                            else:
                                st.dataframe(sorted_df, use_container_width=True)

                            # 개선된 추천 시스템
                            data = updated_df
                            
                            # 데이터 분석 및 특성 추출
                            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                            categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
                            
                            # 데이터 특성 분석
                            data_characteristics = {}
                            for col in numeric_cols:
                                col_data = data[col].dropna()
                                if len(col_data) > 0:
                                    data_characteristics[col] = {
                                        'mean': col_data.mean(),
                                        'std': col_data.std(),
                                        'skewness': col_data.skew(),
                                        'unique_count': col_data.nunique(),
                                        'missing_ratio': data[col].isnull().sum() / len(data)
                                    }
                            
                            # 지능형 전처리 추천
                            preprocessing_recommendations = []
                            
                            for col in numeric_cols:
                                if col in data_characteristics:
                                    char = data_characteristics[col]
                                    recommendations = []
                                    
                                    # 결측치 처리
                                    if char['missing_ratio'] > 0.1:
                                        recommendations.append('Missing Value Imputation (Mean/Median)')
                                    
                                    # 이상치 탐지
                                    if abs(char['skewness']) > 1:
                                        recommendations.append('Outlier Detection')
                                    
                                    # 스케일링 추천
                                    if char['std'] > char['mean'] * 0.5:
                                        recommendations.append('Standardization (Z-score)')
                                    else:
                                        recommendations.append('Normalization (Min-Max)')
                                    
                                    # 분포 특성에 따른 추천
                                    if abs(char['skewness']) > 2:
                                        recommendations.append('Log/Box-Cox Transformation')
                                    
                                    # 기본 추천
                                    if len(recommendations) < 3:
                                        recommendations.extend(['Feature Engineering', 'Robust Scaling'])
                                    
                                    preprocessing_recommendations.append({
                                        'Column': col,
                                        'Data Type': 'Numeric',
                                        'Primary Recommendation': recommendations[0] if recommendations else 'Standardization',
                                        'Secondary Recommendation': recommendations[1] if len(recommendations) > 1 else 'Normalization',
                                        'Additional Techniques': ', '.join(recommendations[2:]) if len(recommendations) > 2 else 'None'
                                    })
                            
                            for col in categorical_cols:
                                unique_count = data[col].nunique()
                                missing_ratio = data[col].isnull().sum() / len(data)
                                
                                recommendations = []
                                if missing_ratio > 0.1:
                                    recommendations.append('Missing Value Imputation')
                                
                                if unique_count < 10:
                                    recommendations.append('One-Hot Encoding')
                                elif unique_count < 50:
                                    recommendations.append('Label Encoding')
                                else:
                                    recommendations.append('Target Encoding')
                                
                                if unique_count > 100:
                                    recommendations.append('Feature Selection')
                                
                                preprocessing_recommendations.append({
                                    'Column': col,
                                    'Data Type': 'Categorical',
                                    # 'Unique Values': unique_count,
                                    # 'Missing Ratio': f"{missing_ratio:.2%}",
                                    'Primary Recommendation': recommendations[0] if recommendations else 'One-Hot Encoding',
                                    'Secondary Recommendation': recommendations[1] if len(recommendations) > 1 else 'Label Encoding',
                                    'Additional Techniques': ', '.join(recommendations[2:]) if len(recommendations) > 2 else 'None'
                                })
                            
                            # 시각화 추천
                            visualization_recommendations = []
                            
                            for col in numeric_cols:
                                if col in data_characteristics:
                                    char = data_characteristics[col]
                                    vis_recommendations = []
                                    
                                    # 분포 특성에 따른 시각화
                                    if abs(char['skewness']) > 1:
                                        vis_recommendations.extend(['Box Plot', 'Histogram with KDE'])
                                    else:
                                        vis_recommendations.extend(['Histogram', 'Density Plot'])
                                    
                                    # 이상치 탐지
                                    if abs(char['skewness']) > 2:
                                        vis_recommendations.append('Outlier Plot')
                                    
                                    # 기본 추천
                                    vis_recommendations.extend(['Distribution Plot', 'Summary Statistics'])
                                    
                                    visualization_recommendations.append({
                                        'Column': col,
                                        'Data Type': 'Numeric',
                                        'Primary Visualization': vis_recommendations[0],
                                        'Secondary Visualization': vis_recommendations[1] if len(vis_recommendations) > 1 else 'Histogram',
                                        'Additional Charts': ', '.join(vis_recommendations[2:]) if len(vis_recommendations) > 2 else 'None'
                                    })
                            
                            for col in categorical_cols:
                                unique_count = data[col].nunique()
                                vis_recommendations = []
                                
                                if unique_count <= 10:
                                    vis_recommendations.extend(['Bar Chart', 'Pie Chart'])
                                else:
                                    vis_recommendations.extend(['Bar Chart', 'Horizontal Bar Chart'])
                                
                                vis_recommendations.extend(['Value Counts', 'Category Distribution'])
                                
                                visualization_recommendations.append({
                                    'Column': col,
                                    'Data Type': 'Categorical',
                                    'Unique Values': unique_count,
                                    'Primary Visualization': vis_recommendations[0],
                                    'Secondary Visualization': vis_recommendations[1] if len(vis_recommendations) > 1 else 'Bar Chart',
                                    'Additional Charts': ', '.join(vis_recommendations[2:]) if len(vis_recommendations) > 2 else 'None'
                                })
                            
                            # 모델 성능 기반 추천
                            best_model = sorted_df.iloc[0] if len(sorted_df) > 0 else None

                            # 결과 표시
                            prepro_recommendations_df = pd.DataFrame(preprocessing_recommendations)
                            vis_recommendations_df = pd.DataFrame(visualization_recommendations)
                            
                            
                            st.subheader("🔧 전처리 추천 결과")
                            st.dataframe(prepro_recommendations_df, use_container_width=True)
                            
                            st.subheader("📈 시각화 추천 결과")
                            st.dataframe(vis_recommendations_df, use_container_width=True)
                            
                            # 통합 추천 결과
                            if best_model is not None:
                                st.subheader("🎯 최적 모델 추천 결과")
                                best_model_name = sorted_df.index[0] # if len(sorted_df) > 0 else "Unknown"
                                
                                recommendation_summary = {
                                    'Best Model': best_model_name,
                                    'Key Preprocessing': prepro_recommendations_df.iloc[0]['Primary Recommendation'] if len(prepro_recommendations_df) > 0 else 'N/A',
                                    'Key Visualization': vis_recommendations_df.iloc[0]['Primary Visualization'] if len(vis_recommendations_df) > 0 else 'N/A'
                                }
                                
                                summary_df = pd.DataFrame([recommendation_summary])
                                st.dataframe(summary_df, use_container_width=True)
                        else:
                            error_data = response.json()
                            st.error(f"API 오류: {error_data.get('error', '알 수 없는 오류')}")
                            
                    except requests.exceptions.Timeout:
                        st.error("요청 시간이 초과되었습니다. 서버가 응답하지 않습니다.")
                    except requests.exceptions.ConnectionError:
                        st.error("서버에 연결할 수 없습니다. 백엔드 서버가 실행 중인지 확인해주세요.")
                    except Exception as e:
                        st.error(f"요청 중 오류가 발생했습니다: {str(e)}")