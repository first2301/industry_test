import pandas as pd
import numpy as np
from old.yaml_system_test import RecommendationEngine, RuleSet
from specific_preprocessing_examples import SpecificPreprocessingMethods
import streamlit as st
import json
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class AutoMLIntegration:
    """
    YAML 기반 추천 시스템과 AutoML 시스템의 통합 클래스
    """
    
    def __init__(self, rules_file: str = "rules.yaml"):
        self.recommendation_engine = RecommendationEngine(rules_file)
        self.preprocessing_methods = SpecificPreprocessingMethods()
        
    def get_preprocessing_recommendations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        데이터프레임에 대한 전처리 추천을 생성합니다.
        
        Args:
            df: 입력 데이터프레임
            
        Returns:
            dict: 추천 결과 및 실행 가능한 코드
        """
        
        # 1. 추천 생성
        recommendations = self.recommendation_engine.run(df)
        
        # 2. 우선순위별 정렬
        priority_recommendations = self._sort_by_priority(recommendations['preprocessing'])
        
        # 3. 실행 가능한 코드 생성
        executable_code = self._generate_executable_code(priority_recommendations)
        
        return {
            'recommendations': priority_recommendations,
            'executable_code': executable_code,
            'summary': self._generate_summary(priority_recommendations)
        }
    
    def _sort_by_priority(self, recommendations: Dict) -> List[Dict]:
        """우선순위별로 추천 정렬"""
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        
        sorted_recs = []
        for col, rec_data in recommendations.items():
            if col.startswith('_'):
                continue
                
            priority = rec_data.get('priority', 'low')
            for action in rec_data.get('recommendations', []):
                sorted_recs.append({
                    'column': col,
                    'action': action,
                    'priority': priority,
                    'priority_order': priority_order.get(priority, 3)
                })
        
        return sorted(sorted_recs, key=lambda x: x['priority_order'])
    
    def _generate_executable_code(self, recommendations: List[Dict]) -> str:
        """실행 가능한 코드 생성"""
        code_lines = [
            "import pandas as pd",
            "import numpy as np",
            "from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler",
            "from sklearn.preprocessing import LabelEncoder, OneHotEncoder",
            "from scipy.stats import boxcox, yeojohnson",
            "from sklearn.feature_selection import VarianceThreshold",
            "",
            "def apply_preprocessing(df):",
            "    \"\"\"자동 생성된 전처리 함수\"\"\"",
            "    processed_df = df.copy()",
            "    results = {}",
            ""
        ]
        
        # 각 추천에 대한 코드 생성
        for rec in recommendations:
            column = rec['column']
            action = rec['action']
            
            # 액션별 코드 생성
            if action == 'yeo_johnson_transform':
                code_lines.extend([
                    f"    # {column}: Yeo-Johnson 변환",
                    f"    transformed_data, lambda_param = yeojohnson(processed_df['{column}'].dropna())",
                    f"    processed_df['{column}'] = transformed_data",
                    f"    results['{column}_yeo_johnson_lambda'] = lambda_param",
                    ""
                ])
            elif action == 'boxcox_transform':
                code_lines.extend([
                    f"    # {column}: Box-Cox 변환",
                    f"    if (processed_df['{column}'] <= 0).any():",
                    f"        processed_df['{column}'] = processed_df['{column}'] - processed_df['{column}'].min() + 1",
                    f"    transformed_data, lambda_param = boxcox(processed_df['{column}'].dropna())",
                    f"    processed_df['{column}'] = transformed_data",
                    f"    results['{column}_boxcox_lambda'] = lambda_param",
                    ""
                ])
            elif action == 'log_transform':
                code_lines.extend([
                    f"    # {column}: 로그 변환",
                    f"    if (processed_df['{column}'] <= 0).any():",
                    f"        processed_df['{column}'] = processed_df['{column}'] - processed_df['{column}'].min() + 1",
                    f"    processed_df['{column}'] = np.log(processed_df['{column}'])",
                    ""
                ])
            elif action == 'standard_scaler':
                code_lines.extend([
                    f"    # {column}: 표준화",
                    f"    scaler = StandardScaler()",
                    f"    processed_df['{column}'] = scaler.fit_transform(processed_df[['{column}']])",
                    ""
                ])
            elif action == 'one_hot_encode':
                code_lines.extend([
                    f"    # {column}: 원핫 인코딩",
                    f"    encoded_cols = pd.get_dummies(processed_df['{column}'], prefix='{column}')",
                    f"    processed_df = pd.concat([processed_df.drop('{column}', axis=1), encoded_cols], axis=1)",
                    ""
                ])
            elif action == 'drop_column':
                code_lines.extend([
                    f"    # {column}: 컬럼 제거",
                    f"    processed_df = processed_df.drop('{column}', axis=1)",
                    f"    results['{column}_dropped'] = True",
                    ""
                ])
        
        code_lines.extend([
            "    return processed_df, results",
            "",
            "# 사용 예시:",
            "# processed_df, preprocessing_results = apply_preprocessing(df)"
        ])
        
        return "\n".join(code_lines)
    
    def _generate_summary(self, recommendations: List[Dict]) -> Dict[str, Any]:
        """추천 요약 생성"""
        summary = {
            'total_recommendations': len(recommendations),
            'high_priority': len([r for r in recommendations if r['priority'] == 'high']),
            'medium_priority': len([r for r in recommendations if r['priority'] == 'medium']),
            'low_priority': len([r for r in recommendations if r['priority'] == 'low']),
            'action_counts': {}
        }
        
        # 액션별 개수 집계
        for rec in recommendations:
            action = rec['action']
            summary['action_counts'][action] = summary['action_counts'].get(action, 0) + 1
        
        return summary
    
    def apply_recommendations_to_streamlit(self, df: pd.DataFrame, 
                                         selected_recommendations: List[str] = None) -> pd.DataFrame:
        """
        Streamlit 환경에서 추천사항을 적용합니다.
        
        Args:
            df: 입력 데이터프레임
            selected_recommendations: 선택된 추천사항 리스트
            
        Returns:
            pd.DataFrame: 전처리된 데이터프레임
        """
        
        processed_df = df.copy()
        
        # 추천 생성
        result = self.get_preprocessing_recommendations(df)
        recommendations = result['recommendations']
        
        # 선택된 추천사항만 적용
        if selected_recommendations:
            recommendations = [r for r in recommendations if r['action'] in selected_recommendations]
        
        # 추천사항 적용
        for rec in recommendations:
            column = rec['column']
            action = rec['action']
            
            try:
                if action == 'yeo_johnson_transform':
                    processed_df[column] = self.preprocessing_methods.yeo_johnson_transform(
                        processed_df[column], column)
                elif action == 'log_transform':
                    processed_df[column] = self.preprocessing_methods.log_transform(
                        processed_df[column], column)
                elif action == 'standard_scaler':
                    processed_df[column] = self.preprocessing_methods.standard_scaler(
                        processed_df[column], column)
                elif action == 'one_hot_encode':
                    encoded_df = self.preprocessing_methods.one_hot_encode(
                        processed_df[column], column)
                    processed_df = pd.concat([processed_df.drop(column, axis=1), encoded_df], axis=1)
                elif action == 'drop_column':
                    processed_df = processed_df.drop(column, axis=1)
                    
            except Exception as e:
                logger.error(f"추천 적용 실패 - {column}: {action}, 오류: {e}")
                continue
        
        return processed_df

# Streamlit 인터페이스 구성 요소
class StreamlitRecommendationUI:
    """Streamlit용 추천 시스템 UI 컴포넌트"""
    
    def __init__(self, integration: AutoMLIntegration):
        self.integration = integration
    
    def render_recommendation_panel(self, df: pd.DataFrame):
        """추천 패널 렌더링"""
        
        st.subheader("🤖 AI 전처리 추천")
        
        # 추천 생성
        with st.spinner("데이터 분석 중..."):
            result = self.integration.get_preprocessing_recommendations(df)
        
        # 요약 정보 표시
        summary = result['summary']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 추천수", summary['total_recommendations'])
        with col2:
            st.metric("높은 우선순위", summary['high_priority'])
        with col3:
            st.metric("중간 우선순위", summary['medium_priority'])
        with col4:
            st.metric("낮은 우선순위", summary['low_priority'])
        
        # 추천사항 표시
        recommendations = result['recommendations']
        
        if recommendations:
            st.subheader("📋 추천사항")
            
            # 추천사항을 데이터프레임으로 변환
            rec_df = pd.DataFrame(recommendations)
            
            # 우선순위별 색상 매핑
            def priority_color(priority):
                colors = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
                return colors.get(priority, '⚪')
            
            rec_df['priority_icon'] = rec_df['priority'].apply(priority_color)
            
            # 선택 가능한 추천사항 표시
            selected_actions = []
            for i, rec in enumerate(recommendations):
                col1, col2 = st.columns([1, 4])
                with col1:
                    selected = st.checkbox(
                        f"{rec['priority_icon']} {rec['priority']}", 
                        key=f"rec_{i}",
                        value=rec['priority'] == 'high'  # 높은 우선순위는 기본 선택
                    )
                with col2:
                    st.write(f"**{rec['column']}**: {rec['action']}")
                
                if selected:
                    selected_actions.append(rec['action'])
            
            # 적용 버튼
            if st.button("선택한 추천사항 적용"):
                with st.spinner("전처리 적용 중..."):
                    processed_df = self.integration.apply_recommendations_to_streamlit(
                        df, selected_actions)
                    st.success("전처리 완료!")
                    return processed_df
        
        # 생성된 코드 표시
        if st.expander("🔧 생성된 코드 보기"):
            st.code(result['executable_code'], language='python')
        
        return df

# 사용 예시 함수
def integrate_with_existing_automl():
    """기존 AutoML 시스템과의 통합 예시"""
    
    # 통합 객체 생성
    integration = AutoMLIntegration()
    
    # Streamlit UI 생성
    ui = StreamlitRecommendationUI(integration)
    
    # 기존 app.py에 추가할 코드
    example_code = """
    # 기존 app.py의 데이터 업로드 후 추가할 코드
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # === 추가: AI 전처리 추천 시스템 ===
        integration = AutoMLIntegration()
        ui = StreamlitRecommendationUI(integration)
        
        # 추천 패널 렌더링
        processed_df = ui.render_recommendation_panel(df)
        
        # 전처리된 데이터를 기존 AutoML 시스템에 전달
        if processed_df is not None:
            updated_df = processed_df
        
        # === 기존 AutoML 코드 계속 ===
        # ... (기존 머신러닝 학습 코드)
    """
    
    print("🔗 기존 AutoML 시스템과의 통합 방법:")
    print(example_code)

if __name__ == "__main__":
    integrate_with_existing_automl() 