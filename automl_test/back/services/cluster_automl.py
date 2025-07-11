from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Clustering:
    def __init__(self, df, scoring='silhouette'):
        self.df = df
        self.scoring = scoring
        self.cv = 3
        self.X = df

    def evaluate_clustering(self, labels):
        """클러스터링 결과를 여러 지표로 평가합니다."""
        if len(np.unique(labels)) < 2:
            return {
                'silhouette_score': 0.0,
                'calinski_harabasz_score': 0.0,
                'davies_bouldin_score': float('inf')
            }
        
        try:
            silhouette = silhouette_score(self.X, labels)
            calinski_harabasz = calinski_harabasz_score(self.X, labels)
            davies_bouldin = davies_bouldin_score(self.X, labels)
            
            return {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz,
                'davies_bouldin_score': davies_bouldin
            }
        except Exception as e:
            logger.error(f"클러스터링 평가 실패: {e}")
            return {
                'silhouette_score': 0.0,
                'calinski_harabasz_score': 0.0,
                'davies_bouldin_score': float('inf')
            }

    def kmeans_model(self):
        """K-means 모델을 정의합니다."""
        try:
            model = KMeans(n_clusters=5, random_state=42)
            labels = model.fit_predict(self.X)
            
            scores = self.evaluate_clustering(labels)
            
            if self.scoring == 'silhouette':
                return scores['silhouette_score']
            elif self.scoring == 'calinski_harabasz':
                return scores['calinski_harabasz_score']
            elif self.scoring == 'davies_bouldin':
                return scores['davies_bouldin_score']
            else:
                return scores['silhouette_score']
                
        except Exception as e:
            logger.error(f"K-means 학습 실패: {e}")
            return 0.0 if self.scoring != 'davies_bouldin' else float('inf')

    def dbscan_model(self):
        """DBSCAN 모델을 정의합니다."""
        try:
            model = DBSCAN(eps=0.5, min_samples=5)
            labels = model.fit_predict(self.X)
            
            scores = self.evaluate_clustering(labels)
            
            if self.scoring == 'silhouette':
                return scores['silhouette_score']
            elif self.scoring == 'calinski_harabasz':
                return scores['calinski_harabasz_score']
            elif self.scoring == 'davies_bouldin':
                return scores['davies_bouldin_score']
            else:
                return scores['silhouette_score']
                
        except Exception as e:
            logger.error(f"DBSCAN 학습 실패: {e}")
            return 0.0 if self.scoring != 'davies_bouldin' else float('inf')

    def agglomerative_model(self):
        """계층적 군집화 모델을 정의합니다."""
        try:
            model = AgglomerativeClustering(n_clusters=5)
            labels = model.fit_predict(self.X)
            
            scores = self.evaluate_clustering(labels)
            
            if self.scoring == 'silhouette':
                return scores['silhouette_score']
            elif self.scoring == 'calinski_harabasz':
                return scores['calinski_harabasz_score']
            elif self.scoring == 'davies_bouldin':
                return scores['davies_bouldin_score']
            else:
                return scores['silhouette_score']
                
        except Exception as e:
            logger.error(f"계층적 군집화 학습 실패: {e}")
            return 0.0 if self.scoring != 'davies_bouldin' else float('inf')

    def spectral_model(self):
        """Spectral Clustering 모델을 정의합니다."""
        try:
            model = SpectralClustering(n_clusters=5, gamma=1.0, random_state=42)
            labels = model.fit_predict(self.X)
            
            scores = self.evaluate_clustering(labels)
            
            if self.scoring == 'silhouette':
                return scores['silhouette_score']
            elif self.scoring == 'calinski_harabasz':
                return scores['calinski_harabasz_score']
            elif self.scoring == 'davies_bouldin':
                return scores['davies_bouldin_score']
            else:
                return scores['silhouette_score']
                
        except Exception as e:
            logger.error(f"Spectral Clustering 학습 실패: {e}")
            return 0.0 if self.scoring != 'davies_bouldin' else float('inf')

    def gaussian_mixture_model(self):
        """Gaussian Mixture 모델을 정의합니다."""
        try:
            model = GaussianMixture(n_components=5, covariance_type='full', random_state=42)
            labels = model.fit_predict(self.X)
            
            scores = self.evaluate_clustering(labels)
            
            if self.scoring == 'silhouette':
                return scores['silhouette_score']
            elif self.scoring == 'calinski_harabasz':
                return scores['calinski_harabasz_score']
            elif self.scoring == 'davies_bouldin':
                return scores['davies_bouldin_score']
            else:
                return scores['silhouette_score']
                
        except Exception as e:
            logger.error(f"Gaussian Mixture 학습 실패: {e}")
            return 0.0 if self.scoring != 'davies_bouldin' else float('inf')

    def run_cluster_models(self):
        """모든 군집화 모델을 실행하고 결과를 반환합니다."""
        logger.info(f"{self.scoring} 지표로 군집화 모델 학습을 시작합니다...")
        
        models = {
            'kmeans': self.kmeans_model(),
            'dbscan': self.dbscan_model(),
            'agglomerative': self.agglomerative_model(),
            'spectral': self.spectral_model(),
            'gaussian_mixture': self.gaussian_mixture_model()
        }
        
        best_results_df = pd.DataFrame.from_dict(models, orient='index', columns=[self.scoring])
        # DataFrame을 JSON 문자열로 변환
        best_results_json = best_results_df.to_json()
    
        logger.info(f"{self.scoring} 지표 학습 완료. 최고 성능: {max(models.values()):.4f}")
        return {'best': best_results_json}

def compare_cluster_models(df):
    '''
    모든 군집화 모델을 여러 지표로 학습합니다.
    
    Args:
        df: 입력 데이터프레임
    
    Returns:
        dict: 각 지표별 결과
    '''
    scorings = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
    results = dict()
    
    logger.info(f"총 {len(scorings)}개 지표로 군집화 모델 학습을 시작합니다.")
    
    for idx, scoring in enumerate(scorings):
        logger.info(f"진행률: {idx+1}/{len(scorings)} - {scoring} 지표 학습 중...")
        results[idx] = Clustering(df, scoring).run_cluster_models()
    
    logger.info("모든 군집화 모델 학습이 완료되었습니다.")
    return results