from flaml import AutoML
import pandas as pd
import numpy as np
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlamlClassification:
    def __init__(self, df, target, scoring='f1_weighted'):
        self.df = df
        self.target = target
        self.X = df.drop(target, axis=1)
        self.y = np.ravel(df[target])
        self.scoring = scoring
        
        # 데이터 검증
        logger.info(f"입력 데이터 형태: X={self.X.shape}, y={self.y.shape}")
        logger.info(f"타겟 컬럼: {target}")
        logger.info(f"특성 컬럼: {list(self.X.columns)}")
        
        # 무한값과 NaN 처리
        self.X = self.X.replace([np.inf, -np.inf], np.nan)
        self.X = self.X.fillna(self.X.mean())
        self.y = np.nan_to_num(self.y, nan=np.nanmean(self.y))

    def run_automl(self):
        """FLAML AutoML 분류 모델을 실행합니다."""
        try:
            automl = AutoML()
            automl.fit(self.X, self.y, task="classification", metric=self.scoring)
            
            best_estimator = automl.model.estimator
            best_config = automl.best_config
            
            logger.info(f"최적 모델: {best_estimator}")
            logger.info(f"최적 설정: {best_config}")
            
            results = {
                'model_name': str(best_estimator),
                'best_score': float(automl.best_loss * -1 if 'neg_' in self.scoring else automl.best_loss),
                'best_config': best_config,
                'scoring_metric': self.scoring
            }
            
            # JSON 문자열로 직렬화
            import json
            results_json = json.dumps(results, ensure_ascii=False)
            
            return {'best': results_json}
            
        except Exception as e:
            logger.error(f"FLAML 분류 모델 학습 실패: {e}")
            import json
            return {'best': json.dumps({'error': str(e)})}

class FlamlRegression:
    def __init__(self, df, target, scoring='neg_mean_squared_error'):
        self.df = df
        self.target = target
        self.X = df.drop(target, axis=1)
        self.y = np.ravel(df[target])
        self.scoring = scoring
        
        # 데이터 검증
        logger.info(f"입력 데이터 형태: X={self.X.shape}, y={self.y.shape}")
        logger.info(f"타겟 컬럼: {target}")
        logger.info(f"특성 컬럼: {list(self.X.columns)}")
        
        # 무한값과 NaN 처리
        self.X = self.X.replace([np.inf, -np.inf], np.nan)
        self.X = self.X.fillna(self.X.mean())
        self.y = np.nan_to_num(self.y, nan=np.nanmean(self.y))

    def run_automl(self):
        """FLAML AutoML 회귀 모델을 실행합니다."""
        try:
            automl = AutoML()
            automl.fit(self.X, self.y, task="regression", metric=self.scoring)
            
            best_estimator = automl.model.estimator
            best_config = automl.best_config
            
            logger.info(f"최적 모델: {best_estimator}")
            logger.info(f"최적 설정: {best_config}")
            
            results = {
                'model_name': str(best_estimator),
                'best_score': float(automl.best_loss * -1 if 'neg_' in self.scoring else automl.best_loss),
                'best_config': best_config,
                'scoring_metric': self.scoring
            }
            
            # JSON 문자열로 직렬화
            import json
            results_json = json.dumps(results, ensure_ascii=False)
            
            return {'best': results_json}
            
        except Exception as e:
            logger.error(f"FLAML 회귀 모델 학습 실패: {e}")
            import json
            return {'best': json.dumps({'error': str(e)})}

class FlamlClustering:
    def __init__(self, df, scoring='silhouette'):
        self.df = df
        self.X = df
        self.scoring = scoring
        
        # 데이터 검증
        logger.info(f"입력 데이터 형태: X={self.X.shape}")
        logger.info(f"특성 컬럼: {list(self.X.columns)}")
        
        # 무한값과 NaN 처리
        self.X = self.X.replace([np.inf, -np.inf], np.nan)
        self.X = self.X.fillna(self.X.mean())

    def run_automl(self):
        """FLAML AutoML 군집화 모델을 실행합니다."""
        try:
            automl = AutoML()
            automl.fit(self.X, task="clustering", metric=self.scoring)
            
            best_estimator = automl.model.estimator
            best_config = automl.best_config
            
            logger.info(f"최적 모델: {best_estimator}")
            logger.info(f"최적 설정: {best_config}")
            
            results = {
                'model_name': str(best_estimator),
                'best_score': float(automl.best_loss * -1 if 'neg_' in self.scoring else automl.best_loss),
                'best_config': best_config,
                'scoring_metric': self.scoring
            }
            
            # JSON 문자열로 직렬화
            import json
            results_json = json.dumps(results, ensure_ascii=False)
            
            return {'best': results_json}
            
        except Exception as e:
            logger.error(f"FLAML 군집화 모델 학습 실패: {e}")
            import json
            return {'best': json.dumps({'error': str(e)})}
