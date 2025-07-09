# 필요한 라이브러리 import
import os
import glob
import time
import json
import psutil
import sqlite3
import logging
import subprocess
import pandas as pd
from tqdm import tqdm
from pathlib import Path


class DataProcessor:
    """데이터 처리와 관련된 기능을 담당하는 클래스"""
    
    @staticmethod
    def format_bytes(size):
        """byte를 KB, MB, GB, TB 등으로 변경하는 함수"""
        volum = 1024
        n = 0
        volum_labels = {0: 'B', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
        size = 0
        while size > volum:
            size /= volum
            n += 1
        return f"{size} {volum_labels[n]}"
    
    @staticmethod
    def normalize_json(json_meta_data):
        """json데이터를 pandas DataFrame로 변경하는 함수"""
        json_df = []
        for file in tqdm(json_meta_data):
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data_df = pd.json_normalize(data)
                json_df.append(data_df)
        return json_df
    
    def extract_image_metadata(self, img_paths):
        """이미지 데이터에서 메타 데이터 추출"""
        return pd.DataFrame({
            'file_id': [img_paths[n].split('.')[-2].split('/')[-1] for n in range(len(img_paths))],
            'full_path': img_paths,
            'folder': [img_paths[n].split('.')[-2].split('/')[-2] for n in range(len(img_paths))],
            'file_name': [path.split('/')[-1] for path in img_paths],
            'file_size': [os.path.getsize(path) for path in tqdm(img_paths)],
        })
    
    def extract_label_metadata(self, label_data):
        """라벨 데이터에서 메타 데이터 추출"""
        return pd.DataFrame({
            "file_id": [label.split('.')[-2].split('/')[-1] for label in label_data],
            "label_paths": label_data,
        })
    
    def process_data(self, data_dir='./data/'):
        """전체 데이터 처리 과정"""
        # 이미지 데이터 경로 수집
        img_dir_list = glob.glob(f'{data_dir}image/TS*')
        img_total_list = [glob.glob(folder+'/*') for folder in img_dir_list]
        img_paths = [img_path for sublist in img_total_list for img_path in sublist]
        
        # 이미지 메타 데이터 추출
        df = self.extract_image_metadata(img_paths)
        
        # 라벨 데이터 경로 수집 및 메타 데이터 추출
        label_data = glob.glob(f'{data_dir}label_data/*.json')
        label_df = self.extract_label_metadata(label_data)
        
        # JSON 데이터를 DataFrame으로 변환
        json_meta_data = self.normalize_json(label_df['label_paths'])
        concat_df = pd.concat(json_meta_data, ignore_index=True)
        
        # 데이터 선별 및 컬럼명 변경
        new_df = concat_df[['id', 'image.size.width', 'image.size.height', 
                           'dataset.quality1', 'dataset.quality2', 'dataset.quality3', 
                           'dataset.quality4', 'dataset.result']]
        new_df.columns = ['file_id', 'width', 'height', '밀도', '흡수율', '안전성', '잔입자_통과량', 'target_data']
        
        # 데이터 병합
        merge_df = pd.merge(df, new_df, on='file_id')
        updated_df = merge_df.drop('file_id', axis=1)
        
        return updated_df, merge_df
    
    def get_file_type_counts(self, merge_df):
        """파일 타입별 개수 계산"""
        img_type_data = [data.split('.')[-1] for data in merge_df['full_path']]
        
        png_num = sum(1 for img_type in img_type_data if img_type == 'png')
        jpg_num = sum(1 for img_type in img_type_data if img_type in ['jpg', 'jpeg'])
        etc = sum(1 for img_type in img_type_data if img_type not in ['png', 'jpg', 'jpeg'])
        
        return png_num, jpg_num, etc


class SystemMonitor:
    """시스템 리소스 모니터링을 담당하는 클래스"""
    
    def __init__(self, data_processor):
        self.data_processor = data_processor
    
    def log_system_resources(self):
        """불필요한 자원이 사용되고 있는지 확인하는 함수"""
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk_usage = psutil.disk_usage('/')
        
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        system_info = f"""
        사용 중인 자원 확인:
        --------------------------------------
        전체 메모리: {self.data_processor.format_bytes(memory_info.total)} 
        사용 가능한 메모리: {self.data_processor.format_bytes(memory_info.available)} 
        사용된 메모리: {self.data_processor.format_bytes(memory_info.used)} 
        메모리 사용 퍼센트: {memory_info.percent}%
        CPU 사용 퍼센트: {cpu_percent}%
        전체 디스크 용량: {self.data_processor.format_bytes(disk_usage.total)} 
        사용된 디스크 용량: {self.data_processor.format_bytes(disk_usage.used)} 
        디스크 사용 퍼센트: {disk_usage.percent}%
        --------------------------------------
        """
        logging.info(system_info)


class DatabaseManager:
    """데이터베이스 관리를 담당하는 클래스"""
    
    def __init__(self, db_path='./database/database.db'):
        self.db_path = db_path
    
    def save_data(self, df, table_name='TB_meta_info'):
        """데이터프레임을 데이터베이스에 저장"""
        conn = sqlite3.connect(self.db_path)
        try:
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            logging.info('데이터베이스에 데이터 저장 완료')
        finally:
            conn.close()
            logging.info('데이터베이스 연결 종료')


class Logger:
    """로깅 기능을 담당하는 클래스"""
    
    def __init__(self, log_file='./log/prepro.log', level=logging.INFO):
        self.log_file = log_file
        self.setup_logging(level)
    
    def setup_logging(self, level):
        """로깅 설정"""
        logging.basicConfig(
            filename=self.log_file, 
            level=level, 
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def log_directory_info(self, data_dir='./data/'):
        """디렉토리 정보 로깅"""
        try:
            data_path = subprocess.run(['tree', '-d', data_dir], text=True, capture_output=True, check=True)
            if data_path.returncode == 0:
                logging.info("데이터 디렉토리 확인:\n%s", data_path.stdout)
        except subprocess.CalledProcessError:
            logging.warning("tree 명령어 실행 실패")
    
    def log_total_directory(self, current_directory=None):
        """전체 디렉토리 구조 로깅"""
        if current_directory is None:
            current_directory = Path.cwd()
        
        try:
            total_dir = subprocess.run(['tree', '-d', current_directory], text=True, capture_output=True, check=True)
            if total_dir.returncode == 0:
                logging.info("전체 디렉토리 구조:\n%s", total_dir.stdout)
        except subprocess.CalledProcessError:
            logging.warning("tree 명령어 실행 실패")
    
    def log_processing_info(self, merge_df, png_num, jpg_num, etc):
        """데이터 처리 정보 로깅"""
        file_size = sum(merge_df['file_size'])
        file_info = f"""
        데이터 처리 정보:
        --------------------------------------
        전체 이미지 데이터 수: {len(merge_df)}
        전체 이미지 데이터 용량: {DataProcessor.format_bytes(file_size)}
        png counts: {png_num} 
        jpg counts: {jpg_num}
        etc counts: {etc}
        --------------------------------------
        """
        logging.info(file_info)


class DataProcessingPipeline:
    """전체 데이터 처리 파이프라인을 관리하는 메인 클래스"""
    
    def __init__(self, data_dir='./data/', log_file='./log/prepro.log', db_path='./database/database.db'):
        self.data_dir = data_dir
        self.start_time = time.time()
        
        # 컴포넌트 초기화
        self.data_processor = DataProcessor()
        self.system_monitor = SystemMonitor(self.data_processor)
        self.database_manager = DatabaseManager(db_path)
        self.logger = Logger(log_file)
    
    def run(self):
        """전체 데이터 처리 파이프라인 실행"""
        logging.info('데이터 처리 시작')
        self.system_monitor.log_system_resources()
        
        # 디렉토리 정보 로깅
        self.logger.log_directory_info(self.data_dir)
        
        logging.info('데이터 로딩')
        logging.info('메타 데이터 추출 시작..')
        
        # 데이터 처리
        updated_df, merge_df = self.data_processor.process_data(self.data_dir)
        
        logging.info('메타 데이터 추출 완료')
        logging.info('테이블 생성 시작..')
        logging.info('데이터 테이블 생성 완료')
        logging.info(f"Table row: {len(updated_df)}, Table columns: {len(updated_df.columns)}")
        
        # 파일 타입별 개수 계산
        png_num, jpg_num, etc = self.data_processor.get_file_type_counts(merge_df)
        
        # 처리 정보 로깅
        self.logger.log_processing_info(merge_df, png_num, jpg_num, etc)
        
        # 데이터베이스 저장
        logging.info('데이터베이스 생성')
        if Path(self.database_manager.db_path).exists():
            logging.info("데이터베이스 경로: %s", self.database_manager.db_path)
        
        self.database_manager.save_data(updated_df)
        
        # 전체 디렉토리 정보 로깅
        self.logger.log_total_directory()
        
        # 실행 시간 계산 및 로깅
        self._log_execution_time()
        logging.info('데이터 처리 종료')
    
    def _log_execution_time(self):
        """실행 시간 계산 및 로깅"""
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        minutes, seconds = divmod(elapsed_time, 60)
        logging.info("경과 시간: {}분 {}초".format(int(minutes), int(seconds)))


def main():
    """메인 실행 함수"""
    pipeline = DataProcessingPipeline()
    pipeline.run()


if __name__ == "__main__":
    main() 