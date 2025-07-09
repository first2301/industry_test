import polars as pl
from PIL import Image
import psutil
import time
import os
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import signal
import functools


def timeout_handler(signum, frame):
    """타임아웃 핸들러"""
    raise TimeoutError("파일 처리 시간 초과")


def timeout(seconds):
    """타임아웃 데코레이터"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Windows에서는 signal.SIGALRM이 지원되지 않으므로 다른 방식 사용
            if os.name == 'nt':  # Windows
                import threading
                result = [None]
                exception = [None]
                
                def target():
                    try:
                        result[0] = func(*args, **kwargs)
                    except Exception as e:
                        exception[0] = e
                
                thread = threading.Thread(target=target)
                thread.daemon = True
                thread.start()
                thread.join(seconds)
                
                if thread.is_alive():
                    # 스레드가 여전히 실행 중이면 타임아웃
                    raise TimeoutError(f"함수 실행 시간 초과 ({seconds}초)")
                
                if exception[0]:
                    raise exception[0]
                
                return result[0]
            else:  # Unix/Linux
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(seconds)
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                return result
        return wrapper
    return decorator


class DataProcessor:
    """데이터 처리와 관련된 기능을 담당하는 클래스"""
    
    @staticmethod
    def format_bytes(size: int) -> str:
        """byte를 KB, MB, GB, TB 등으로 변경하는 함수"""
        volum = 1024
        n = 0
        volum_labels = {0: 'B', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
        original_size = size
        while original_size >= volum and n < len(volum_labels) - 1:
            original_size /= volum
            n += 1
        return f"{original_size:.1f} {volum_labels[n]}"
    
    @staticmethod
    def get_directory_info(directory: str, summary_only: bool = False) -> Optional[str]:
        """디렉토리 정보를 가져오는 함수 (간단한 요약만)"""
        try:
            path = Path(directory)
            if not path.exists():
                return f"디렉토리가 존재하지 않습니다: {directory}"
            
            if summary_only:
                # 빠른 요약(폴더/파일 개수만)
                file_count = 0
                dir_count = 0
                for root, dirs, files in os.walk(path):
                    dir_count += len(dirs)
                    file_count += len(files)
                return f"[요약] 폴더 {dir_count}개, 파일 {file_count}개"
            else:
                # 전체 디렉토리 구조 (개선된 형태)
                result = f"디렉토리 구조: {directory}\n"
                result += "=" * 80 + "\n"
                
                # 최대 깊이 제한 (너무 깊은 구조 방지)
                max_depth = 5
                max_files_per_dir = 10
                
                for root, dirs, files in os.walk(path):
                    # 깊이 계산
                    depth = root.replace(str(path), '').count(os.sep)
                    if depth > max_depth:
                        continue
                    
                    # 들여쓰기
                    indent = '  ' * depth
                    dir_name = os.path.basename(root) if root != str(path) else os.path.basename(path)
                    
                    # 디렉토리 표시
                    if depth == 0:
                        result += f"{indent}📁 {dir_name}/\n"
                    else:
                        result += f"{indent}📁 {dir_name}/\n"
                    
                    # 파일 표시 (처음 10개만)
                    if files:
                        for i, file in enumerate(files[:max_files_per_dir]):
                            file_size = ""
                            try:
                                file_path = os.path.join(root, file)
                                if os.path.exists(file_path):
                                    size = os.path.getsize(file_path)
                                    if size > 1024 * 1024:  # 1MB 이상
                                        file_size = f" ({DataProcessor.format_bytes(size)})"
                            except:
                                pass
                            result += f"{indent}  📄 {file}{file_size}\n"
                        
                        if len(files) > max_files_per_dir:
                            result += f"{indent}  ... ({len(files) - max_files_per_dir}개 더)\n"
                    
                    # 하위 디렉토리 수 표시
                    if dirs:
                        result += f"{indent}  📂 하위 디렉토리: {len(dirs)}개\n"
                
                result += "=" * 80 + "\n"
                return result
        except Exception as e:
            logging.error(f"디렉토리 정보 수집 실패: {e}")
            return None
    
    @staticmethod
    def split_file_list(file_paths: List[Path], batch_size: int) -> List[List[Path]]:
        """파일 리스트를 배치로 분할"""
        batches = []
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            batches.append(batch)
        return batches
    
    @staticmethod
    def process_batch(batch: List[Path], batch_num: int, total_batches: int, 
                     max_workers: int = 4) -> Tuple[pd.DataFrame, str]:
        """단일 배치 처리 (로깅 최적화)"""
        batch_start_time = time.time()
        
        # 배치 메타데이터 추출 (배치 정보 전달)
        metadata_list = DataProcessor.extract_file_metadata(
            batch, max_workers, batch_num, total_batches
        )
        
        # DataFrame 생성
        df = pl.DataFrame(metadata_list)
        pandas_df = df.to_pandas()
        
        # 배치 처리 시간 계산
        batch_end_time = time.time()
        batch_elapsed = batch_end_time - batch_start_time
        
        # 배치 정보 생성
        batch_size_bytes = sum(pandas_df['file_size'])
        batch_info = (f"배치 {batch_num}/{total_batches}: {len(batch)}개 파일, "
                     f"{DataProcessor.format_bytes(batch_size_bytes)}, "
                     f"처리시간: {batch_elapsed:.1f}초")
        
        return pandas_df, batch_info
    
    @staticmethod
    @timeout(30)  # 30초 타임아웃
    def _extract_single_image_resolution(file_path: Path) -> Tuple[int, int]:
        """단일 이미지 해상도 추출 (병렬 처리용)"""
        try:
            with Image.open(file_path) as img:
                return img.size
        except Exception as e:
            logging.warning(f"이미지 해상도 추출 실패 {file_path}: {e}")
            return (0, 0)
    
    @staticmethod
    def extract_file_metadata(file_paths: List[Path], max_workers: int = 4, 
                             batch_num: int = 0, total_batches: int = 0) -> List[Dict[str, Any]]:
        """파일 메타데이터 추출 - 기본 정보 (배치 처리 최적화)"""
        metadata_list = []
        total_files = len(file_paths)
        
        # 배치 내부에서는 로깅하지 않음 (전체 진행률만 표시)
        for i, path in enumerate(file_paths):
            try:
                # 기본 정보 추출
                metadata = {
                    'full_path': str(path),
                    'file_id': path.stem,
                    'folder_name': path.parent.name,
                    'file_size': path.stat().st_size if path.exists() else 0,
                    'file_type': path.suffix.lower(),
                }
                
                metadata_list.append(metadata)
            except Exception as e:
                logging.warning(f"파일 메타데이터 추출 실패 {path}: {e}")
                # 실패한 경우에도 기본 정보는 포함
                metadata = {
                    'full_path': str(path),
                    'file_id': path.stem,
                    'folder_name': path.parent.name,
                    'file_size': 0,
                }
                metadata_list.append(metadata)
        
        return metadata_list
    
    @staticmethod
    def extract_image_resolutions(file_paths: List[Path], max_workers: int = 4,
                                 batch_num: int = 0, total_batches: int = 0) -> List[Tuple[int, int]]:
        """이미지 해상도 추출 (width, height) - 병렬 처리 적용 (배치 처리 최적화)"""
        resolutions = []
        total_files = len(file_paths)
        
        # 파일 수가 적으면 순차 처리
        if len(file_paths) < 10:
            for i, path in enumerate(file_paths):
                resolutions.append(DataProcessor._extract_single_image_resolution(path))
        else:
            # 병렬 처리
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_path = {executor.submit(DataProcessor._extract_single_image_resolution, path): path 
                                for path in file_paths}
                
                for i, future in enumerate(as_completed(future_to_path)):
                    resolutions.append(future.result())
        
        return resolutions
    
    @staticmethod
    def make_polars_dataframe(file_paths: List[Path], max_workers: int = 4) -> pl.DataFrame:
        """polars DataFrame 생성 - 기본 정보 + 이미지 해상도"""
        logging.info(f"총 {len(file_paths)}개 파일 처리 시작")
        
        # 기본 메타데이터 추출
        metadata_list = DataProcessor.extract_file_metadata(file_paths, max_workers)
        
        # 메타데이터를 DataFrame으로 변환
        df = pl.DataFrame(metadata_list)
        
        # 컬럼 순서 정리
        df = df.select(['full_path', 'file_id', 'folder_name', 'file_size', 'image_width', 'image_height'])
        
        logging.info(f"DataFrame 생성 완료: {len(df)} 행, {len(df.columns)} 열")
        return df
    
    @staticmethod
    def make_pandas_dataframe(file_paths: List[Path], max_workers: int = 4) -> pd.DataFrame:
        """pandas DataFrame 생성"""
        df = DataProcessor.make_polars_dataframe(file_paths, max_workers)
        return df.to_pandas()
    
    @staticmethod
    def log_system_resources() -> str:
        """시스템 리소스 정보를 수집하고 반환하는 함수"""
        try:
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Windows에서는 루트 디스크 대신 현재 작업 디렉토리 사용
            current_dir = Path.cwd()
            disk_usage = psutil.disk_usage(str(current_dir))
            
            # G 드라이브 정보 추가
            g_drive_info = ""
            try:
                g_disk_usage = psutil.disk_usage('G:\\')
                g_drive_info = f"""
            G 드라이브 정보:
            --------------------------------------
            G 드라이브 전체 용량: {DataProcessor.format_bytes(g_disk_usage.total)} 
            G 드라이브 사용된 용량: {DataProcessor.format_bytes(g_disk_usage.used)} 
            G 드라이브 사용 퍼센트: {g_disk_usage.percent}%
            G 드라이브 여유 용량: {DataProcessor.format_bytes(g_disk_usage.free)}
            --------------------------------------"""
            except Exception as e:
                g_drive_info = f"""
            G 드라이브 정보:
            --------------------------------------
            G 드라이브 접근 불가: {e}
            --------------------------------------"""
            
            current_time = time.strftime('%Y-%m-%d %H:%M:%S') # 현재 시간 기록
            # 시스템 리소스 정보를 표 형태로 생성
            system_info = f"""
            사용 중인 자원 확인:
            --------------------------------------
            전체 메모리: {DataProcessor.format_bytes(memory_info.total)} 
            사용 가능한 메모리: {DataProcessor.format_bytes(memory_info.available)} 
            사용된 메모리: {DataProcessor.format_bytes(memory_info.used)} 
            메모리 사용 퍼센트: {memory_info.percent}%
            CPU 사용 퍼센트: {cpu_percent}%
            현재 디스크 용량: {DataProcessor.format_bytes(disk_usage.total)} 
            현재 사용된 디스크 용량: {DataProcessor.format_bytes(disk_usage.used)} 
            현재 디스크 사용 퍼센트: {disk_usage.percent}%{g_drive_info}
            """
            return system_info
        except Exception as e:
            logging.error(f"시스템 리소스 정보 수집 실패: {e}")
            return "시스템 리소스 정보 수집 실패"
    
    @staticmethod
    def setup_logging(log_file: str = './log/prepro.log', level: int = logging.INFO) -> None:
        """로깅 설정을 초기화하는 함수"""
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)  # log 디렉토리 생성
            
            # 기존 핸들러 제거
            for handler in logging.getLogger().handlers[:]:
                logging.getLogger().removeHandler(handler)
            
            # 파일 핸들러 설정
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            
            # 콘솔 핸들러 설정 (진행률은 INFO 레벨 이상만)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            
            # 루트 로거 설정
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.DEBUG)  # 모든 레벨 허용
            root_logger.addHandler(file_handler)
            root_logger.addHandler(console_handler)
            
        except Exception as e:
            print(f"로깅 설정 실패: {e}")
    
    @staticmethod
    def log_progress(message: str, level: int = logging.INFO, force_console: bool = False):
        """진행률 로깅 (콘솔 출력 제어)"""
        if force_console:
            # 강제로 콘솔에 출력 (flush=True로 즉시 출력)
            print(f"[진행률] {message}", flush=True)
        else:
            logging.log(level, message)
    
    @staticmethod
    def generate_file_info(pandas_df: pd.DataFrame) -> str:
        """파일 정보 요약 생성"""
        try:
            file_size = sum(pandas_df['file_size'])
            
            file_info = f"""
            데이터 처리 정보:
            --------------------------------------
            전체 파일 수: {len(pandas_df)}
            전체 파일 용량: {DataProcessor.format_bytes(file_size)}
            --------------------------------------
            """
            return file_info
        except Exception as e:
            logging.error(f"파일 정보 생성 실패: {e}")
            return "파일 정보 생성 실패"
    
    @staticmethod
    def save_to_database(pandas_df: pd.DataFrame, db_path: str = './database/database.db') -> str:
        """데이터베이스에 데이터 저장"""
        import sqlite3
        
        try:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)  # database 디렉토리 생성
            conn = sqlite3.connect(db_path)
            
            # 데이터 베이스에 데이터 저장
            pandas_df.to_sql('TB_meta_info', conn, if_exists='replace', index=False)
            
            conn.close()
            logging.info(f"데이터베이스 저장 완료: {db_path}")
            return db_path
        except Exception as e:
            logging.error(f"데이터베이스 저장 실패: {e}")
            raise
    
    @staticmethod
    def append_to_database(pandas_df: pd.DataFrame, db_path: str = './database/database.db', 
                          table_name: str = 'TB_meta_info') -> str:
        """데이터베이스에 데이터 추가 (기존 데이터 유지)"""
        import sqlite3
        
        try:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)  # database 디렉토리 생성
            conn = sqlite3.connect(db_path)
            
            # 데이터베이스에 데이터 추가
            pandas_df.to_sql(table_name, conn, if_exists='append', index=False)
            
            conn.close()
            # 로그 제거 - 너무 많은 로그 출력 방지
            return db_path
        except Exception as e:
            logging.error(f"데이터베이스 추가 실패: {e}")
            raise
    
    @staticmethod
    def validate_data_path(data_path: str) -> bool:
        """데이터 경로 유효성 검사"""
        path = Path(data_path)
        if not path.exists():
            logging.error(f"데이터 경로가 존재하지 않습니다: {data_path}")
            return False
        if not path.is_dir():
            logging.error(f"데이터 경로가 디렉토리가 아닙니다: {data_path}")
            return False
        return True
    
    @staticmethod
    def process_data_pipeline(data_path: str, output_db_path: str = './database/database.db', 
                            max_workers: int = 4, batch_size: int = 1000) -> Tuple[pd.DataFrame, str, str]:
        """전체 데이터 처리 파이프라인 (배치 처리 지원, 로깅 최적화)"""
        pipeline_start_time = time.time()
        
        try:
            # 데이터 경로 유효성 검사
            if not DataProcessor.validate_data_path(data_path):
                raise ValueError(f"유효하지 않은 데이터 경로: {data_path}")
            
            logging.info(f"데이터 처리 시작: {data_path}")
            
            # 데이터 로딩
            all_path = Path(data_path).rglob('*')
            file_only = [i for i in all_path if i.is_file()]
            
            if not file_only:
                logging.warning(f"처리할 파일이 없습니다: {data_path}")
                return pd.DataFrame(), "처리할 파일이 없습니다", output_db_path
            
            logging.info(f"총 {len(file_only)}개 파일 발견")
            
            # 배치 크기 결정 (메모리 사용량 고려)
            if batch_size <= 0:
                batch_size = 1000  # 기본값
            
            # 배치로 분할
            batches = DataProcessor.split_file_list(file_only, batch_size)
            total_batches = len(batches)
            logging.info(f"총 {total_batches}개 배치로 분할 (배치 크기: {batch_size})")
            
            # 진행률 로깅 간격 결정 (배치 수에 따라 적응적 조정)
            if total_batches <= 100:
                progress_interval = 1  # 100개 이하면 매 배치마다
            elif total_batches <= 1000:
                progress_interval = 5  # 1000개 이하면 5배치마다
            elif total_batches <= 5000:
                progress_interval = 25  # 5000개 이하면 25배치마다
            else:
                progress_interval = 50  # 5000개 이상이면 50배치마다
            
            # 진행률 로깅을 위한 변수
            last_logged_batch = 0
            
            # 첫 번째 배치로 데이터베이스 초기화
            if batches:
                first_batch_df, first_batch_info = DataProcessor.process_batch(
                    batches[0], 1, total_batches, max_workers
                )
                DataProcessor.save_to_database(first_batch_df, output_db_path)
                
                # 첫 번째 배치 완료 로그
                current_progress = (1 / total_batches) * 100
                logging.info(f"첫 번째 배치 완료, 진행률 로그 호출 예정: {current_progress:.1f}%")
                DataProcessor.log_progress(f"전체 진행률: {current_progress:.1f}% (1/{total_batches} 배치)", force_console=True)
                last_logged_batch = 1
                
                # 나머지 배치들 처리
                all_dfs = [first_batch_df]
                batch_infos = [first_batch_info]
                
                # 전체 진행률 로깅 (적응적 간격)
                for i, batch in enumerate(batches[1:], 2):
                    batch_df, batch_info = DataProcessor.process_batch(
                        batch, i, total_batches, max_workers
                    )
                    DataProcessor.append_to_database(batch_df, output_db_path)
                    all_dfs.append(batch_df)
                    batch_infos.append(batch_info)
                    
                    # 진행률 로깅 (적응적 간격)
                    if i >= last_logged_batch + progress_interval:
                        current_progress = (i / total_batches) * 100
                        logging.info(f"진행률 로그 호출: {current_progress:.1f}% ({i}/{total_batches})")
                        DataProcessor.log_progress(f"전체 진행률: {current_progress:.1f}% ({i}/{total_batches} 배치)", force_console=True)
                        last_logged_batch = i
                
                # 마지막 배치 완료 로그
                if last_logged_batch < total_batches:
                    final_progress = (total_batches / total_batches) * 100
                    DataProcessor.log_progress(f"전체 진행률: {final_progress:.1f}% ({total_batches}/{total_batches} 배치)", force_console=True)
                
                # 전체 DataFrame 결합
                pandas_df = pd.concat(all_dfs, ignore_index=True)
                
                # 전체 처리 시간 계산
                pipeline_end_time = time.time()
                # total_elapsed = pipeline_end_time - pipeline_start_time
                
                # 파일 정보 생성
                file_info = DataProcessor.generate_file_info(pandas_df)
                # file_info += f"\n전체 처리 시간: {total_elapsed:.1f}초"
                
                # 배치 처리 요약 정보만 표시
                # total_files_processed = sum(len(batch) for batch in batches)
                # total_size_processed = sum(pandas_df['file_size'])
                # avg_batch_time = total_elapsed / total_batches
                
                # file_info += f"\n\n배치 처리 요약:"
                # file_info += f"\n- 총 배치 수: {total_batches}개"
                # file_info += f"\n- 총 파일 수: {total_files_processed:,}개"
                # file_info += f"\n- 총 용량: {DataProcessor.format_bytes(total_size_processed)}"
                # file_info += f"\n- 평균 배치 처리 시간: {avg_batch_time:.1f}초"
                # file_info += f"\n- 처리 속도: {total_files_processed/total_elapsed:.0f} 파일/초"
                
                # logging.info(f"배치 데이터 처리 파이프라인 완료 (총 {total_elapsed:.1f}초)")
                return pandas_df, file_info, output_db_path
            else:
                return pd.DataFrame(), "처리할 배치가 없습니다", output_db_path
            
        except Exception as e:
            logging.error(f"데이터 처리 파이프라인 실패: {e}")
            raise