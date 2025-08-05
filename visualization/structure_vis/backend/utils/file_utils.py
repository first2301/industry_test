import pandas as pd
import io
import uuid
import time
from typing import Optional, Dict, Any
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 메모리 기반 세션 스토리지 (실제 프로덕션에서는 Redis 사용 권장)
session_storage = {}

def generate_session_id() -> str:
    """고유한 세션 ID를 생성합니다."""
    return str(uuid.uuid4())

def store_data_in_session(session_id: str, data: pd.DataFrame, metadata: Dict[str, Any] = None) -> bool:
    """세션에 데이터를 저장합니다."""
    try:
        session_storage[session_id] = {
            'data': data,
            'metadata': metadata or {},
            'created_at': time.time(),
            'last_accessed': time.time()
        }
        logger.info(f"Data stored in session: {session_id}")
        return True
    except Exception as e:
        logger.error(f"Error storing data in session {session_id}: {e}")
        return False

def get_data_from_session(session_id: str) -> Optional[pd.DataFrame]:
    """세션에서 데이터를 가져옵니다."""
    try:
        if session_id in session_storage:
            session_storage[session_id]['last_accessed'] = time.time()
            return session_storage[session_id]['data']
        return None
    except Exception as e:
        logger.error(f"Error retrieving data from session {session_id}: {e}")
        return None

def get_session_metadata(session_id: str) -> Optional[Dict[str, Any]]:
    """세션 메타데이터를 가져옵니다."""
    try:
        if session_id in session_storage:
            return session_storage[session_id]['metadata']
        return None
    except Exception as e:
        logger.error(f"Error retrieving metadata from session {session_id}: {e}")
        return None

def update_session_metadata(session_id: str, metadata: Dict[str, Any]) -> bool:
    """세션 메타데이터를 업데이트합니다."""
    try:
        if session_id in session_storage:
            session_storage[session_id]['metadata'].update(metadata)
            session_storage[session_id]['last_accessed'] = time.time()
            return True
        return False
    except Exception as e:
        logger.error(f"Error updating metadata for session {session_id}: {e}")
        return False

def delete_session(session_id: str) -> bool:
    """세션을 삭제합니다."""
    try:
        if session_id in session_storage:
            del session_storage[session_id]
            logger.info(f"Session deleted: {session_id}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        return False

def cleanup_expired_sessions(max_age_hours: int = 24) -> int:
    """만료된 세션들을 정리합니다."""
    current_time = time.time()
    expired_sessions = []
    
    for session_id, session_data in session_storage.items():
        if current_time - session_data['last_accessed'] > max_age_hours * 3600:
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        delete_session(session_id)
    
    logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    return len(expired_sessions)

def validate_csv_file(file_content: bytes, max_size_mb: int = 300) -> Dict[str, Any]:
    """CSV 파일을 검증합니다."""
    try:
        # 파일 크기 검증
        file_size_mb = len(file_content) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return {
                'valid': False,
                'error': f'파일 크기가 {max_size_mb}MB를 초과합니다. (현재: {file_size_mb:.1f}MB)'
            }
        
        # CSV 파싱 테스트
        try:
            df = pd.read_csv(io.BytesIO(file_content))
            if df.empty:
                return {
                    'valid': False,
                    'error': '빈 CSV 파일입니다.'
                }
            
            return {
                'valid': True,
                'row_count': len(df),
                'column_count': len(df.columns),
                'file_size_mb': file_size_mb
            }
        except Exception as e:
            return {
                'valid': False,
                'error': f'CSV 파일 파싱 오류: {str(e)}'
            }
            
    except Exception as e:
        logger.error(f"Error validating CSV file: {e}")
        return {
            'valid': False,
            'error': '파일 검증 중 오류가 발생했습니다.'
        }

def get_memory_usage_mb() -> float:
    """현재 메모리 사용량을 MB 단위로 반환합니다."""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024) 