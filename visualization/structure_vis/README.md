# CSV 데이터 증강 및 시각화 도구

CSV 파일을 업로드하여 데이터 증강을 수행하고 시각화할 수 있는 웹 애플리케이션입니다.

## 🏗️ 아키텍처

이 프로젝트는 **FastAPI 백엔드**와 **Streamlit 프론트엔드**로 구성되어 있습니다.

```
structure_vis/
├── backend/                 # FastAPI 백엔드
│   ├── main.py             # FastAPI 앱 진입점
│   ├── api/                # API 라우터
│   ├── services/           # 비즈니스 로직
│   ├── models/             # 데이터 모델
│   ├── utils/              # 유틸리티
│   └── requirements.txt    # 백엔드 의존성
├── frontend/               # Streamlit 프론트엔드
│   ├── structure_vis.py    # 메인 UI
│   └── requirements.txt    # 프론트엔드 의존성
└── README.md
```

## 🚀 시작하기

### 1. 백엔드 실행

```bash
cd structure_vis/backend
pip install -r requirements.txt
python main.py
```

백엔드는 `http://localhost:8000`에서 실행됩니다.

### 2. 프론트엔드 실행

```bash
cd structure_vis/frontend
pip install -r requirements.txt
streamlit run structure_vis.py
```

프론트엔드는 `http://localhost:8501`에서 실행됩니다.

## 📋 주요 기능

### 데이터 업로드 및 분석
- CSV 파일 업로드 (최대 300MB)
- 데이터 품질 분석
- 컬럼 타입 자동 감지
- 결측값 및 중복값 분석

### 데이터 증강
- **노이즈 추가**: 수치형 데이터에 랜덤 노이즈 추가
- **중복 생성**: 기존 데이터를 복제하여 증강
- **특성 증강**: 새로운 특성 생성
- **SMOTE**: 불균형 데이터 증강
- **일반 증강**: 다양한 증강 방법 조합

### 시각화
- **히스토그램**: 원본 vs 증강 데이터 분포 비교
- **박스플롯**: 데이터 분포 및 이상치 비교
- **산점도**: 두 변수 간 관계 비교
- **범주형 비교**: 카테고리별 분포 비교
- **통계 요약**: 증강 전후 통계 비교

## 🔌 API 엔드포인트

### 데이터 관련
- `POST /api/data/upload` - CSV 파일 업로드
- `GET /api/data/analyze/{session_id}` - 데이터 분석
- `GET /api/data/preview/{session_id}` - 데이터 미리보기
- `GET /api/data/download/{session_id}` - 데이터 다운로드

### 증강 관련
- `POST /api/augmentation/process` - 데이터 증강 실행
- `GET /api/augmentation/estimate-time/{session_id}` - 처리 시간 예측
- `GET /api/augmentation/summary/{session_id}` - 증강 결과 요약

### 시각화 관련
- `GET /api/visualization/histogram/{session_id}/{column}` - 히스토그램
- `GET /api/visualization/boxplot/{session_id}/{column}` - 박스플롯
- `GET /api/visualization/scatter/{session_id}` - 산점도
- `GET /api/visualization/categorical/{session_id}/{column}` - 범주형 비교

## ⚙️ 설정

### 파일 크기 제한
- 기본: 300MB
- `backend/utils/file_utils.py`에서 `validate_csv_file` 함수의 `max_size_mb` 파라미터 수정

### 세션 관리
- 메모리 기반 세션 스토리지 사용
- 24시간 후 자동 만료
- Redis 사용 가능 (프로덕션 환경 권장)

### 에러 처리
- 백엔드: 상세 로그를 서버 콘솔에 출력
- 프론트엔드: 사용자 친화적인 에러 메시지 표시

## 🛠️ 기술 스택

### 백엔드
- **FastAPI**: 고성능 웹 프레임워크
- **Pydantic**: 데이터 검증
- **Pandas**: 데이터 처리
- **Scikit-learn**: 머신러닝 (SMOTE)
- **Plotly**: 차트 생성

### 프론트엔드
- **Streamlit**: 웹 UI 프레임워크
- **Requests**: HTTP 클라이언트
- **Plotly**: 인터랙티브 차트

## 📊 사용자 경험

### 진행률 표시
- 파일 업로드 진행률
- 데이터 처리 진행률
- 예상 처리 시간 표시

### 상태 표시
- 단계별 성공/진행/대기 상태
- 색상 코딩 및 아이콘 사용
- 직관적인 피드백

### 취소 기능
- 처리 중단 버튼
- 세션 삭제 기능

## 🔧 개발

### 로컬 개발 환경 설정

1. **가상환경 생성**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. **의존성 설치**
```bash
# 백엔드
cd backend
pip install -r requirements.txt

# 프론트엔드
cd ../frontend
pip install -r requirements.txt
```

3. **실행**
```bash
# 백엔드 (터미널 1)
cd backend
python main.py

# 프론트엔드 (터미널 2)
cd frontend
streamlit run structure_vis.py
```

### API 문서
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 