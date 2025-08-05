# 데이터 시각화 및 증강 도구

이 프로젝트는 데이터 시각화와 증강 기능을 제공하는 웹 애플리케이션입니다. FastAPI 백엔드와 Streamlit 프론트엔드로 구성되어 있습니다.

## 🏗️ 프로젝트 구조

```
structure_vis/
├── backend/                    # FastAPI 백엔드
│   ├── main.py                # 메인 애플리케이션
│   ├── requirements.txt       # 백엔드 의존성
│   ├── README.md             # 백엔드 문서
│   ├── api/                  # API 엔드포인트
│   │   ├── __init__.py
│   │   ├── visualization_api.py
│   │   └── data_augmentation_api.py
│   ├── services/             # 서비스 레이어
│   │   ├── __init__.py
│   │   ├── visualization_service.py
│   │   └── data_augmentation_service.py
│   └── lib/                  # 핵심 라이브러리
│       ├── __init__.py
│       ├── visualization.py
│       ├── data_augmentation.py
│       └── data_utils.py
├── frontend/                  # Streamlit 프론트엔드
│   ├── app.py               # 메인 애플리케이션
│   ├── requirements.txt     # 프론트엔드 의존성
│   └── structure_vis.py     # 기존 애플리케이션 (참고용)
├── run_app.py               # 통합 실행 스크립트
└── README.md               # 프로젝트 문서
```

## 🚀 빠른 시작

### 1. 통합 실행 (권장)

```bash
cd structure_vis
python run_app.py
```

실행 스크립트에서 다음 옵션 중 선택할 수 있습니다:
- **1**: 백엔드만 실행 (FastAPI 서버)
- **2**: 프론트엔드만 실행 (Streamlit 앱)
- **3**: 백엔드 + 프론트엔드 동시 실행 (권장)

### 2. 개별 실행

#### 백엔드 실행
```bash
cd structure_vis/backend
pip install -r requirements.txt
python main.py
```

#### 프론트엔드 실행
```bash
cd structure_vis/frontend
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 접속 정보

- **백엔드 API**: http://localhost:8000
- **API 문서**: http://localhost:8000/docs
- **프론트엔드**: http://localhost:8501

## 📊 주요 기능

### 데이터 증강
- **노이즈 추가**: 수치형 데이터에 랜덤 노이즈 추가
- **중복 생성**: 데이터 행 복제
- **SMOTE**: 불균형 데이터 처리
- **조합 증강**: 여러 방법을 조합한 증강

### 데이터 시각화
- **히스토그램 비교**: 원본 vs 증강 데이터 분포 비교
- **박스플롯 비교**: 통계적 분포 비교
- **산점도 비교**: 두 변수 간 관계 비교
- **범주형 차트**: 막대그래프, 파이차트 등

### 데이터 분석
- **컬럼 타입 분석**: 수치형/범주형 자동 분류
- **데이터 미리보기**: 업로드된 데이터 확인
- **증강 통계**: 증강 전후 데이터 통계

## 🔧 기술 스택

### 백엔드
- **FastAPI**: 고성능 웹 프레임워크
- **Pandas**: 데이터 처리
- **NumPy**: 수치 계산
- **Plotly**: 시각화 생성
- **Scikit-learn**: 머신러닝 (SMOTE 등)

### 프론트엔드
- **Streamlit**: 웹 애플리케이션 프레임워크
- **Plotly**: 인터랙티브 차트
- **Requests**: HTTP 클라이언트

## 📋 API 엔드포인트

### 시각화 API (`/visualization`)
- `POST /upload-data`: 데이터 파일 업로드
- `POST /get-column-types`: 컬럼 타입 분석
- `POST /create-histogram-comparison`: 히스토그램 비교
- `POST /create-boxplot-comparison`: 박스플롯 비교
- `POST /create-scatter-comparison`: 산점도 비교
- `POST /create-categorical-chart`: 범주형 차트
- `POST /create-comparison-dashboard`: 비교 대시보드

### 데이터 증강 API (`/augmentation`)
- `GET /methods`: 사용 가능한 증강 방법
- `POST /validate-params`: 파라미터 검증
- `POST /preview`: 증강 결과 미리보기
- `POST /augment`: 데이터 증강 실행
- `POST /batch-augment`: 배치 증강

## 💡 사용 예시

### 1. 데이터 업로드
웹 인터페이스에서 CSV 파일을 업로드하거나, API를 직접 호출할 수 있습니다.

### 2. 데이터 증강
```python
import requests

# 증강 파라미터 설정
params = {
    "noise_level": 0.05,
    "augmentation_ratio": 0.5,
    "dup_count": 2
}

# 데이터 증강 실행
response = requests.post(
    "http://localhost:8000/augmentation/augment",
    json={
        "data": your_data,
        "method": "조합 증강",
        "parameters": params
    }
)
```

### 3. 시각화 생성
```python
# 히스토그램 비교 차트 생성
response = requests.post(
    "http://localhost:8000/visualization/create-histogram-comparison",
    json={
        "original_data": original_data,
        "augmented_data": augmented_data,
        "column": "feature_name"
    }
)
```

## 🔍 아키텍처

### 계층 구조
1. **API Layer**: HTTP 엔드포인트 제공
2. **Service Layer**: 비즈니스 로직 처리
3. **Library Layer**: 핵심 기능 구현

### 데이터 흐름
```
Frontend (Streamlit) → Backend API → Service Layer → Library Layer
```

## 🛠️ 개발 환경 설정

### 필수 요구사항
- Python 3.8+
- pip

### 의존성 설치
```bash
# 백엔드 의존성
cd structure_vis/backend
pip install -r requirements.txt

# 프론트엔드 의존성
cd structure_vis/frontend
pip install -r requirements.txt
```

## 🐛 문제 해결

### 백엔드 연결 오류
1. 백엔드 서버가 실행 중인지 확인
2. 포트 8000이 사용 가능한지 확인
3. CORS 설정 확인

### 의존성 오류
1. Python 버전 확인 (3.8+)
2. pip 업그레이드: `pip install --upgrade pip`
3. 가상환경 사용 권장

## 📝 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 지원

문제가 발생하거나 질문이 있으시면 이슈를 생성해주세요. 