# 📦 SOPO-Dashboard

## 생활물류 AI 기반 예측/관리 대시보드
![image](https://github.com/user-attachments/assets/7f20dc65-ff28-4e67-ade7-89f2d5430658)



서울시 생활물류 데이터를 기반으로 **MFC 관리자**가 현장 운영에 필요한 인사이트를 직관적으로 확인할 수 있도록 설계된 웹 기반 대시보드 시스템입니다.  
Streamlit을 활용한 대시보드를 FastAPI 서버를 통해 중계하고, <br>
Spring Boot 웹사이트에서 iframe으로 통합 제공함으로써 **AI 예측 기능을 가진 업무지원 도구**로 활용 가능합니다.

---

# 프로젝트 개요

- 📅 **데이터**: 6년간의 일별 물류센터 × 품목별 물동량 데이터 (전처리한 서울시 생활물류 공공 데이터)
- 🎯 **목표**: 
  - MFC 운영 효율화를 위한 EDA & ML 기반 수요 예측 기반 인사이트 제공
  - 이상치 탐지 및 원인 자동 분류 기능 포함
  - 관리자 친화적인 직관적인 UI 기반 실시간 대시보드 제공
- 🧱 **구성**: 데이터 탐색적 분석 + 예측 모델링 + 이상치 탐지 + 인사이트 도출

---

# 프로젝트 주요 기능

| 기능 | 설명 |
|------|------|
| 📊 데이터 요약 | 전체 물동량 흐름 요약, 품목별 평균/표준편차, 요일별 변화 등 통계적 요약 |
| 📈 품목 추이 분석 | 선택 품목의 기간별 추이 및 비중 시각화 (Plotly 이용하여 동적 시각화) |
| 📉 이상치 탐지 | Z-score 기반 탐지 + 공휴일 영향 여부 자동 판단 |
| 🧠 예측 모델링 | Prophet / LightGBM 기반 예측 및 결과 시각화 |
| 📊 성능 비교 | 예측 모델의 MAE/RMSE/R² 지표 및 결과 비교 |
| 📊 인사이트 대시보드 | 품목별/요일별 변화, 명절 전후 수요 변화 등 정량적 인사이트 제공 |
| ❌ 오차 분석 | 예측과 실제값 차이에 대한 원인(요일/명절/원인불명 등) 분류 |
| 🧩 시스템 통합 | FastAPI 기반 프록시 서버를 이용하여 SpringBoot 웹서비스와 iframe 연동 |
---


## 🛠 기술 스택

- **Main Stack**: Streamlit + Plotly + Matplotlib
- **Backend**: FastAPI + Spring Boot (추후 스프링부터 웹서비스와 연결 예정)
- **ML/통계**: Prophet, LightGBM, Scikit-learn
- **Infra**: Docker (Streamlit/FastAPI 컨테이너화)
- **ETL/Preprocessing**: Pandas, Numpy, Python datetime, holidays

---

## 📁 프로젝트 구조

.  <br>
├── data/  <br>
│ └── logistics_by_center.csv # 연결+전처리+그룹핑 완료된 데이터 (6년치)  <br>
├── pages/ # Streamlit 개별 기능 페이지 <br>
│ ├── anomaly_detection.py <br>
│ ├── center_comparison.py <br>
│ ├── data_summary.py <br>
│ ├── error_analysis.py <br>
│ ├── insight_dashboard.py <br>
│ ├── item_trend.py <br>
│ ├── lgbm_forecast.py <br>
│ ├── model_comparison.py <br>
│ ├── model_ranking.py <br>
│ └── prophet_forecast.py <br>
├── src/  <br>
│ ├── loader.py <br>
│ └── visualizer.py <br>
├── app.py # Streamlit 진입점 <br>
└── main.py # FastAPI 서버 (개발 진행 중) <br>

---
# 실행 방법

▶︎ 1. Streamlit 대시보드 실행
```bash
streamlit run app.py
```
▶︎ 2. FastAPI 중계 서버 실행
```bash
uvicorn main:app --port 8005 --reload
```
▶︎ 3. SpringBoot 웹에서 iframe 삽입
```html
<iframe src="http://localhost:8501" style="width:100%; height:1000px; border:none;"></iframe>
```
▶︎ Docker Image 이용한 실행
```bash
Docker Hub 배포 예정
```

---

# 향후 개선 방향

- MySQL 기반 데이터베이스 연결 (현재 csv 파일로부터 데이터 로드 -> 교체 염두해두고 확장 용이한 형태로 개발함)

- MLOps: 모델 재학습 파이프라인 개발 (현재 고정된 데이터 기반으로 학습됨)

- 성능 개선을 위한 비동기 로딩 및 API 최적화

- 관리자 설정 기반 필터링 옵션 정교화 (더 세부적으로 필터링 가능하도록)

- 주간 리포트 자동 생성 및 이메일 전송 기능

