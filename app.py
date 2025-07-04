# 📌 app.py - MFC 관리자용 홈 페이지 with 서비스 상세 설명

import streamlit as st
from datetime import date

# 페이지 기본 설정
st.set_page_config(
    page_title="📦 생활물류 AI 대시보드",
    layout="wide",
)

# 타이틀 및 인사말
st.title("📦 생활물류 AI 분석 대시보드")
st.markdown("### 👋 도심형 물류센터(MFC) 운영 관리자를 위한 통합 분석 시스템")

# 오늘 날짜 표시
st.markdown(f"**📅 오늘 날짜:** `{date.today().strftime('%Y-%m-%d')}`")

# 프로젝트 소개
st.markdown("""
---
이 대시보드는 **서울시 도심형 물류센터(MFC)**의 일별·품목별 물동량 데이터를 기반으로,  
AI 기반 수요 예측과 이상치 탐지, 운영 인사이트를 제공하는 **스마트 물류 의사결정 도구**입니다.

---

### 🧭 주요 기능 설명

#### 📌 [1. 데이터 요약 및 품목별 트렌드 분석 (`data_summary.py`, `item_trend.py`)]
- 전체 데이터 기반으로 **품목별 평균 물동량 비중**, **요일별 추이**, **공휴일/명절 영향** 등을 시각화합니다.
- **Pie, Bar, Line 그래프**를 활용하여 **품목 간 편차와 계절성 패턴**을 쉽게 파악할 수 있습니다.

#### 🏢 [2. 물류센터 간 비교 분석 (`center_comparison.py`)]
- 선택한 기간 동안 주요 물류센터 간 **총 물동량**, **품목 비중**, **증감률**을 비교할 수 있습니다.
- 센터별 효율성, 수요 집중도 등을 시각적으로 분석합니다.

#### 🔮 [3. 수요 예측: Prophet 기반 (`prophet_forecast.py`)]
- Facebook Prophet 모델을 이용해 **요일/명절 효과를 반영한 시계열 기반 예측**을 수행합니다.
- 예측 결과를 라인 차트로 시각화하며, MAE/RMSE 지표도 함께 제공합니다.

#### 🌲 [4. 수요 예측: LightGBM 기반 (`lgbm_forecast.py`)]
- Lag, 이동평균, 변동계수 등 다양한 피처를 활용한 **머신러닝 기반 예측 모델**입니다.
- 예측값이 음수가 되지 않도록 후처리를 적용하며, 실제와 예측을 비교해 보여줍니다.

#### ⚖️ [5. 예측 모델 비교 (`model_comparison.py`)]
- Prophet과 LightGBM 모델의 성능을 **동일 기간 내 지표(MAE, RMSE, R²)**로 비교합니다.
- 어떤 모델이 어떤 조건에서 더 정확한지 실무 관점에서 해석할 수 있습니다.

#### 🚨 [6. 이상치 탐지 및 원인 분석 (`anomaly_detection.py`, `error_analysis.py`)]
- 공휴일 및 요일별 계절성을 고려한 **Z-score 기반 이상치 탐지** 기능입니다.
- 수요 급증/급감의 원인을 명절, 연휴, 특정 요일 효과 등과 연결지어 분석 결과를 제공합니다.

#### 🧠 [7. AI 기반 운영 인사이트 (`model_ranking.py`, `insight_dashboard.py`)]
- 평균 수요, 변동성, 예측 안정성 등을 기준으로 **품목별/센터별 위험도 순위**를 제공하고,
- 전체 데이터를 기반으로 한 **자동 인사이트 시각화**로 경영 판단을 지원합니다.

---

👈 **좌측 사이드바에서 원하는 기능 페이지를 선택해 사용할 수 있습니다.**

💡 *향후 로그인 기반 사용자별 맞춤 리포트, 자동 이메일 발송 기능도 추가 예정입니다.*
""")
