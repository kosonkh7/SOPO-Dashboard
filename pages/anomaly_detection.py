# pages/anomaly_detection.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os

# src 경로 추가 및 로더 불러오기
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.loader import load_logistics_data

# -------------------------
# 1. 페이지 설정
# -------------------------
st.set_page_config(page_title="Anomaly Detection", layout="wide")
st.title("요일 기반 이상치 탐지")

# -------------------------
# 2. 데이터 불러오기
# -------------------------
DATA_PATH = "data/logistics_by_center.csv"

@st.cache_data
def load_data():
    return load_logistics_data(DATA_PATH)

df = load_data()
df['weekday'] = df['date'].dt.day_name()  # 요일 이름 컬럼 추가

# -------------------------
# 3. 사용자 필터
# -------------------------
st.sidebar.header("필터 옵션")

center = st.sidebar.selectbox("센터 선택", df["center_name"].unique())
item = st.sidebar.selectbox("품목 선택", df.columns[2:13])

# -------------------------
# 4. Z-score 이상치 탐지 함수
# -------------------------
def detect_outliers_by_weekday(df, center_name, item_col, z_thresh=2.5):
    """
    요일 기준 Z-score 기반 이상치 탐지.
    """
    center_df = df[df["center_name"] == center_name].copy()

    # 요일별 평균 및 표준편차 계산
    stats = center_df.groupby("weekday")[item_col].agg(["mean", "std"]).rename(columns={"mean": "avg", "std": "std"})

    # Z-score 계산 및 이상치 판단
    center_df["avg"] = center_df["weekday"].map(stats["avg"])
    center_df["std"] = center_df["weekday"].map(stats["std"])
    center_df["z_score"] = (center_df[item_col] - center_df["avg"]) / center_df["std"]
    center_df["is_outlier"] = center_df["z_score"].abs() > z_thresh

    return center_df

# -------------------------
# 5. 이상치 판단 기준 설명
# -------------------------
st.markdown("""
### ❗ 이상치 판단 기준

- **Z-score 방식 사용**  
- `Z-score = (해당일 물동량 - 요일 평균) / 요일 표준편차`
- `|Z-score| > 2.5`인 경우 **이상치로 간주**합니다  
- 예: `월요일 평균 = 1000`, 표준편차 = 100 → 1300이면 `Z-score = 3.0` → 이상치
""")

# -------------------------
# 6. 이상치 탐지 및 시각화
# -------------------------
result_df = detect_outliers_by_weekday(df, center, item)

st.subheader(f"📊 {center} - {item} 물동량 (이상치 강조)")

fig = go.Figure()

# 정상값 라인
fig.add_trace(go.Scatter(
    x=result_df["date"],
    y=result_df[item],
    mode="lines+markers",
    name="정상값",
    line=dict(color="blue", width=2),
    marker=dict(size=6),
))

# 이상치 점만 별도 추가 (X 마커, 빨간색)
outliers = result_df[result_df["is_outlier"]]
fig.add_trace(go.Scatter(
    x=outliers["date"],
    y=outliers[item],
    mode="markers",
    name="이상치",
    marker=dict(color="red", size=10, symbol="x"),
))

fig.update_layout(
    xaxis_title="날짜",
    yaxis_title="물동량",
    template="plotly_white",
    legend_title="구분",
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# 7. 이상치 목록 테이블
# -------------------------
st.markdown("### 📋 이상치 목록")

st.dataframe(
    outliers[["date", "weekday", item, "z_score"]].sort_values("date"),
    use_container_width=True
)
