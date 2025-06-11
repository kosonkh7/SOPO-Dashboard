# pages/forecast.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
import sys
import os

# src 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.loader import load_logistics_data

# -------------------------
# 1. 페이지 설정
# -------------------------
st.set_page_config(page_title="Forecast", layout="wide")
st.title("📈 품목별 물동량 예측 (Prophet 기반)")

# -------------------------
# 2. 데이터 로딩
# -------------------------
DATA_PATH = "data/logistics_by_center.csv"

@st.cache_data
def load_data():
    return load_logistics_data(DATA_PATH)

df = load_data()

# -------------------------
# 3. 사용자 입력
# -------------------------
st.sidebar.header("예측 조건")

center = st.sidebar.selectbox("센터 선택", df["center_name"].unique())
item = st.sidebar.selectbox("품목 선택", df.columns[2:13])
period_days = st.sidebar.selectbox("예측 기간 (일)", [7, 14, 30], index=1)

# -------------------------
# 4. Prophet 학습 준비
# -------------------------
# 선택된 센터 + 품목 필터링
target_df = df[df["center_name"] == center][["date", item]].copy()
target_df = target_df.rename(columns={"date": "ds", item: "y"})

# Prophet 모델 구성
model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
model.fit(target_df)

# 예측 날짜 생성
future = model.make_future_dataframe(periods=period_days)
forecast = model.predict(future)

# -------------------------
# 5. 시각화
# -------------------------
st.subheader(f"{center} - {item} 향후 {period_days}일 예측")

fig = go.Figure()

# 실제값
fig.add_trace(go.Scatter(
    x=target_df["ds"],
    y=target_df["y"],
    mode="lines",
    name="실제값",
    line=dict(color="blue")
))

# 예측값
fig.add_trace(go.Scatter(
    x=forecast["ds"],
    y=forecast["yhat"],
    mode="lines",
    name="예측값",
    line=dict(color="green")
))

# 신뢰구간
fig.add_trace(go.Scatter(
    x=forecast["ds"],
    y=forecast["yhat_upper"],
    name="상한",
    mode="lines",
    line=dict(width=0),
    showlegend=False
))
fig.add_trace(go.Scatter(
    x=forecast["ds"],
    y=forecast["yhat_lower"],
    name="하한",
    mode="lines",
    fill='tonexty',
    fillcolor='rgba(0, 255, 0, 0.1)',
    line=dict(width=0),
    showlegend=True
))

fig.update_layout(
    xaxis_title="날짜",
    yaxis_title="물동량",
    template="plotly_white",
    hovermode="x unified",
    legend_title="구분"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# 6. 예측 데이터 테이블
# -------------------------
st.markdown("### 📋 예측 결과 테이블")
forecast_result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(period_days)
forecast_result.columns = ["날짜", "예측값", "하한", "상한"]

st.dataframe(forecast_result.set_index("날짜").round(2), use_container_width=True)
