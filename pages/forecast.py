# pages/forecast.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
import holidays
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import sys
import os

# src 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.loader import load_logistics_data

# -------------------------
# 1. 페이지 설정
# -------------------------
st.set_page_config(page_title="Forecast", layout="wide")
st.title("📈 품목별 물동량 예측 (고도화 Prophet + 성능 평가)")

# -------------------------
# 2. 데이터 로딩
# -------------------------
DATA_PATH = "data/logistics_by_center.csv"

@st.cache_data
def load_data():
    return load_logistics_data(DATA_PATH)

df = load_data()

# -------------------------
# 3. 사용자 입력 필터
# -------------------------
st.sidebar.header("예측 조건")

center = st.sidebar.selectbox("센터 선택", df["center_name"].unique())
item = st.sidebar.selectbox("품목 선택", df.columns[2:13])
period_days = st.sidebar.selectbox("예측 기간 (일)", [7, 14, 30], index=1)

# -------------------------
# 4. 데이터 전처리 (외생 변수 포함)
# -------------------------
target_df = df[df["center_name"] == center][["date", item]].copy()
target_df = target_df.rename(columns={"date": "ds", item: "y"})

# 요일 원핫 인코딩
target_df["dow"] = target_df["ds"].dt.dayofweek
dow_dummies = pd.get_dummies(target_df["dow"], prefix="dow")
target_df = pd.concat([target_df, dow_dummies], axis=1)

# 공휴일 여부
kr_holidays = holidays.KR(years=target_df["ds"].dt.year.unique())
target_df["is_holiday"] = target_df["ds"].isin(kr_holidays).astype(int)

# lag_1
target_df["lag_1"] = target_df["y"].shift(1)
target_df = target_df.dropna()

# -------------------------
# 5. Prophet 모델 학습
# -------------------------
model = Prophet(
    weekly_seasonality=False,
    yearly_seasonality=True,
    daily_seasonality=False
)

for col in ["is_holiday", "lag_1"] + list(dow_dummies.columns):
    model.add_regressor(col)

model.fit(target_df)

# -------------------------
# 6. future 생성 및 외생 변수 반영
# -------------------------
future = model.make_future_dataframe(periods=period_days)

# 요일 원핫
future["dow"] = future["ds"].dt.dayofweek
dow_dummies_future = pd.get_dummies(future["dow"], prefix="dow")
for i in range(7):
    col = f"dow_{i}"
    if col not in dow_dummies_future:
        dow_dummies_future[col] = 0
dow_dummies_future = dow_dummies_future[[f"dow_{i}" for i in range(7)]]

# 공휴일 여부
future["is_holiday"] = future["ds"].isin(kr_holidays).astype(int)

# lag_1: 과거 y에서 시프트 + ffill + NaN → 0
y_full = pd.concat([target_df["y"], pd.Series([None] * period_days)], ignore_index=True)
lag_1_full = y_full.shift(1).fillna(method="ffill").fillna(0)
future["lag_1"] = lag_1_full[-len(future):].values

# 외생 변수 결합
future = pd.concat([future, dow_dummies_future], axis=1)

# -------------------------
# 7. 예측 실행
# -------------------------
forecast = model.predict(future)

# -------------------------
# 8. 예측 시각화
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
    mode="lines",
    name="상한",
    line=dict(width=0),
    showlegend=False
))
fig.add_trace(go.Scatter(
    x=forecast["ds"],
    y=forecast["yhat_lower"],
    mode="lines",
    fill='tonexty',
    fillcolor='rgba(0, 255, 0, 0.1)',
    line=dict(width=0),
    name="하한"
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
# 9. 성능 평가 (예측 기간 이전 구간에 대해 측정)
# -------------------------
# 실측값 vs 예측값 비교 (가능한 구간 내)
compare_days = min(period_days, len(target_df))
y_true = target_df["y"].values[-compare_days:]
y_pred = forecast["yhat"].values[-(compare_days + period_days):-period_days]  # 마지막 n일 전

if len(y_true) == len(y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    st.markdown(f"""
    ### 🧪 예측 성능 평가 (최근 {compare_days}일 기준)
    - **MAE** (평균절대오차): `{mae:.2f}`
    - **RMSE** (평균제곱근오차): `{rmse:.2f}`
    - **R² Score**: `{r2:.3f}`
    """)
else:
    st.warning("실측값과 예측값 비교 불가 (데이터 부족)")

# -------------------------
# 10. 예측 결과 테이블
# -------------------------
st.markdown("### 📋 향후 예측 결과")

# 예측값이 0 미만인 경우, 0으로 보정 (물동량은 음수 불가)
forecast["yhat"] = forecast["yhat"].apply(lambda x: max(x, 0))
forecast["yhat_lower"] = forecast["yhat_lower"].apply(lambda x: max(x, 0))
forecast["yhat_upper"] = forecast["yhat_upper"].apply(lambda x: max(x, 0))


forecast_result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(period_days)
forecast_result.columns = ["날짜", "예측값", "하한", "상한"]

st.dataframe(forecast_result.set_index("날짜").round(2), use_container_width=True)
