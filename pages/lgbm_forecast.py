# pages/lgbm_forecast.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import holidays
import sys
import os

# src 경로 추가 및 로더 불러오기
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.loader import load_logistics_data

# -------------------------
# 1. 페이지 설정
# -------------------------
st.set_page_config(page_title="LightGBM Forecast", layout="wide")
st.title("🔍 LightGBM 기반 물동량 예측")

# -------------------------
# 2. 데이터 로딩
# -------------------------
DATA_PATH = "data/logistics_by_center.csv"

@st.cache_data
def load_data():
    return load_logistics_data(DATA_PATH)

df = load_data()

# -------------------------
# 3. 사용자 필터
# -------------------------
st.sidebar.header("예측 조건")
center = st.sidebar.selectbox("센터 선택", df["center_name"].unique())
item = st.sidebar.selectbox("품목 선택", df.columns[2:13])
period_days = st.sidebar.selectbox("예측 기간 (일)", [7, 14, 30], index=1)

# -------------------------
# 4. 피처 생성
# -------------------------
target_df = df[df["center_name"] == center][["date", item]].copy()
target_df = target_df.rename(columns={"date": "ds", item: "y"})

# 시계열 피처
target_df["lag_1"] = target_df["y"].shift(1)
target_df["lag_7"] = target_df["y"].shift(7)
target_df["rolling_mean_7"] = target_df["y"].rolling(7).mean()

# 날짜 피처
target_df["dow"] = target_df["ds"].dt.dayofweek
kr_holidays = holidays.KR(years=target_df["ds"].dt.year.unique())
target_df["is_holiday"] = target_df["ds"].isin(kr_holidays).astype(int)

# 결측 제거
target_df = target_df.dropna()

# -------------------------
# 5. 학습/예측 데이터 분리
# -------------------------
train_df = target_df.iloc[:-period_days]
test_df = target_df.iloc[-period_days:]

feature_cols = ["lag_1", "lag_7", "rolling_mean_7", "dow", "is_holiday"]

X_train = train_df[feature_cols]
y_train = train_df["y"]
X_test = test_df[feature_cols]
y_test = test_df["y"]

# -------------------------
# 6. 모델 학습 및 예측
# -------------------------
model = LGBMRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ⛔ 예측값 보정 (음수 제거)
y_pred = np.where(y_pred < 0, 0, y_pred)

# -------------------------
# 7. 평가 지표
# -------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.markdown(f"""
### 🧪 예측 성능 평가 (LightGBM)
- **MAE** (평균절대오차): `{mae:.2f}`
- **RMSE** (평균제곱근오차): `{rmse:.2f}`
- **R² Score**: `{r2:.3f}`
""")

# -------------------------
# 8. 시각화
# -------------------------
st.subheader(f"{center} - {item} 예측 결과 비교")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=test_df["ds"],
    y=y_test.values,
    mode="lines+markers",
    name="실제값",
    line=dict(color="blue")
))

fig.add_trace(go.Scatter(
    x=test_df["ds"],
    y=y_pred,
    mode="lines+markers",
    name="예측값",
    line=dict(color="green")
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
# 9. 예측 결과 테이블
# -------------------------
st.markdown("### 📋 예측 결과 테이블")

result_df = pd.DataFrame({
    "날짜": test_df["ds"].values,
    "실제값": y_test.values,
    "예측값": y_pred
})

st.dataframe(result_df.set_index("날짜").round(2), use_container_width=True)
