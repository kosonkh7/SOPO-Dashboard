import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import plotly.graph_objects as go
import holidays
import sys
import os

# 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.loader import load_logistics_data

# -------------------------------
# 1. 페이지 설정
# -------------------------------
st.set_page_config(page_title="모델 성능 비교", layout="wide")
st.title("📊 Prophet vs LightGBM 예측 성능 비교")

# -------------------------------
# 2. 데이터 로딩
# -------------------------------
DATA_PATH = "data/logistics_by_center.csv"

@st.cache_data
def load_data():
    return load_logistics_data(DATA_PATH)

df = load_data()

# -------------------------------
# 3. 사용자 입력
# -------------------------------
st.sidebar.header("예측 조건 선택")
center = st.sidebar.selectbox("센터", df["center_name"].unique())
item = st.sidebar.selectbox("품목", df.columns[2:13])
period_days = st.sidebar.selectbox("예측 기간 (일)", [7, 14, 30], index=1)

# -------------------------------
# 4. 데이터 전처리
# -------------------------------
target_df = df[df["center_name"] == center][["date", item]].copy()
target_df = target_df.rename(columns={"date": "ds", item: "y"})
target_df["is_holiday"] = target_df["ds"].isin(
    holidays.KR(years=target_df["ds"].dt.year.unique())
).astype(int)
target_df["dow"] = target_df["ds"].dt.dayofweek
target_df["lag_1"] = target_df["y"].shift(1)
target_df["lag_7"] = target_df["y"].shift(7)
target_df["rolling_mean_7"] = target_df["y"].rolling(7).mean()
target_df = target_df.dropna().reset_index(drop=True)

# 학습/테스트 분리
train_df = target_df.iloc[:-period_days].copy().reset_index(drop=True)
test_df = target_df.iloc[-period_days:].copy().reset_index(drop=True)

# -------------------------------
# 5. Prophet 모델 학습
# -------------------------------
prophet = Prophet(
    weekly_seasonality=False,
    yearly_seasonality=True,
    daily_seasonality=False
)
prophet.add_regressor("is_holiday")
prophet.add_regressor("dow")
prophet.add_regressor("lag_1")
prophet.fit(train_df)

future = test_df[["ds", "is_holiday", "dow", "lag_1"]]
forecast = prophet.predict(future)
forecast["yhat"] = forecast["yhat"].apply(lambda x: max(x, 0))  # 음수 제거

# -------------------------------
# 6. LightGBM 모델 학습
# -------------------------------
features = ["lag_1", "lag_7", "rolling_mean_7", "dow", "is_holiday"]
X_train = train_df[features]
y_train = train_df["y"]
X_test = test_df[features]
y_test = test_df["y"]

lgbm = LGBMRegressor(random_state=42)
lgbm.fit(X_train, y_train)
lgbm_pred = lgbm.predict(X_test)
lgbm_pred = np.where(lgbm_pred < 0, 0, lgbm_pred)  # 음수 제거

# -------------------------------
# 7. 성능 평가 함수
# -------------------------------
def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

prophet_metrics = evaluate(y_test, forecast["yhat"])
lgbm_metrics = evaluate(y_test, lgbm_pred)

# -------------------------------
# 8. 성능 지표 출력
# -------------------------------
st.markdown("### 🧪 성능 지표 비교")
metrics_df = pd.DataFrame([prophet_metrics, lgbm_metrics], index=["Prophet", "LightGBM"])
st.dataframe(metrics_df.style.format("{:.3f}"), use_container_width=True)

# -------------------------------
# 9. 예측 결과 시각화
# -------------------------------
st.markdown("### 📈 예측 결과 시각화")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=test_df["ds"],
    y=y_test,
    mode="lines+markers",
    name="실제값",
    line=dict(color="black")
))
fig.add_trace(go.Scatter(
    x=test_df["ds"],
    y=forecast["yhat"],
    mode="lines+markers",
    name="Prophet 예측",
    line=dict(color="blue")
))
fig.add_trace(go.Scatter(
    x=test_df["ds"],
    y=lgbm_pred,
    mode="lines+markers",
    name="LightGBM 예측",
    line=dict(color="green")
))

fig.update_layout(
    xaxis_title="날짜",
    yaxis_title="물동량",
    template="plotly_white",
    legend_title="모델",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 10. 예측 결과 테이블
# -------------------------------
st.markdown("### 📋 예측 결과 테이블")
result_df = pd.DataFrame({
    "날짜": test_df["ds"],
    "실제값": y_test,
    "Prophet 예측": forecast["yhat"],
    "LightGBM 예측": lgbm_pred
})
st.dataframe(result_df.set_index("날짜").round(2), use_container_width=True)
