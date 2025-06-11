import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import holidays
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.loader import load_logistics_data

# -------------------------
# 1. 페이지 설정
# -------------------------
st.set_page_config(page_title="Prophet Forecast", layout="wide")
st.title("🔮 Prophet 기반 물동량 예측")

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
# 4. Prophet용 데이터 전처리
# -------------------------
target_df = df[df["center_name"] == center][["date", item]].copy()
target_df = target_df.rename(columns={"date": "ds", item: "y"})

# 공휴일 정보
kr_holidays = holidays.KR(years=target_df["ds"].dt.year.unique())
target_df["is_holiday"] = target_df["ds"].isin(kr_holidays).astype(int)
target_df["dow"] = target_df["ds"].dt.dayofweek
target_df["lag_1"] = target_df["y"].shift(1)
target_df = target_df.dropna()

# 학습 / 평가 분리
train_df = target_df.iloc[:-period_days].copy()
test_df = target_df.iloc[-period_days:].copy()

# -------------------------
# 5. Prophet 모델 구성 및 학습
# -------------------------
model = Prophet(
    weekly_seasonality=False,
    yearly_seasonality=True,
    daily_seasonality=False
)

# 외생 변수 등록
model.add_regressor("is_holiday")
model.add_regressor("dow")
model.add_regressor("lag_1")

model.fit(train_df)

# -------------------------
# 6. 예측용 future 데이터 구성
# -------------------------
future = test_df[["ds", "is_holiday", "dow", "lag_1"]].copy()
forecast = model.predict(future)

# 음수값 보정
forecast["yhat"] = forecast["yhat"].apply(lambda x: max(x, 0))
forecast["yhat_lower"] = forecast["yhat_lower"].apply(lambda x: max(x, 0))
forecast["yhat_upper"] = forecast["yhat_upper"].apply(lambda x: max(x, 0))

# -------------------------
# 7. 성능 평가
# -------------------------
y_true = test_df["y"].values
y_pred = forecast["yhat"].values

mae = mean_absolute_error(y_true, y_pred)
rmse = root_mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

st.markdown(f"""
### 🧪 예측 성능 평가 (Prophet)
- **MAE**: `{mae:.2f}`
- **RMSE**: `{rmse:.2f}`
- **R² Score**: `{r2:.3f}`
""")

# -------------------------
# 8. 시각화
# -------------------------
st.subheader(f"{center} - {item} 예측 결과")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=test_df["ds"],
    y=y_true,
    mode="lines+markers",
    name="실제값",
    line=dict(color="blue")
))

fig.add_trace(go.Scatter(
    x=forecast["ds"],
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
    "날짜": forecast["ds"],
    "실제값": y_true,
    "예측값": y_pred
})

st.dataframe(result_df.set_index("날짜").round(2), use_container_width=True)
