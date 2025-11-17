import streamlit as st
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import holidays
import sys
import os

# 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.loader import load_logistics_data

# -------------------------------
# 1. 페이지 기본 설정
# -------------------------------
st.set_page_config(page_title="모델 성능 순위", layout="wide")
st.title("📊 LightGBM 예측 성능 순위 (센터 × 품목)")

# -------------------------------
# 2. 데이터 로딩
# -------------------------------
DATA_PATH = "data/logistics_by_center.csv"

@st.cache_data
def load_data():
    return load_logistics_data(DATA_PATH)

df = load_data()

# -------------------------------
# 3. 설정
# -------------------------------
period_days = st.sidebar.selectbox("예측 기간 (일)", [7, 14, 30], index=1)
kr_holidays = holidays.KR()

# -------------------------------
# 4. 성능 계산
# -------------------------------
st.info("모든 센터 × 품목에 대해 LightGBM 예측을 수행 중입니다...")

results = []
centers = df["center_name"].unique()
items = df.columns[2:]

for center in centers:
    for item in items:
        try:
            target_df = df[df["center_name"] == center][["date", item]].copy()
            target_df = target_df.rename(columns={"date": "ds", item: "y"})

            # 특징 생성
            target_df["is_holiday"] = target_df["ds"].isin(
                holidays.KR(years=target_df["ds"].dt.year.unique())
            ).astype(int)
            target_df["dow"] = target_df["ds"].dt.dayofweek
            target_df["lag_1"] = target_df["y"].shift(1)
            target_df["lag_7"] = target_df["y"].shift(7)
            target_df["rolling_mean_7"] = target_df["y"].rolling(7).mean()
            target_df = target_df.dropna().reset_index(drop=True)

            if len(target_df) <= period_days:
                continue

            train_df = target_df.iloc[:-period_days].copy().reset_index(drop=True)
            test_df = target_df.iloc[-period_days:].copy().reset_index(drop=True)

            features = ["lag_1", "lag_7", "rolling_mean_7", "dow", "is_holiday"]
            X_train = train_df[features]
            y_train = train_df["y"]
            X_test = test_df[features]
            y_test = test_df["y"]

            # 모델 학습 및 예측
            model = LGBMRegressor(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred = np.where(y_pred < 0, 0, y_pred)

            # 성능 지표 계산
            mae = mean_absolute_error(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results.append({
                "센터": center,
                "품목": item,
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "R2": round(r2, 3)
            })

        except Exception as e:
            st.warning(f"🚨 오류 발생 - {center} / {item}: {e}")
            continue

# -------------------------------
# 5. 결과 출력
# -------------------------------
if results:
    result_df = pd.DataFrame(results)
    sort_by = st.selectbox("정렬 기준", ["RMSE", "MAE", "R2"], index=0)
    ascending = st.radio("정렬 순서", ["오름차순", "내림차순"]) == "오름차순"

    result_df_sorted = result_df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)

    st.markdown("### 📋 예측 성능 순위표")
    st.dataframe(result_df_sorted, use_container_width=True)
else:
    st.error("⚠️ 계산 가능한 조합이 없습니다. 데이터를 다시 확인해주세요.")
