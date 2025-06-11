# 이전 코드 재실행 (lightgbm 설치 후)
import streamlit as st
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from scipy.stats import zscore
import holidays
import sys
import os

# src 경로 추가 및 데이터 로더 import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.loader import load_logistics_data

# 데이터 로딩
DATA_PATH = "data/logistics_by_center.csv"

@st.cache_data
def load_data():
    return load_logistics_data(DATA_PATH)

df = load_data()

kr_holidays = holidays.KR()
period_days = 14

# 평가 결과 저장 리스트
error_analysis_results = []

# 고정 품목 & 센터 리스트
centers = df["center_name"].unique()
items = df.columns[2:]

# 전체 성능 저장
performance = []

# 모든 센터 × 품목 조합 순회
for center in centers:
    for item in items:
        try:
            target_df = df[df["center_name"] == center][["date", item]].copy()
            target_df = target_df.rename(columns={"date": "ds", item: "y"})

            # 특징 생성
            target_df["is_holiday"] = target_df["ds"].isin(kr_holidays).astype(int)
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

            model = LGBMRegressor(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred = np.where(y_pred < 0, 0, y_pred)

            rmse = root_mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            performance.append({
                "center": center,
                "item": item,
                "rmse": rmse,
                "r2": r2,
                "y_true": y_test.values,
                "y_pred": y_pred,
                "dates": test_df["ds"].values
            })

        except Exception:
            continue

# RMSE 기준 하위 10개 선택
sorted_perf = sorted(performance, key=lambda x: x["rmse"], reverse=True)[:10]

# 오차 원인 자동 분석
for entry in sorted_perf:
    y_true = entry["y_true"]
    y_pred = entry["y_pred"]
    dates = pd.to_datetime(entry["dates"])

    reasons = []

    # 1. 공휴일 영향
    if any(d in kr_holidays for d in dates):
        reasons.append("공휴일 포함")

    # 2. 급등/급락
    pct_change = (y_true[-1] - y_true[0]) / (np.mean(y_true) + 1e-6)
    if abs(pct_change) > 0.5:
        reasons.append("급등/급락 패턴")

    # 3. 요일 계절성
    true_std_by_dow = pd.Series(y_true, index=dates).groupby(dates.dayofweek).std().mean()
    pred_std_by_dow = pd.Series(y_pred, index=dates).groupby(dates.dayofweek).std().mean()
    if true_std_by_dow > 1.5 * pred_std_by_dow:
        reasons.append("요일 효과 미반영")

    # 4. 이상치 존재 여부
    if any(abs(zscore(y_true)) > 3):
        reasons.append("이상치 포함")

    if not reasons:
        reasons = ["패턴 불명확"]

    error_analysis_results.append({
        "센터": entry["center"],
        "품목": entry["item"],
        "RMSE": round(entry["rmse"], 2),
        "R2": round(entry["r2"], 3),
        "주요 원인": ", ".join(reasons)
    })

result_df = pd.DataFrame(error_analysis_results).sort_values(by="RMSE", ascending=False).reset_index(drop=True)

st.subheader("예측 오차 원인 분석 결과")
st.dataframe(result_df)