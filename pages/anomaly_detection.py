# pages/anomaly_detection.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import holidays
import sys
import os

# src 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.loader import load_logistics_data

# -------------------------
# 1. 페이지 설정
# -------------------------
st.set_page_config(page_title="Anomaly Detection", layout="wide")
st.title("📊 요일 및 공휴일 기반 이상치 탐지")

# -------------------------
# 2. 데이터 불러오기
# -------------------------
DATA_PATH = "data/logistics_by_center.csv"

@st.cache_data
def load_data():
    return load_logistics_data(DATA_PATH)

df = load_data()
df['weekday'] = df['date'].dt.day_name()

# -------------------------
# 3. 사용자 필터
# -------------------------
st.sidebar.header("필터 옵션")
center = st.sidebar.selectbox("센터 선택", df["center_name"].unique())
item = st.sidebar.selectbox("품목 선택", df.columns[2:13])

# -------------------------
# 4. 이상치 탐지 함수
# -------------------------
def detect_outliers_by_weekday(df, center_name, item_col, z_thresh=2.5):
    center_df = df[df["center_name"] == center_name].copy()
    stats = center_df.groupby("weekday")[item_col].agg(["mean", "std"]).rename(columns={"mean": "avg", "std": "std"})

    center_df["avg"] = center_df["weekday"].map(stats["avg"])
    center_df["std"] = center_df["weekday"].map(stats["std"])
    center_df["z_score"] = (center_df[item_col] - center_df["avg"]) / center_df["std"]
    center_df["is_outlier"] = center_df["z_score"].abs() > z_thresh

    return center_df

# -------------------------
# 5. 공휴일 영향 판단 함수
# -------------------------
def mark_holiday_related_outliers(df: pd.DataFrame, country='KR'):
    years = df['date'].dt.year.unique().tolist()
    kr_holidays = holidays.KR(years=years)

    # 공휴일 날짜 목록 생성
    holiday_df = pd.DataFrame({
        "holiday_date": pd.to_datetime(list(kr_holidays.keys())),
        "holiday_name": list(kr_holidays.values())
    })

    # ±2일 범위 포함 (추석/설날도 자동 포함됨)
    extended_holidays = []
    for _, row in holiday_df.iterrows():
        base_date = row['holiday_date']
        for offset in range(-2, 3):  # -2, -1, 0, +1, +2
            date = base_date + pd.Timedelta(days=offset)
            extended_holidays.append((date, row['holiday_name']))

    holiday_map = pd.DataFrame(extended_holidays, columns=["date", "holiday_name"]).drop_duplicates()
    df = df.merge(holiday_map, how="left", on="date")
    df["is_holiday_related"] = df["holiday_name"].notna()

    return df

# -------------------------
# 6. 판단 기준 설명
# -------------------------
st.markdown("""
### 🧠 이상치 판단 기준
- **Z-score 방식**: 요일별 평균을 기준으로 `(관측값 - 평균) / 표준편차` 계산
- `|Z-score| > 2.5`이면 이상치로 판단  
- **공휴일 영향**: 설날/추석/신정/크리스마스 등 공휴일 **±2일 이내**이면 공휴일 영향으로 간주
""")

# -------------------------
# 7. 탐지 실행
# -------------------------
result_df = detect_outliers_by_weekday(df, center, item)
result_df = mark_holiday_related_outliers(result_df)

# -------------------------
# 8. 시각화
# -------------------------
st.subheader(f"{center} - {item} 이상치 분류 시각화")

fig = go.Figure()

# 정상 데이터
normal = result_df[~result_df["is_outlier"]]
fig.add_trace(go.Scatter(
    x=normal["date"], y=normal[item],
    mode="lines+markers", name="정상",
    line=dict(color="gray"), marker=dict(size=5)
))

# 공휴일 관련 이상치
holiday_outliers = result_df[(result_df["is_outlier"]) & (result_df["is_holiday_related"])]
fig.add_trace(go.Scatter(
    x=holiday_outliers["date"], y=holiday_outliers[item],
    mode="markers", name="공휴일 이상치",
    marker=dict(color="orange", size=12, symbol="star")
))

# 일반 이상치
normal_outliers = result_df[(result_df["is_outlier"]) & (~result_df["is_holiday_related"])]
fig.add_trace(go.Scatter(
    x=normal_outliers["date"], y=normal_outliers[item],
    mode="markers", name="일반 이상치",
    marker=dict(color="red", size=10, symbol="x")
))

fig.update_layout(
    xaxis_title="날짜",
    yaxis_title="물동량",
    template="plotly_white",
    legend_title="데이터 구분",
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# 9. 이상치 테이블
# -------------------------
st.markdown("### 📋 이상치 상세 목록")

display_df = result_df[result_df["is_outlier"]][["date", "weekday", item, "z_score", "is_holiday_related", "holiday_name"]]
display_df = display_df.rename(columns={
    "z_score": "Z점수",
    "is_holiday_related": "공휴일영향여부",
    "holiday_name": "공휴일이름"
})

st.dataframe(display_df.sort_values("date"), use_container_width=True)
