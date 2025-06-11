# 코드 재실행 (환경 초기화됨)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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

# 한국 공휴일 정의
kr_holidays = holidays.KR()
df["is_holiday"] = df["date"].isin(kr_holidays)
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["dow"] = df["date"].dt.dayofweek
df["year_month"] = df["date"].dt.to_period("M").astype(str)

# 📌 1. 품목별 비중 계산 (전체 평균)
item_columns = df.columns[2:13]
mean_by_item = df[item_columns].mean()
pie_df = pd.DataFrame({"item": mean_by_item.index, "avg_volume": mean_by_item.values})

# 📊 2. 요일별 평균 물동량
weekday_avg = df.groupby("dow")[item_columns].mean().T
weekday_avg.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
weekday_avg["item"] = weekday_avg.index

# 📉 3. 품목별 요일 표준편차 (변동성)
weekday_std = df.groupby("dow")[item_columns].std().T.mean(axis=1)
std_df = pd.DataFrame({"item": weekday_std.index, "std_dev": weekday_std.values})

# 🎎 4. 명절 vs 일반 주간 평균 비교 (예시: food 품목만)
df["is_festival_week"] = df["date"].apply(
    lambda d: any([(d - pd.Timedelta(days=i)) in kr_holidays for i in range(3)])
)
festival_vs_normal = df.groupby("is_festival_week")[["food"]].mean().reset_index()
festival_vs_normal["label"] = festival_vs_normal["is_festival_week"].map({True: "명절 주간", False: "일반 주간"})

# 🗺️ 5. 센터별 총 물동량 상위 10
center_total = df.groupby("center_name")[item_columns].sum().sum(axis=1).sort_values(ascending=False).head(10)
center_df = pd.DataFrame({"center": center_total.index, "total_volume": center_total.values})

# 📈 6. 월별 물동량 추이
monthly_total = df.groupby("year_month")[item_columns].sum().sum(axis=1).reset_index()
monthly_total.columns = ["year_month", "total_volume"]

# 시각화 생성
fig_pie = px.pie(pie_df, names="item", values="avg_volume", title="📌 품목별 평균 물동량 비중")

fig_weekday = go.Figure()
for i, row in weekday_avg.iterrows():
    fig_weekday.add_trace(go.Scatter(x=weekday_avg.columns[:-1], y=row[:-1], mode="lines+markers", name=row["item"]))
fig_weekday.update_layout(title="📊 품목별 요일 평균 물동량 추이", xaxis_title="요일", yaxis_title="평균 물동량")

fig_std = px.bar(std_df, x="item", y="std_dev", title="📉 품목별 요일별 표준편차 (변동성)", text_auto=".2s")

fig_festival = px.bar(festival_vs_normal, x="label", y="food", title="🎎 명절 vs 일반 주간 food 물동량 비교")

fig_center = px.bar(center_df, x="center", y="total_volume", title="🏢 센터별 누적 물동량 (상위 10)", text_auto=".2s")

fig_monthly = px.line(monthly_total, x="year_month", y="total_volume", title="📈 월별 전체 물동량 추이")
fig_monthly.update_layout(xaxis_tickangle=-45)

# 시각화 객체 반환 (Streamlit 환경 외부라 화면 출력은 하지 않음)
(fig_pie, fig_weekday, fig_std, fig_festival, fig_center, fig_monthly)
