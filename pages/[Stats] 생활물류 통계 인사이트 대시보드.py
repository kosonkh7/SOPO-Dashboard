from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import holidays
import streamlit as st
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

# 🗓️ 날짜 파생 컬럼
kr_holidays = holidays.KR()
df["is_holiday"] = df["date"].isin(kr_holidays)
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["dow"] = df["date"].dt.dayofweek
df["year_month"] = df["date"].dt.to_period("M").astype(str)

item_columns = df.columns[2:13]

# 📌 1. 품목 평균 비중 (Pie Chart)
mean_by_item = df[item_columns].mean()
pie_df = pd.DataFrame({"item": mean_by_item.index, "avg_volume": mean_by_item.values})
fig_pie = px.pie(
    pie_df.sort_values("avg_volume", ascending=False),
    names="item", values="avg_volume",
    title="📌 품목별 평균 물동량 비중"
)

# 📊 2. 요일별 평균 물동량 (Line Chart)
weekday_avg = df.groupby("dow")[item_columns].mean().T
weekday_avg.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
weekday_colors = px.colors.qualitative.Set3
fig_weekday = go.Figure()
for idx, item_name in enumerate(weekday_avg.index):
    fig_weekday.add_trace(go.Scatter(
        x=weekday_avg.columns,
        y=weekday_avg.loc[item_name],
        mode="lines+markers",
        name=item_name,
        line=dict(color=weekday_colors[idx % len(weekday_colors)])
    ))
fig_weekday.update_layout(
    title="📊 품목별 요일 평균 물동량 추이",
    xaxis_title="요일", yaxis_title="평균 물동량"
)

# 📉 3. 요일 변동성 표준편차
weekday_std = df.groupby("dow")[item_columns].std().T.mean(axis=1)
std_df = pd.DataFrame({"item": weekday_std.index, "std_dev": weekday_std.values})
fig_std = px.bar(
    std_df.sort_values("std_dev", ascending=False),
    x="item", y="std_dev",
    title="📉 품목별 요일별 표준편차 (변동성)",
    color="item", text_auto=".2s"
)

# 🎎 4. 명절 주간 vs 일반 주간 (food)
df["is_festival_week"] = df["date"].apply(
    lambda d: any([(d - pd.Timedelta(days=i)) in kr_holidays for i in range(3)])
)
festival_vs_normal = df.groupby("is_festival_week")[["food"]].mean().reset_index()
festival_vs_normal["label"] = festival_vs_normal["is_festival_week"].map({True: "명절 주간", False: "일반 주간"})
fig_festival = px.bar(
    festival_vs_normal, x="label", y="food",
    title="🎎 명절 vs 일반 주간 food 물동량 비교",
    color="label", text_auto=".2s"
)

# 🏢 5. 센터별 누적 물동량 상위 10
center_total = df.groupby("center_name")[item_columns].sum().sum(axis=1).sort_values(ascending=False).head(10)
center_df = pd.DataFrame({"center": center_total.index, "total_volume": center_total.values})
fig_center = px.bar(
    center_df, x="center", y="total_volume",
    title="🏢 센터별 누적 물동량 (상위 10)",
    color="center", text_auto=".2s"
)

# 📈 6. 월별 물동량 추이
monthly_total = df.groupby("year_month")[item_columns].sum().sum(axis=1).reset_index()
monthly_total.columns = ["year_month", "total_volume"]
fig_monthly = px.line(
    monthly_total, x="year_month", y="total_volume",
    title="📈 월별 전체 물동량 추이"
)
fig_monthly.update_layout(xaxis_tickangle=-45)

# 🎛️ Streamlit 시각화 배치
st.title("📦 생활물류 통계 인사이트 대시보드")

# ▶️ 2열 배치 (품목 비중 / 센터 누적)
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_pie, use_container_width=True)
with col2:
    st.plotly_chart(fig_center, use_container_width=True)

# ▶️ 단일 차트 (요일 추이)
st.plotly_chart(fig_weekday, use_container_width=True)

# ▶️ 2열 배치 (표준편차 / 명절비교)
col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(fig_std, use_container_width=True)
with col4:
    st.plotly_chart(fig_festival, use_container_width=True)

# ▶️ 단일 차트 (월별 추이)
st.plotly_chart(fig_monthly, use_container_width=True)