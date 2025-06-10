# pages/center_comparison.py

import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# src 경로 추가 및 로더 불러오기
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.loader import load_logistics_data
from src.visualizer import bar_chart_by_item

# -------------------------
# 1. 페이지 설정
# -------------------------
st.set_page_config(page_title="Center Comparison", layout="wide")
st.title("센터 간 품목별 물동량 비교")

# -------------------------
# 2. 데이터 불러오기
# -------------------------
DATA_PATH = "data/logistics_by_center.csv"

@st.cache_data
def load_data():
    return load_logistics_data(DATA_PATH)

df = load_data()

# -------------------------
# 3. 사용자 입력 필터
# -------------------------
st.sidebar.header("필터 옵션")

# 날짜 선택 (한 날짜만)
selected_date = st.sidebar.date_input(
    "날짜 선택",
    value=df["date"].min(),
    min_value=df["date"].min(),
    max_value=df["date"].max()
)

# 센터 선택
selected_centers = st.sidebar.multiselect(
    "센터 선택",
    options=df["center_name"].unique().tolist(),
    default=df["center_name"].unique().tolist()[:5]
)

# 품목 선택
category_columns = df.columns[2:13]
selected_items = st.sidebar.multiselect(
    "품목 선택",
    options=category_columns,
    default=["food", "digital", "fashion"]
)

# -------------------------
# 4. 필터링
# -------------------------
filtered_df = df[
    (df["date"] == pd.to_datetime(selected_date)) &
    (df["center_name"].isin(selected_centers))
]

# -------------------------
# 5. 시각화
# -------------------------
st.subheader(f"{selected_date.strftime('%Y-%m-%d')} 기준 센터 간 품목 비교")

if filtered_df.empty:
    st.warning("조건에 맞는 데이터가 없습니다.")
else:
    # 긴 포맷으로 변환
    melted_df = filtered_df.melt(
        id_vars="center_name",
        value_vars=selected_items,
        var_name="품목",
        value_name="물동량"
    )

    # Plotly barplot
    fig = bar_chart_by_item(melted_df)

    st.plotly_chart(fig, use_container_width=True)
