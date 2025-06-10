# pages/item_trend.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import sys
import os

# src 경로 추가 및 데이터 로더 import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.loader import load_logistics_data

# -------------------------
# 0. 한글 폰트 설정 (맑은 고딕)
# -------------------------
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 기준
plt.rcParams['axes.unicode_minus'] = False     # 음수 깨짐 방지

# -------------------------
# 1. 페이지 설정
# -------------------------
st.set_page_config(page_title="Item Trend", layout="wide")
st.title("품목별 물동량 추이")

# -------------------------
# 2. 데이터 불러오기
# -------------------------
DATA_PATH = "data/logistics_by_center.csv"

@st.cache_data
def load_data():
    return load_logistics_data(DATA_PATH)

df = load_data()

# -------------------------
# 3. 필터 구성
# -------------------------
st.sidebar.header("필터 옵션")

# (1) 연도 또는 연-월 선택
df['year'] = df['date'].dt.year
df['year_month'] = df['date'].dt.to_period("M").astype(str)  # 예: "2018-01"

mode = st.sidebar.radio("필터 기준", ["연도별", "연-월별"])

if mode == "연도별":
    selected_years = st.sidebar.multiselect("연도 선택", sorted(df['year'].unique()), default=[2018])
    df = df[df['year'].isin(selected_years)]
else:
    selected_ym = st.sidebar.multiselect("연-월 선택", sorted(df['year_month'].unique()), default=[df['year_month'].iloc[0]])
    df = df[df['year_month'].isin(selected_ym)]

# (2) 센터 선택
selected_centers = st.sidebar.multiselect(
    "센터 선택",
    options=df["center_name"].unique().tolist(),
    default=df["center_name"].unique().tolist()[:3]
)

# (3) 품목 선택
category_columns = df.columns[2:13]  # 정확한 범위로 품목만 선택
selected_item = st.sidebar.selectbox("품목 선택", options=category_columns)

# -------------------------
# 4. 데이터 필터링
# -------------------------
filtered_df = df[df["center_name"].isin(selected_centers)]


# -------------------------
# 5. 시각화 (Plotly)
# -------------------------
import plotly.graph_objects as go

st.subheader(f"{selected_item} 일별 추이 (센터별 비교)")

if filtered_df.empty:
    st.warning("해당 조건에 맞는 데이터가 없습니다.")
else:
    pivot_df = filtered_df.pivot_table(index="date", columns="center_name", values=selected_item)

    # Plotly figure 생성
    fig = go.Figure()

    # 센터별로 선 추가
    for center in pivot_df.columns:
        fig.add_trace(go.Scatter(
            x=pivot_df.index,
            y=pivot_df[center],
            mode='lines+markers',
            name=center,
            line=dict(width=2),
            marker=dict(size=4)
        ))

    fig.update_layout(
        title=f"{selected_item} 일별 추이",
        xaxis_title="날짜",
        yaxis_title="물동량",
        hovermode="x unified",
        template="plotly_white",
        legend_title="센터"
    )

    st.plotly_chart(fig, use_container_width=True)


# # -------------------------
# # 5. 시각화 (Matplotlib) 
# # -------------------------
# st.subheader(f"{selected_item} 일별 추이 (센터별 비교)")

# if filtered_df.empty:
#     st.warning("해당 조건에 맞는 데이터가 없습니다.")
# else:
#     # 센터별로 날짜별 물동량 추이를 선 그래프로 비교
#     pivot_df = filtered_df.pivot_table(index="date", columns="center_name", values=selected_item)

#     fig, ax = plt.subplots(figsize=(12, 6))
#     pivot_df.plot(ax=ax, linewidth=2)  # 선 두께를 키워서 더 잘 보이게
#     ax.set_ylabel("물동량")
#     ax.set_xlabel("날짜")
#     ax.set_title(f"{selected_item} 일별 추이")
#     ax.legend(title="센터")
#     ax.grid(True)
#     plt.tight_layout()

#     st.pyplot(fig)

