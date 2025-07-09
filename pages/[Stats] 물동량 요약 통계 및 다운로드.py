import streamlit as st
import pandas as pd
import sys
import os

# src 경로 추가 및 데이터 로더 import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.loader import load_logistics_data

# -------------------------
# 1. 페이지 설정
# -------------------------
st.set_page_config(page_title="Data Summary", layout="wide")
st.title("📊 물동량 요약 통계 및 다운로드")

# -------------------------
# 2. 데이터 로딩
# -------------------------
DATA_PATH = "data/logistics_by_center.csv"

@st.cache_data
def load_data():
    return load_logistics_data(DATA_PATH)

df = load_data()

# -------------------------
# 3. 필터 옵션
# -------------------------
st.sidebar.header("필터 옵션")

# 센터 선택
selected_centers = st.sidebar.multiselect(
    "센터 선택",
    options=df["center_name"].unique().tolist(),
    default=df["center_name"].unique().tolist()
)

# 품목 선택
category_columns = df.columns[2:13]
selected_items = st.sidebar.multiselect(
    "품목 선택",
    options=category_columns,
    default=list(category_columns)
)

# -------------------------
# 4. 데이터 필터링
# -------------------------
filtered_df = df[df["center_name"].isin(selected_centers)]

# -------------------------
# 5. 요약 통계 계산
# -------------------------
if filtered_df.empty:
    st.warning("선택된 센터에 해당하는 데이터가 없습니다.")
else:
    st.subheader("📈 선택된 센터 및 품목의 요약 통계")

    # 그룹: 센터 × 품목별 평균/표준편차/최소/최대
    summary = filtered_df.groupby("center_name")[selected_items].agg(["mean", "std", "min", "max"])
    summary.columns = ['_'.join(col) for col in summary.columns]  # 다중 컬럼 flatten

    st.dataframe(summary.round(2), use_container_width=True)

    # -------------------------
    # 6. CSV 다운로드
    # -------------------------
    csv = summary.to_csv().encode("utf-8-sig")

    st.download_button(
        label="⬇️ 요약 통계 CSV 다운로드",
        data=csv,
        file_name="logistics_summary.csv",
        mime="text/csv"
    )
