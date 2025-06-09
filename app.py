import streamlit as st

# 페이지 기본 설정
st.set_page_config(
    page_title="📦 생활물류 물동량 대시보드",
    page_icon="📦",
    layout="wide",
)

# 메인 화면 (기본 안내)
st.title("📦 생활물류 물동량 대시보드")
st.markdown("""
안녕하세요!  
왼쪽 사이드바에서 분석 페이지를 선택해주세요.  

데이터 출처: `logistics_by_center.csv`  
분석 대상: 2018년 1월 ~ 2023년 12월, 일별 물류센터 품목별 생활물류 물동량
""")