# 📦 코드 분리	각 페이지가 시각화 로직으로부터 독립됨
# 🔁 재사용성 증가	동일 그래프를 여러 페이지에서 쉽게 활용
# 🧪 유닛 테스트 가능	시각화 함수 단위로 테스트/개선 용이

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def line_chart_by_center(pivot_df: pd.DataFrame, item_name: str) -> go.Figure:
    """
    센터별 품목의 시계열 추이를 선 그래프로 시각화합니다.

    Parameters:
    - pivot_df: date x center_name 형태의 데이터프레임
    - item_name: 품목 이름 (그래프 제목용)

    Returns:
    - plotly.graph_objects.Figure
    """
    fig = go.Figure()
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
        title=f"{item_name} 일별 추이",
        xaxis_title="날짜",
        yaxis_title="물동량",
        hovermode="x unified",
        template="plotly_white",
        legend_title="센터"
    )
    return fig


def bar_chart_by_item(melted_df: pd.DataFrame) -> px.bar:
    """
    센터별 품목 물동량을 막대그래프로 시각화합니다.

    Parameters:
    - melted_df: center_name, 품목, 물동량 컬럼을 가진 긴 형태의 데이터프레임

    Returns:
    - plotly.express.bar 객체
    """
    fig = px.bar(
        melted_df,
        x="center_name",
        y="물동량",
        color="품목",
        barmode="group",
        text_auto=".2s",
        title="센터별 품목 물동량 비교"
    )
    fig.update_layout(
        xaxis_title="센터명",
        yaxis_title="물동량",
        legend_title="품목",
        template="plotly_white"
    )
    return fig
