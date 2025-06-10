import pandas as pd

def load_logistics_data(filepath: str) -> pd.DataFrame:
    """
    주어진 CSV 파일 경로에서 물류 데이터를 불러오는 함수입니다.

    Parameters:
    - filepath (str): CSV 파일 경로

    Returns:
    - pd.DataFrame: 'date' 컬럼은 datetime 형식으로 변환된 DataFrame
    """
    df = pd.read_csv(filepath, encoding="euc-kr")

    # 'date' 컬럼이 문자열 형식이라면 datetime 형식으로 변환
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

    return df