import streamlit as st
import pandas as pd

df = pd.read_csv('kchs_2024.csv', encoding='utf-8')

# 각 컬럼 코드→의미 dict (예시, 필요시 dict 추가/수정)
conversion_dicts = {
    'sex': {1: '남자', 2: '여자'},
    'CTPRVN_CODE': {
        11: '서울', 26: '부산', 27: '대구', 28: '인천', 29: '광주', 30: '대전', 31: '울산', 41: '경기',
        42: '강원', 43: '충북', 44: '충남', 45: '전북', 46: '전남', 47: '경북', 48: '경남', 49: '제주'
    },
    'fma_04z1': {
        1: '그렇다',
        2: '지금은 아니지만, 과거에 수급자였던 적이 있다',
        3: '아니다',
        7: '응답거부',
        9: '모름'
    },
    'mta_01z1': {
        1: '대단히 많이 느낀다',
        2: '많이 느끼는 편이다',
        3: '조금 느끼는 편이다',
        4: '거의 느끼지 않는다',
        7: '응답거부',
        9: '모름'
    },
    'mtb_01z1': {1: '예', 2: '아니오', 7: '응답거부', 9: '모름'},
    'mtb_02z1': {1: '예', 2: '아니오', 7: '응답거부', 8: '비해당', 9: '모름'},
    'mtd_01z1': {1: '예', 2: '아니오', 7: '응답거부', 9: '모름'},
    'mtd_02z1': {1: '예', 2: '아니오', 7: '응답거부', 8: '비해당', 9: '모름'},
    'mtc_13z1': {1: '매우 좋음', 2: '상당히 좋음', 3: '상당히 나쁨', 4: '매우 나쁨', 9: '모름'},
    'mtc_14z1': {1: '전혀 없었다', 2: '한 주에 1번', 3: '한 주에 1~2번', 4: '한 주에 3번 이상', 7: '응답거부', 9: '모름'}
    # 필요시 다른 컬럼의 응답 dict도 여기에 추가!
}

all_cols = [
    'age', 'sex', 'CTPRVN_CODE', 'mbhld_co', 'reside_adult_co', 'fma_19z3', 'fma_04z1',
    'fma_13z1', 'nue_01z1', 'mtc_17z1', 'mtc_18z1', 'mta_01z1', 'mta_02z1',
    'mtb_01z1', 'mtb_02z1', 'mtd_01z1', 'mtd_02z1', 'edit_mtc_03z1', 'mtc_04z1',
    'mtc_05z1', 'mtc_06z1', 'mtc_08z1', 'mtc_09z1', 'mtc_13z1', 'mtc_14z1'
]

# 모든 컬럼 의미변환, 없으면 원본값 그대로 쓴다
display_cols = []
for col in all_cols:
    if col in conversion_dicts and col in df.columns:
        df[f'{col}_의미'] = df[col].map(conversion_dicts[col])
        display_cols.append(f'{col}_의미')
    else:
        display_cols.append(col)

# 필터 UI: 모든 컬럼을 반드시 포함 (고유값 리스트로 필터, 의미변환이 있으면 그 값으로)
st.sidebar.header("전 컬럼 필터")
user_filters = {}
for col in display_cols:
    options = sorted(df[col].dropna().unique())
    if df[col].dtype in [int, float] and len(options) > 10:
        min_val, max_val = min(options), max(options)
        user_filters[col] = st.sidebar.slider(col, int(min_val), int(max_val), (int(min_val), int(max_val)))
    else:
        user_filters[col] = st.sidebar.multiselect(col, options, options)

# 필터 적용
filtered = df.copy()
for col, sel in user_filters.items():
    if isinstance(sel, tuple) and df[col].dtype in [int, float]:
        filtered = filtered[(filtered[col] >= sel[0]) & (filtered[col] <= sel[1])]
    elif sel:
        filtered = filtered[filtered[col].isin(sel)]

st.title("KCHS 모든 변수(응답 의미/숫자) 대시보드")

st.metric("필터 적용 후 응답자 수", filtered.shape[0])
st.dataframe(filtered[display_cols].head(30))

for col in display_cols:
    fig = px.histogram(filtered, x=col, title=f"{col} 분포")
    st.plotly_chart(fig)

st.info(
    "모든 25개 컬럼을 사이드바에서 필터/분포 확인 가능하며, 코드형(variable)은 사람이 읽기 쉬운 문장으로 자동 의미 변환됨."
)
