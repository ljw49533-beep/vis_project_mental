import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv('kchs_2024.csv', encoding='utf-8')

# 컬럼별 코드→의미 딕셔너리
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
    'mtb_02z1': {1: '예', 2: '아니오', 7: '응답거부', 8:'비해당', 9: '모름'},
    'mtd_01z1': {1: '예', 2: '아니오', 7: '응답거부', 9: '모름'},
    'mtd_02z1': {1: '예', 2: '아니오', 7: '응답거부', 8:'비해당', 9: '모름'},
    'mtc_13z1': {1: '매우 좋음', 2: '상당히 좋음', 3: '상당히 나쁨', 4: '매우 나쁨', 9: '모름'},
    'mtc_14z1': {1: '전혀 없었다', 2: '한 주에 1번', 3: '한 주에 1~2번', 4: '한 주에 3번 이상', 7: '응답거부', 9: '모름'},
    # 필요에 따라 모든 코드형 컬럼을 추가하세요.
}

# 모든 컬럼 의미변환 적용
for col in df.columns:
    if col in conversion_dicts and col in df.columns:
        df[f'{col}_의미'] = df[col].map(conversion_dicts[col])

# 필터 UI (모든 컬럼에 대해 의미 컬럼 있으면 필터 제공)
st.sidebar.header("모든 변수 응답 의미 필터")

filter_cols = [col for col in df.columns if col.endswith('_의미')]
sidebar_filters = {}
for col in filter_cols:
    options = [v for v in df[col].dropna().unique() if v not in ['모름', '응답거부']]
    sidebar_filters[col] = st.sidebar.multiselect(col.replace('_의미',''), options, options)

# 필터 적용
filtered = df.copy()
for col, options in sidebar_filters.items():
    if options:
        filtered = filtered[filtered[col].isin(options)]

st.title("정신건강·수면·생활 조사 전 컬럼 의미변환 대시보드")

st.metric("필터 적용 응답자 수", filtered.shape[0])

# 한눈에 표로 모든 컬럼 보여주기 (숫자형 포함, 의미변환 된 컬럼 포함)
preview_cols = [col for col in df.columns if col.endswith('_의미')] + \
    [c for c in ['age','mbhld_co','reside_adult_co','fma_19z3','fma_13z1','nue_01z1','mtc_17z1','mtc_18z1','edit_mtc_03z1','mtc_04z1','mtc_05z1','mtc_06z1','mtc_08z1','mtc_09z1'] if c in df.columns]

st.subheader("응답 데이터 미리보기 (모든 문항 의미 변환)")
st.dataframe(filtered[preview_cols].head(30))

# 대표 시각화 예시
for col in filter_cols:
    if col in filtered.columns:
        fig = px.histogram(filtered, x=col, title=f"{col.replace('_의미','')} 분포")
        st.plotly_chart(fig)

st.info(
    "✅ 모든 변수의 응답 코드를 직관적인 문장(의미변환)으로 실시간 필터·분포·표에서 보여줍니다.\n"
    "❗ 코드형 컬럼은 dict만 추가하면 바로 처리, 숫자형은 결과 자체 표시됨."
)
