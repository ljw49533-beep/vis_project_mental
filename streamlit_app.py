import streamlit as st
import pandas as pd

df = pd.read_csv('kchs_2024.csv', encoding='utf-8')

# 코드→의미 변환 딕셔너리 예시 (필요 컬럼별로 모두 추가, 아래 예시에서 변환할 컬럼만 복사 확장)

conversion_dicts = {
    # 성별
    'sex': {1: '남자', 2: '여자'},

    # 시도코드
    'CTPRVN_CODE': {
        11: '서울', 26: '부산', 27: '대구', 28: '인천', 29: '광주', 30: '대전', 31: '울산', 41: '경기',
        42: '강원', 43: '충북', 44: '충남', 45: '전북', 46: '전남', 47: '경북', 48: '경남', 49: '제주'
    },

    # 주관적 스트레스 수준
    'mta_01z1': {
        1: '대단히 많이 느낀다',
        2: '많이 느끼는 편이다',
        3: '조금 느끼는 편이다',
        4: '거의 느끼지 않는다',
        7: '응답거부',
        9: '모름'
    },
    # 전반적인 수면의 질
    'mtc_13z1': {
        1: '매우 좋음',
        2: '상당히 좋음',
        3: '상당히 나쁨',
        4: '매우 나쁨',
        9: '모름'
    },
    # 우울감 경험 여부
    'mtb_01z1': {
        1: '예',
        2: '아니오',
        7: '응답거부',
        9: '모름'
    },
    # 자살생각 경험 여부
    'mtd_01z1': {
        1: '예',
        2: '아니오',
        7: '응답거부',
        9: '모름'
    },
    # 자살생각으로 인한 정신상담 여부
    'mtd_02z1': {
        1: '예',
        2: '아니오',
        7: '응답거부',
        8: '비해당',
        9: '모름'
    },
    # 기초생활수급자 여부
    'fma_04z1': {
        1: '그렇다',
        2: '지금은 아니지만, 과거에 수급자였던 적이 있다',
        3: '아니다',
        7: '응답거부',
        9: '모름'
    },
    # 기타 캡쳐 변환 예시
    # (여기서 모든 컬럼 추가 가능, 각 이미지 내 코드를 딕셔너리로 정리해서 추가!)
}

# 모든 컬럼에 대해 의미변환 컬럼 자동 생성
for col, mapping in conversion_dicts.items():
    if col in df.columns:
        df[f'{col}_의미'] = df[col].map(mapping)

## Streamlit 필터 및 시각화 예시: (각 의미 컬럼을 바로 사용)
st.sidebar.header("응답 문항 필터")

if 'sex_의미' in df.columns:
    sex_options = sorted(df['sex_의미'].dropna().unique())
    sex_filter = st.sidebar.multiselect('성별', sex_options, sex_options)
else:
    sex_filter = []

if 'mta_01z1_의미' in df.columns:
    stress_options = [v for v in df['mta_01z1_의미'].dropna().unique() if v not in ['응답거부', '모름']]
    stress_filter = st.sidebar.multiselect('주관적 스트레스 수준', stress_options, stress_options)
else:
    stress_filter = []

if 'mtc_13z1_의미' in df.columns:
    sleep_options = [v for v in df['mtc_13z1_의미'].dropna().unique() if v != '모름']
    sleep_filter = st.sidebar.multiselect('전반적인 수면의 질', sleep_options, sleep_options)
else:
    sleep_filter = []

filtered = df.copy()
if sex_filter:
    filtered = filtered[filtered['sex_의미'].isin(sex_filter)]
if stress_filter:
    filtered = filtered[filtered['mta_01z1_의미'].isin(stress_filter)]
if sleep_filter:
    filtered = filtered[filtered['mtc_13z1_의미'].isin(sleep_filter)]

st.title("정신건강 및 수면 관련 문항 대시보드 (의미변환 적용)")

st.metric("필터 적용 대상자 수", filtered.shape[0])

if 'mtc_13z1_의미' in filtered.columns:
    fig = px.histogram(filtered, x='mtc_13z1_의미', title="전반적인 수면의 질")
    st.plotly_chart(fig)
if 'mta_01z1_의미' in filtered.columns:
    fig2 = px.histogram(filtered, x='mta_01z1_의미', title="스트레스 수준 응답 분포")
    st.plotly_chart(fig2)
if 'mtb_01z1_의미' in filtered.columns:
    fig3 = px.histogram(filtered, x='mtb_01z1_의미', title="우울감 경험 여부")
    st.plotly_chart(fig3)

st.subheader("미리보기(의미로 변환된 컬럼):")
preview_cols = [c for c in filtered.columns if c.endswith('_의미')]
st.dataframe(filtered[preview_cols + ['age']].head(20))
