import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv('kchs_clean_ready.csv', encoding='utf-8')

column_labels = {
    'age': '만 나이',
    'sex': '성별',
    'CTPRVN_CODE': '시도명',
    'mbhld_co': '가구원수 전체',
    'reside_adult_co': '가구원수 만 19세 이상',
    'fma_19z3': '세대 유형',
    'fma_04z1': '기초생활수급자 여부',
    'fma_13z1': '가구소득',
    'nue_01z1': '가구유형',
    'mtc_17z1': '하루 평균 수면시간(주중)',
    'mtc_18z1': '하루 평균 수면시간(주말)',
    'mta_01z1': '주관적 스트레스 수준',
    'mta_02z1': '스트레스로 인한 정신상담 여부',
    'mtb_01z1': '우울감 경험 여부',
    'mtb_02z1': '우울감으로 인한 정신상담 여부',
    'mtd_01z1': '자살생각 경험 여부',
    'mtd_02z1': '자살생각으로 인한 정신상담 여부',
    'fma_13z1': '가구소득',
    '수면 소요시간(분)': '수면 소요시간(분)',
    '잠자는 시각': '잠자는 시각',
    '기상 시각': '기상 시각'
}

response_maps = {
    'sex': {1: '남자', 2: '여자'},
    'CTPRVN_CODE': {11: '서울', 26: '부산', 27: '대구', 28: '인천', 29: '광주', 30: '대전', 31: '울산', 41: '경기', 42: '강원', 43: '충북', 44: '충남', 45: '전북', 46: '전남', 47: '경북', 48: '경남', 49: '제주'},
    'fma_19z3': {1: '1세대 가구', 2: '2세대 가구', 3: '3세대 이상 가구', 4: '부부', 5: '한부모', 6: '기타', 7: '응답거부'},
    'nue_01z1': {1: '1인', 2: '2인', 3: '3인', 4: '4인', 5: '5인', 6: '6인 이상', 99: '모름'},
    'fma_04z1': {1: '그렇다', 2: '지금은 아니지만, 과거에 수급자였던 적이 있다', 3: '아니다', 7: '응답거부', 9: '모름'},
    'mta_01z1': {1: '대단히 많이 느낀다', 2: '많이 느끼는 편이다', 3: '조금 느끼는 편이다', 4: '거의 느끼지 않는다'},
    'mta_02z1': {1: '예', 2: '아니오'},
    'mtb_01z1': {1: '예', 2: '아니오'},
    'mtb_02z1': {1: '예', 2: '아니오'},
    'mtd_01z1': {1: '예', 2: '아니오'},
    'mtd_02z1': {1: '예', 2: '아니오'}
}

display_cols = []
for code, label in column_labels.items():
    if code in response_maps and code in df.columns:
        df[label] = df[code].map(response_maps[code])
        display_cols.append(label)
    elif code in df.columns:
        df[label] = df[code]
        display_cols.append(label)

st.set_page_config(page_title="KCHS 30분단위 시각/소요시간 대시보드", layout="wide")
st.title("KCHS: 30분 단위 시각/수면 통합, 문항·의미 분석 대시보드")

st.sidebar.header("전체 설문문항 필터")
filters = {}
for label in display_cols:
    if label not in df.columns: continue
    options = [v for v in df[label].dropna().unique()]
    # 소요시간·주중/주말 수면시간은 슬라이더!
    if '분' in label and pd.api.types.is_numeric_dtype(df[label]) and len(options) > 10:
        min_val, max_val = float(min(options)), float(max(options))
        filters[label] = st.sidebar.slider(label, min_val, max_val, (min_val, max_val))
    # 30분단위 시각은 카테고리 멀티셀렉트
    elif label in ['잠자는 시각', '기상 시각']:
        options = sorted([v for v in df[label].dropna().unique()])
        filters[label] = st.sidebar.multiselect(label, options, default=options)
    else:
        filters[label] = st.sidebar.multiselect(label, options, default=options)

filtered = df[display_cols].copy()
for label, sel in filters.items():
    col = filtered[label]
    if isinstance(sel, tuple) and pd.api.types.is_numeric_dtype(col):
        filtered = filtered[(col >= sel[0]) & (col <= sel[1])]
    elif isinstance(sel, list) and len(sel) < len(df[label].dropna().unique()):
        filtered = filtered[col.isin(sel)]

st.metric("필터 적용 후 응답자 수", filtered.shape[0])
st.dataframe(filtered.head(30))

for label in display_cols:
    # 수치형 변수는 히스토그램(수면 소요, 주중/주말 등)
    if label in filtered.columns and filtered[label].notna().sum() > 0 and pd.api.types.is_numeric_dtype(filtered[label]):
        fig = px.histogram(filtered, x=label, title=f"{label} 응답 분포")
        st.plotly_chart(fig)
    # 30분단위 시각은 카운트로 빈도 보기
    elif label in ['잠자는 시각', '기상 시각'] and label in filtered.columns:
        fig = px.histogram(filtered, x=label, title=f"{label} (30분 단위) 응답 빈도")
        st.plotly_chart(fig)

st.info(
    "⏰ 잠자는 시각/기상 시각은 30분 단위(HH:00/HH:30/HH+1:00)로, 수면 소요시간은 합산 '분' 단위로, 모든 변수/응답은 한글로 표기 및 필터링됩니다."
)
