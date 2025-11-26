import streamlit as st
import pandas as pd

df = pd.read_csv('kchs_2024.csv', encoding='utf-8')

# 사진에서 정의된 변수 → 한글 질문 mapping
column_labels = {
    'age': '만 나이',
    'sex': '성별',
    'CTPRVN_CODE': '시도명',
    'mbhld_co': '가구원수 전체',
    'reside_adult_co': '가구원수 만 19세 이상',
    'fma_19z3': '세대유형',
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
    'edit_mtc_03z1': '잠자는 시각(시)',
    'mtc_04z1': '잠자는 시각(분)',
    'mtc_05z1': '수면 소요시간(시)',
    'mtc_06z1': '수면 소요시간(분)',
    'mtc_08z1': '기상 시각(시)',
    'mtc_09z1': '기상 시각(분)',
    'mtc_13z1': '전반적인 수면의 질',
    'mtc_14z1': '수면을 위한 약 복용'
}

# 주요 코드형 변수 응답 → 의미 (사진 내 문항)
response_maps = {
    'sex': {1: '남자', 2: '여자'},
    'CTPRVN_CODE': {
        11: '서울', 26: '부산', 27: '대구', 28: '인천', 29: '광주', 30: '대전', 31: '울산', 41: '경기',
        42: '강원', 43: '충북', 44: '충남', 45: '전북', 46: '전남', 47: '경북', 48: '경남', 49: '제주'
    },
    'fma_04z1': {
        1: '그렇다',
        2: '지금은 아니지만, 과거에 수급자였던 적이 있다',
        3: '아니다'
    },
    'mta_01z1': {
        1: '대단히 많이 느낀다',
        2: '많이 느끼는 편이다',
        3: '조금 느끼는 편이다',
        4: '거의 느끼지 않는다'
    },
    'mtb_01z1': {1: '예', 2: '아니오'},
    'mtb_02z1': {1: '예', 2: '아니오'},
    'mtd_01z1': {1: '예', 2: '아니오'},
    'mtd_02z1': {1: '예', 2: '아니오'},
    'mtc_13z1': {1: '매우 좋음', 2: '상당히 좋음', 3: '상당히 나쁨', 4: '매우 나쁨'},
    'mtc_14z1': {1: '전혀 없었다', 2: '한 주에 1번', 3: '한 주에 1~2번', 4: '한 주에 3번 이상'}
    # 기타 수치(나이, 인원, 시계/분 등)는 숫자 그대로 의미가 명확하므로 변환 불필요.
}

# 응답 코드형 컬럼 → 의미로 변환, 기타는 숫자 그대로
display_dict = {}
for col in column_labels:
    label = column_labels[col]
    if col in response_maps and col in df.columns:
        df[label] = df[col].map(response_maps[col])
        display_dict[label] = df[label]
    elif col in df.columns:
        df[label] = df[col] # 숫자는 의미 그대로
        display_dict[label] = df[label]

# Streamlit 사이드바에 무조건 모든 문항(한글 풀네임)으로 필터 등장
st.sidebar.header("모든 설문 문항(한글) 필터")
filters = {}
for label in column_labels.values():
    if label in df.columns:
        options = sorted(df[label].dropna().unique())
        if df[label].dtype in [int, float] and len(options) > 10:
            min_val, max_val = min(options), max(options)
            filters[label] = st.sidebar.slider(label, float(min_val), float(max_val), (float(min_val), float(max_val)))
        else:
            filters[label] = st.sidebar.multiselect(label, options, options)

# 필터 적용
filtered = df.copy()
for label, sel in filters.items():
    if isinstance(sel, tuple) and label in filtered.columns and filtered[label].dtype in [int, float]:
        filtered = filtered[(filtered[label] >= sel[0]) & (filtered[label] <= sel[1])]
    elif label in filtered.columns and len(sel) > 0:
        filtered = filtered[filtered[label].isin(sel)]

st.title("KCHS 모든 설문 문항(한글) 및 응답 의미 대시보드")

st.metric("필터 적용 응답자 수", filtered.shape[0])
st.dataframe(filtered[list(column_labels.values())].head(30))

# 주요 문항별 한글 분포 시각화
for label in column_labels.values():
    if label in filtered.columns:
        fig = px.histogram(filtered, x=label, title=f"{label} 분포")
        st.plotly_chart(fig)

st.info(
    "모든 변수는 코드명이 아닌 실제 설문 한글 문항, 응답도 사람이 바로 해석할 수 있는 의미로 변환되어 표시됩니다. 빈값·모름·응답거부는 기본적으로 제외."
)
