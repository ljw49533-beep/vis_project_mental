import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv('kchs_2024.csv', encoding='utf-8')

# 변수 코드명 → 한글 질문 평가
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
    'edit_mtc_03z1': '잠자는 시각(시)',
    'mtc_04z1': '잠자는 시각(분)',
    'mtc_05z1': '수면 소요시간(시)',
    'mtc_06z1': '수면 소요시간(분)',
    'mtc_08z1': '기상 시각(시)',
    'mtc_09z1': '기상 시각(분)',
    'mtc_13z1': '전반적인 수면의 질',
    'mtc_14z1': '수면을 위한 약 복용'
}

response_maps = {
    'sex': {1: '남자', 2: '여자'},
    'CTPRVN_CODE': {
        11: '서울', 26: '부산', 27: '대구', 28: '인천', 29: '광주', 30: '대전', 31: '울산', 41: '경기',
        42: '강원', 43: '충북', 44: '충남', 45: '전북', 46: '전남', 47: '경북', 48: '경남', 49: '제주'
    },
    'fma_19z3': {
        1: '1세대 가구',
        2: '2세대 가구',
        3: '3세대 이상 가구',
        4: '부부',
        5: '한부모',
        6: '기타',
        7: '응답거부'
    },
    'nue_01z1': {
        1: '1인',
        2: '2인',
        3: '3인',
        4: '4인',
        5: '5인',
        6: '6인 이상',
        99: '모름'
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
        4: '거의 느끼지 않는다'
    },
    'mta_02z1': {
        1: '예',
        2: '아니오'
    },
    'mtb_01z1': {1: '예', 2: '아니오'},
    'mtb_02z1': {1: '예', 2: '아니오'},
    'mtd_01z1': {1: '예', 2: '아니오'},
    'mtd_02z1': {1: '예', 2: '아니오'},
    'mtc_13z1': {1: '매우 좋음', 2: '상당히 좋음', 3: '상당히 나쁨', 4: '매우 나쁨'},
    'mtc_14z1': {1: '전혀 없었다', 2: '한 주에 1번', 3: '한 주에 1~2번', 4: '한 주에 3번 이상'}
    # 추가/수정 사항 있으면 response_maps에만 바꿔서 확장
}

display_cols = []
for code, label in column_labels.items():
    # 원본 값 확인(디버깅용, 문제 진단)
    if code in df.columns:
        st.write(f"{label}({code}) 값 분포:", df[code].value_counts().to_dict())
    if code in response_maps and code in df.columns:
        df[label] = df[code].map(response_maps[code])
        display_cols.append(label)
    elif code in df.columns:
        df[label] = df[code]
        display_cols.append(label)

st.sidebar.header("전체 문항 필터(한글/의미)")
filters = {}
for label in display_cols:
    options = [v for v in df[label].dropna().unique() if v not in ['모름', '응답거부', '', None]]
    if df[label].dtype in ['int64', 'float64'] and len(options) > 10:
        min_val, max_val = float(min(options)), float(max(options))
        filters[label] = st.sidebar.slider(label, min_val, max_val, (min_val, max_val))
    else:
        filters[label] = st.sidebar.multiselect(label, options, options)

filtered = df[display_cols].copy()
for label, sel in filters.items():
    if isinstance(sel, tuple) and filtered[label].dtype in ['int64', 'float64']:
        filtered = filtered[(filtered[label] >= sel[0]) & (filtered[label] <= sel[1])]
    elif len(sel) > 0:
        filtered = filtered[filtered[label].isin(sel)]

st.title("KCHS 모든 설문 문항(한글) 및 응답 의미 시각화")
st.metric("필터 적용 후 응답자 수", filtered.shape[0])
st.dataframe(filtered.head(30))

for label in display_cols:
    if filtered[label].notna().sum() > 0:
        fig = px.histogram(filtered, x=label, title=f"{label} 응답 분포")
        st.plotly_chart(fig)

st.info(
    "✅ 모든 설문 문항이 한글 질문명과 응답 의미로 표시됩니다. 문제 응답 확인 결과는 위 표 지침대로 자동 변환/필터. 전처리/원본값 이상 있을 땐 dictionary와 value_counts 참고."
)
