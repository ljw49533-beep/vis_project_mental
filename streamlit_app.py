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

st.set_page_config(page_title="KCHS 수면·기상 30분단위 시간순 대시보드", layout="wide")
st.title("KCHS 대시보드 | 수면·기상 시각 시간순 정렬 + 필터/분석")

def time_order_sort(times):
    # HH:MM 형태 문자열을 분으로 변환해서 정렬
    def time_to_minutes(s):
        if isinstance(s, str) and ':' in s:
            h, m = s.split(":")
            return int(h)*60 + int(m)
        return float('inf')
    return sorted([t for t in times if t is not None and pd.notnull(t)], key=time_to_minutes)

st.sidebar.header("전체 설문문항 필터")
filters = {}
for label in display_cols:
    if label not in df.columns: continue
    options = [v for v in df[label].dropna().unique()]
    # 시간순 정렬 (잠자는, 기상 시각)
    if label in ['잠자는 시각', '기상 시각']:
        sorted_options = time_order_sort(options)
        filters[label] = st.sidebar.multiselect(label, sorted_options, default=sorted_options)
    # 분 단위·수면시간 등은 슬라이더
    elif '분' in label and pd.api.types.is_numeric_dtype(df[label]) and len(options) > 10:
        min_val, max_val = float(min(options)), float(max(options))
        filters[label] = st.sidebar.slider(label, min_val, max_val, (min_val, max_val))
    else:
        filters[label] = st.sidebar.multiselect(label, sorted(options), default=sorted(options))

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
    # 수면 소요/평균 등은 히스토그램, 시각은 시간순 x축으로 정렬
    if label in filtered.columns and filtered[label].notna().sum() > 0 and pd.api.types.is_numeric_dtype(filtered[label]):
        fig = px.histogram(filtered, x=label, title=f"{label} 응답 분포")
        st.plotly_chart(fig)
    elif label in ['잠자는 시각', '기상 시각'] and label in filtered.columns:
        times_sorted = time_order_sort(filtered[label].dropna().unique())
        fig = px.histogram(filtered, x=label, category_orders={label: times_sorted}, title=f"{label} (시간순 정렬)")
        st.plotly_chart(fig)

st.info(
    "잠자는 시각·기상 시각은 30분 단위(HH:00, HH:30, HH+1:00)로 시간순으로 자동 정렬된 필터/분포/표가 제공됩니다."
)

import numpy as np
import plotly.express as px

# 가구소득 분포 그래프 개선(이 부분만 붙이면 됨)
if '가구소득' in filtered.columns:
    soc_col = filtered['가구소득']
    # 이상치 코드 + 실질적으로 분석할 최대/최소 값 기준
    outlier_vals = [90000, 99999, 77777, 88888, 9999, None, np.nan]
    minval, maxval = 0, 20000   # 원하는 분석구간
    soc_clean = soc_col[~soc_col.isin(outlier_vals)]
    soc_clean = soc_clean[(soc_clean >= minval) & (soc_clean <= maxval)]

    st.subheader("가구소득(0~20000만원 구간, 이상치 제거) 분포")
    # 시각적으로 더 명확하게 보이게 bin개수 조정 및 x/y축 표시
    fig_soc = px.histogram(soc_clean, x='가구소득', title='가구소득 응답 분포(정상 구간, 이상치 완전 제외)', nbins=40)
    fig_soc.update_xaxes(range=[minval, maxval])
    st.plotly_chart(fig_soc)

    st.caption("외곽치(90000, 99999, 77777, 88888 등) 및 상식적 분석구간 이외 값 완전 제거, 실질 분포만 강조!")
