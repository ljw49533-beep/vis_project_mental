import streamlit as st
import pandas as pd
import plotly.express as px

# 1. 데이터 읽기
df = pd.read_csv('kchs_2024_1.csv', encoding='utf-8')

st.set_page_config(page_title="KCHS 수면·정신건강 대시보드", layout="wide")

# ---- Sidebar Layout: 핵심 필터 ----
st.sidebar.header("인구통계/건강 상태별 필터")

# 주요 변수 추출
sleep_col = 'mtc_17z1'      # 평일 평균 수면(시간)
stress_col = 'mta_01z1'     # 주관적 스트레스 수준
depress_col = 'mtb_01z1'    # 우울감 경험 여부

# 필터 위젯
min_sleep = int(df[sleep_col].min())
max_sleep = int(df[sleep_col].max())
sleep_range = st.sidebar.slider("주중 평균 수면시간 (시간)", min_sleep, max_sleep, (min_sleep, max_sleep))

stress_options = sorted(df[stress_col].dropna().unique())
stress_filter = st.sidebar.multiselect("스트레스 수준", stress_options, stress_options)
depress_options = sorted(df[depress_col].dropna().unique())
depress_filter = st.sidebar.multiselect("우울감 경험", depress_options, depress_options)

# 필터 적용
filtered = df[
    (df[sleep_col] >= sleep_range[0]) & (df[sleep_col] <= sleep_range[1]) &
    (df[stress_col].isin(stress_filter)) &
    (df[depress_col].isin(depress_filter))
]

# ---- Main Layout: KPI 및 인사이트 요약 ----
st.title("KCHS 수면·정신건강 인사이트 대시보드")
st.caption("국민건강조사(샘플) 데이터 기반 실시간 분석")

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("필터 적용 행 개수", len(filtered))
kpi2.metric("평균 수면시간(주중)", f"{filtered[sleep_col].mean():.1f}시간")
kpi3.metric("우울감 경험률", f"{(filtered[depress_col].eq(depress_options[-1]).mean()*100):.1f}%")

# ---- 1. 수면시간 분포(히스토그램) ----
st.subheader("주중 평균 수면시간 분포")
fig1 = px.histogram(filtered, x=sleep_col, nbins=15, title="수면시간 히스토그램")
st.plotly_chart(fig1, use_container_width=True)

# ---- 2. 스트레스 수준별 수면시간(박스플롯) ----
st.subheader("스트레스 수준별 주중 수면시간")
fig2 = px.box(filtered, x=stress_col, y=sleep_col, color=stress_col, points="outliers", title="스트레스별 수면시간 분포")
st.plotly_chart(fig2, use_container_width=True)

# ---- 3. 우울감 경험별 수면 분포(박스플롯) ----
st.subheader("우울감 경험별 수면 분포")
fig3 = px.box(filtered, x=depress_col, y=sleep_col, color=depress_col, points="outliers", title="우울감 경험에 따른 수면시간")
st.plotly_chart(fig3, use_container_width=True)

# ---- 4. 상세 데이터 프리뷰 ----
with st.expander("상세 데이터 미리보기"):
    st.dataframe(filtered.head(50))

st.info(
    "✅ [최적의 베이스: KPI→분포→교차분석→프리뷰… 모든 시각화에 필터가 실시간 적용]\n"
    "✅ [남녀/연령 등 인구통계 칼럼이 있으면 sidebar multiselect로 바로 확장 가능]"
)
