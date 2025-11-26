import streamlit as st
import pandas as pd
import plotly.express as px

# 데이터 불러오기 (같은 폴더의 csv)
df = pd.read_csv('kchs_2024_1.csv', encoding='utf-8')

st.title("KCHS 수면·정신건강 데이터 대시보드 샘플")

# 필터: 수면시간
sleep_min = int(df['mtc_17z1'].min())
sleep_max = int(df['mtc_17z1'].max())
sleep_time = st.slider('주중 평균 수면시간 (시간)', sleep_min, sleep_max, (sleep_min, sleep_max))

# 필터 적용
filtered = df[(df['mtc_17z1'] >= sleep_time[0]) & (df['mtc_17z1'] <= sleep_time[1])]

st.write(f"필터 적용 후 데이터 행수: {filtered.shape[0]}")

# 분포 시각화 (히스토그램)
fig_sleep = px.histogram(filtered, x='mtc_17z1', nbins=15, title='주중 평균 수면시간 분포')
st.plotly_chart(fig_sleep)

# 추가 그래프 (예: 스트레스 수준별 박스플롯)
if 'mta_01z1' in df.columns:
    fig_box = px.box(df, x='mta_01z1', y='mtc_17z1', title='스트레스 수준별 주중 수면시간')
    st.plotly_chart(fig_box)

# 데이터 표 미리보기
st.subheader("데이터 샘플 미리보기")
st.dataframe(filtered.head(10))
