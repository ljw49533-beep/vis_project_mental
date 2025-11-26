import streamlit as st
import pandas as pd
import plotly.express as px

# 1. 데이터 로드
df = pd.read_csv('kchs_sleep_mental_selected.csv', encoding='utf-8')

# 2. 코드→의미 변환 딕셔너리 예시 (필요 컬럼별로 생성)
mtc_13z1_map = {
    1: '매우 좋음',
    2: '상당히 좋음',
    3: '상당히 나쁨',
    4: '매우 나쁨',
    9: '모름'
}
# 다른 변수도 위와 같은 방식으로 변환 dict 추가

# 3. 의미 컬럼 추가 (예시: 데이터가 숫자 코드면 아래처럼 변환)
if 'mtc_13z1' in df.columns:
    df['전반적인 수면의 질'] = df['mtc_13z1'].map(mtc_13z1_map)

# 4. Streamlit 사이드바 필터 (컬럼명이 사람이 이해할 수 있도록)
sleep_quality_options = [v for k,v in mtc_13z1_map.items() if v != '모름']
sleep_quality = st.sidebar.multiselect('전반적인 수면의 질 선택', sleep_quality_options, sleep_quality_options)

filtered = df[df['전반적인 수면의 질'].isin(sleep_quality)]

# 5. 시각화 예시
st.title("전반적인 수면의 질 분석 대시보드")

st.metric("분석 대상 응답자 수", filtered.shape[0])

# 수면질별 응답 분포
fig = px.histogram(filtered, x='전반적인 수면의 질', title="수면의 질 응답 분포")
st.plotly_chart(fig)

# 다른 변수도 위와 같이 코드-의미 변환
if 'mta_01z1' in df.columns:
    mta_01z1_map = {
        1: '대단히 많이 느낀다',
        2: '많이 느끼는 편이다',
        3: '조금 느끼는 편이다',
        4: '거의 느끼지 않는다',
        7: '응답거부',
        9: '모름'
    }
    df['주관적 스트레스 수준'] = df['mta_01z1'].map(mta_01z1_map)
    stress_options = [v for k,v in mta_01z1_map.items() if v not in ['응답거부','모름']]
    stress_filter = st.sidebar.multiselect('주관적 스트레스 수준', stress_options, stress_options)
    filtered = filtered[filtered['주관적 스트레스 수준'].isin(stress_filter)]

    # 스트레스 수준별 수면질 응답
    fig2 = px.box(filtered, x='주관적 스트레스 수준', y='mtc_17z1', title='스트레스별 수면시간')
    st.plotly_chart(fig2)

# 데이터 테이블
st.subheader('응답별 데이터 미리보기')
st.dataframe(filtered.head(20))

st.info(
    "모든 변수는 응답 코드가 아닌 사람이 이해하기 쉬운 문장으로 필터 및 그래프에 자동 반영됩니다."
)

