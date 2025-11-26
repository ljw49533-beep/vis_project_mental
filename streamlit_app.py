import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --------------------
# 데이터 불러오기
# --------------------
df = pd.read_csv("kchs_clean_ready.csv", encoding="utf-8")

# --------------------
# 코드 → 라벨 매핑
# --------------------
column_labels = {
    "age": "만 나이",
    "sex": "성별",
    "CTPRVN_CODE": "시도명",
    "mbhld_co": "가구원수 전체",
    "reside_adult_co": "가구원수 만 19세 이상",
    "fma_19z3": "세대 유형",
    "fma_04z1": "기초생활수급자 여부",
    "fma_13z1": "가구소득",
    "nue_01z1": "가구유형",
    "mtc_17z1": "하루 평균 수면시간(주중)",
    "mtc_18z1": "하루 평균 수면시간(주말)",
    "mta_01z1": "주관적 스트레스 수준",
    "mta_02z1": "스트레스로 인한 정신상담 여부",
    "mtb_01z1": "우울감 경험 여부",
    "mtb_02z1": "우울감으로 인한 정신상담 여부",
    "mtd_01z1": "자살생각 경험 여부",
    "mtd_02z1": "자살생각으로 인한 정신상담 여부",
    "mtc_06z1": "수면 소요시간(분)",
    "잠자는 시각": "잠자는 시각",
    "기상 시각": "기상 시각",
}

response_maps = {
    "sex": {1: "남자", 2: "여자"},
    "CTPRVN_CODE": {
        11: "서울", 26: "부산", 27: "대구", 28: "인천", 29: "광주",
        30: "대전", 31: "울산", 41: "경기", 42: "강원", 43: "충북",
        44: "충남", 45: "전북", 46: "전남", 47: "경북", 48: "경남", 49: "제주",
    },
    "fma_19z3": {
        1: "1세대 가구", 2: "2세대 가구", 3: "3세대 이상 가구",
        4: "부부", 5: "한부모", 6: "기타", 7: "응답거부",
    },
    "nue_01z1": {
        1: "1인", 2: "2인", 3: "3인", 4: "4인",
        5: "5인", 6: "6인 이상", 99: "모름",
    },
    "fma_04z1": {
        1: "그렇다",
        2: "지금은 아니지만, 과거에 수급자였던 적이 있다",
        3: "아니다", 7: "응답거부", 9: "모름",
    },
    "mta_01z1": {
        1: "대단히 많이 느낀다",
        2: "많이 느끼는 편이다",
        3: "조금 느끼는 편이다",
        4: "거의 느끼지 않는다",
    },
    "mta_02z1": {1: "예", 2: "아니오"},
    "mtb_01z1": {1: "예", 2: "아니오"},
    "mtb_02z1": {1: "예", 2: "아니오"},
    "mtd_01z1": {1: "예", 2: "아니오"},
    "mtd_02z1": {1: "예", 2: "아니오"},
}

# --------------------
# 라벨 컬럼 생성
# --------------------
display_cols = []
for code, label in column_labels.items():
    if code in response_maps and code in df.columns:
        df[label] = df[code].map(response_maps[code])
        display_cols.append(label)
    elif code in df.columns:
        df[label] = df[code]
        display_cols.append(label)

# --------------------
# 시간값 정렬 함수
# --------------------
def time_order_sort(times):
    def time_to_minutes(s):
        if isinstance(s, str) and ":" in s:
            h, m = s.split(":")
            return int(h) * 60 + int(m)
        return float("inf")

    valid_times = [t for t in times if t is not None and pd.notnull(t)]
    return sorted(valid_times, key=time_to_minutes)

# --------------------
# 페이지 설정
# --------------------
st.set_page_config(
    page_title="KCHS 분석 대시보드(전체 분포)",
    layout="wide",
)
st.title("KCHS | 전체 분포 확인 대시보드")

# --------------------
# 나이 10살 단위 구간 생성
# --------------------
if "만 나이" in df.columns:
    age_min = int(np.nanmin(df["만 나이"]))
    age_max = int(np.nanmax(df["만 나이"]))
    bin_edges = list(range(age_min // 10 * 10, age_max + 10, 10))
    df["나이 구간(10살 단위)"] = pd.cut(df["만 나이"], bins=bin_edges, right=False)
    df["나이 구간(10살 단위)_str"] = df["나이 구간(10살 단위)"].astype(str)
    age_bins_str = [
        str(interval)
        for interval in sorted(df["나이 구간(10살 단위)"].dropna().unique())
    ]
else:
    age_bins_str = []

# --------------------
# 데이터 미리보기
# --------------------
st.subheader("데이터 미리보기")
st.dataframe(df.head(30))

# --------------------
# 10살 단위 나이 분포
# --------------------
if "나이 구간(10살 단위)_str" in df.columns:
    st.subheader("10살 단위 나이별 분포")
    fig_age = px.histogram(
        df,
        x="나이 구간(10살 단위)_str",
        category_orders={"나이 구간(10살 단위)_str": age_bins_str},
    )
    st.plotly_chart(fig_age, use_container_width=True)

# --------------------
# 가구소득 분포 (이상치 제거)
# --------------------
if "가구소득" in df.columns:
    st.subheader("가구소득(0~20000만원, 이상치 제거) 분포")
    soc_col = df["가구소득"]
    outlier_vals = [90000, 99999, 77777, 88888, 9999, None, np.nan]
    minval, maxval = 0, 20000
    soc_clean = soc_col[~soc_col.isin(outlier_vals)]
    soc_clean = soc_clean[(soc_clean >= minval) & (soc_clean <= maxval)]
    if len(soc_clean) > 0:
        fig_soc = px.histogram(
            soc_clean,
            x=soc_clean,
            nbins=40,
        )
        fig_soc.update_xaxes(range=[minval, maxval])
        st.plotly_chart(fig_soc, use_container_width=True)

# --------------------
# 수면 소요시간(분): 이상치 제거 + 15분 단위 구간 (0~360분)
# --------------------
if "수면 소요시간(분)" in df.columns:
    st.subheader("수면 소요시간(분) 분포 (0~360분, 15분 단위, 이상치 제거)")

    sleep_dur = pd.to_numeric(df["수면 소요시간(분)"], errors="coerce")
    sleep_dur = sleep_dur.dropna()

    if len(sleep_dur) > 0:
        # 이상치 제거: 0분 이하, 360분(6시간) 초과 값 제거
        sleep_dur_clean = sleep_dur[(sleep_dur > 0) & (sleep_dur <= 360)]

        if len(sleep_dur_clean) > 0:
            # 15분 단위 구간: 0,15,...,360
            bins = list(range(0, 361, 15))
            labels = [f"{b}-{b+15}" for b in bins[:-1]]

            sleep_bins = pd.cut(
                sleep_dur_clean, bins=bins, labels=labels, right=False
            )

            sleep_df = pd.DataFrame(
                {"수면 소요시간(15분 구간)": sleep_bins}
            ).dropna()

            fig_sd = px.histogram(
                sleep_df,
                x="수면 소요시간(15분 구간)",
                title="수면 소요시간(분) 분포 (0~360분, 15분 단위 구간)",
            )
            st.plotly_chart(fig_sd, use_container_width=True)
        else:
            st.write("0~360분 구간 내 수면 소요시간 데이터가 거의 없습니다.")

# --------------------
# 하루 평균 수면시간(주중/주말) 기본 분포
# --------------------
for col in ["하루 평균 수면시간(주중)", "하루 평균 수면시간(주말)"]:
    if col in df.columns:
        st.subheader(f"{col} 분포")
        col_val = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(col_val) > 0:
            fig_col = px.histogram(
                col_val,
                x=col_val,
                nbins=40,
            )
            st.plotly_chart(fig_col, use_container_width=True)

# --------------------
# 시간 변수(잠자는 시각, 기상 시각) 분포
# --------------------
for label in ["잠자는 시각", "기상 시각"]:
    if label in df.columns:
        times_sorted = time_order_sort(df[label].dropna().unique())
        if len(times_sorted) > 0:
            st.subheader(f"{label} 분포 (시간순 정렬)")
            fig_t = px.histogram(
                df,
                x=label,
                category_orders={label: times_sorted},
            )
            st.plotly_chart(fig_t, use_container_width=True)

st.info(
    "이 페이지는 사이드바 필터 없이, 전체 데이터의 분포를 한눈에 보기 위한 구조 파악용 페이지입니다. "
    "수면 소요시간(분)은 0~360분(6시간)만 남기고 15분 단위로 구간화하여 표시했습니다."
)
