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
}

# 라벨 컬럼 생성
for code, label in column_labels.items():
    if code in df.columns:
        if code in response_maps:
            df[label] = df[code].map(response_maps[code])
        else:
            df[label] = df[code]

# 만 나이 10살 구간
if "age" in df.columns:
    if "만 나이" not in df.columns:
        df["만 나이"] = df["age"]
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

# 시간값 정렬 함수
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
    page_title="KCHS 캐릭터 변수 분포",
    layout="wide",
)
st.title("KCHS | 캐릭터 변수 전체 분포 대시보드")

st.markdown(
    "- 이 페이지는 사람의 캐릭터(인구학·수면·소득 등)를 표현하는 변수들의 전체 분포를 한 번에 보여줍니다.\n"
    "- 필터 없이 전체 데이터 기준 분포입니다."
)

# --------------------
# 데이터 미리보기
# --------------------
st.subheader("데이터 미리보기")
preview_cols = [label for label in column_labels.values() if label in df.columns]
st.dataframe(df[preview_cols].head(30))

# --------------------
# 1. 나이 관련 그래프
# --------------------

if "나이 구간(10살 단위)_str" in df.columns:
    st.subheader("나이 구간(10살 단위) 분포")
    fig_age_bin = px.histogram(
        df,
        x="나이 구간(10살 단위)_str",
        category_orders={"나이 구간(10살 단위)_str": age_bins_str},
    )
    st.plotly_chart(fig_age_bin, use_container_width=True)
# 성별: 원그래프
if "성별" in df.columns:
    st.subheader("성별 분포 (원그래프)")
    sex_counts = df["성별"].value_counts(dropna=True).reset_index()
    sex_counts.columns = ["성별", "count"]
    fig_sex_pie = px.pie(
        sex_counts,
        names="성별",
        values="count",
        hole=0.3,  # 도넛 형태 원하면 유지, 아니면 이 줄 삭제
    )
    st.plotly_chart(fig_sex_pie, use_container_width=True)

# --------------------
# 2. 범주형 캐릭터 변수 (성별, 시도, 세대 유형, 기초생활수급, 가구유형)
# --------------------
cat_vars = [
    "시도명",
    "세대 유형",
    "기초생활수급자 여부",
    "가구유형",
]

for label in cat_vars:
    if label in df.columns:
        st.subheader(f"{label} 분포")
        fig_cat = px.histogram(
            df,
            x=label,
        )
        st.plotly_chart(fig_cat, use_container_width=True)

# --------------------
# 3. 가구원수 관련 (전체 / 19세 이상)
# --------------------
for label in ["가구원수 전체", "가구원수 만 19세 이상"]:
    if label in df.columns:
        st.subheader(f"{label} 분포")
        col_val = pd.to_numeric(df[label], errors="coerce").dropna()
        if len(col_val) > 0:
            fig_num = px.histogram(
                col_val,
                x=col_val,
                nbins=10,
            )
            st.plotly_chart(fig_num, use_container_width=True)

# --------------------
# 4. 가구소득 (이상치 제거)
# --------------------
if "가구소득" in df.columns:
    st.subheader("가구소득(0~20000만원, 이상치 제거) 분포")
    soc_col = pd.to_numeric(df["가구소득"], errors="coerce")
    outlier_vals = [90000, 99999, 77777, 88888, 9999]
    soc_clean = soc_col[~soc_col.isin(outlier_vals)]
    soc_clean = soc_clean[(soc_clean >= 0) & (soc_clean <= 20000)]
    if len(soc_clean) > 0:
        fig_soc = px.histogram(
            soc_clean,
            x=soc_clean,
            nbins=40,
        )
        fig_soc.update_xaxes(range=[0, 20000])
        st.plotly_chart(fig_soc, use_container_width=True)

# --------------------
# 5. 하루 평균 수면시간 (주중 / 주말)
# --------------------
for label in ["하루 평균 수면시간(주중)", "하루 평균 수면시간(주말)"]:
    if label in df.columns:
        st.subheader(f"{label} 분포")
        col_val = pd.to_numeric(df[label], errors="coerce").dropna()
        if len(col_val) > 0:
            fig_sleep = px.histogram(
                col_val,
                x=col_val,
                nbins=30,
            )
            st.plotly_chart(fig_sleep, use_container_width=True)

# --------------------
# 6. 수면 소요시간(분): 이상치 제거 + 15분 단위 구간 (0~360분)
# --------------------
if "수면 소요시간(분)" in df.columns:
    st.subheader("수면 소요시간(분) 분포 (0~360분, 15분 단위, 이상치 제거)")

    sleep_dur = pd.to_numeric(df["수면 소요시간(분)"], errors="coerce").dropna()

    if len(sleep_dur) > 0:
        # 이상치: 0분 이하, 360분(6시간) 초과 제거
        sleep_dur_clean = sleep_dur[(sleep_dur > 0) & (sleep_dur <= 360)]

        if len(sleep_dur_clean) > 0:
            bins = list(range(0, 361, 15))  # 0,15,...,360
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
# 7. 시간 변수: 잠자는 시각 / 기상 시각
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
    "위 그래프들은 모두 필터 없이 전체 데이터 기준 분포입니다. "
    "각 캐릭터 변수의 분포를 한눈에 보고, 이후 우울 지표와의 관계 분석에 활용할 수 있습니다."
)
