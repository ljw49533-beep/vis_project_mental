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
    "nue_01z1": "식생활 형편",
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
    "fma_04z1": {
        1: "그렇다",
        2: "지금은 아니지만, 과거에 수급자였던 적이 있다",
        3: "아니다",
        7: "응답거부",
        9: "모름",
    },
    # nue_01z1: 식생활 형편 (1~4)
    "nue_01z1": {
        1: "우리 식구 모두가 원하는 만큼의 충분한 양과 다양한 종류의 음식을 먹을 수 있었다",
        2: "우리 식구 모두가 충분한 양의 음식을 먹을 수 있었으나 다양한 종류의 음식은 먹지 못했다",
        3: "경제적으로 어려워서 가끔 먹을 것이 부족했다",
        4: "경제적으로 어려워서 자주 먹을 것이 부족했다",
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
st.set_page_config(page_title="KCHS 캐릭터+식생활 분포", layout="wide")
st.title("KCHS | 캐릭터 및 식생활 형편 분포 대시보드")

st.markdown(
    "- 인구학·소득·수면 변수와 함께, 식생활 형편(nue_01z1)을 라벨로 변환해 분포를 보여줍니다.\n"
    "- 필터 없이 전체 데이터 기준입니다."
)

# --------------------
# 데이터 미리보기
# --------------------
st.subheader("데이터 미리보기")
preview_cols = [label for label in column_labels.values() if label in df.columns]
st.dataframe(df[preview_cols].head(30))

# --------------------
# 나이: 10살 단위만
# --------------------
if "나이 구간(10살 단위)_str" in df.columns:
    st.subheader("나이 구간(10살 단위) 분포")
    fig_age_bin = px.histogram(
        df,
        x="나이 구간(10살 단위)_str",
        category_orders={"나이 구간(10살 단위)_str": age_bins_str},
    )
    st.plotly_chart(fig_age_bin, use_container_width=True)

# --------------------
# 성별: 원그래프
# --------------------
if "성별" in df.columns:
    st.subheader("성별 분포 (원그래프)")
    sex_counts = df["성별"].value_counts(dropna=True).reset_index()
    sex_counts.columns = ["성별", "count"]
    fig_sex_pie = px.pie(
        sex_counts,
        names="성별",
        values="count",
        hole=0.3,
    )
    st.plotly_chart(fig_sex_pie, use_container_width=True)

# --------------------
# 시도명, 세대 유형, 기초생활수급자 여부
# --------------------
for label in ["시도명", "세대 유형", "기초생활수급자 여부"]:
    if label in df.columns:
        st.subheader(f"{label} 분포")
        fig_cat = px.histogram(df, x=label)
        st.plotly_chart(fig_cat, use_container_width=True)

# --------------------
# 식생활 형편(nue_01z1 → 라벨) 분포
# --------------------
if "식생활 형편" in df.columns:
    st.subheader("최근 1년간 식생활 형편 분포")
    # 응답 순서 유지: 1→2→3→4
    order_food = [
        "우리 식구 모두가 원하는 만큼의 충분한 양과 다양한 종류의 음식을 먹을 수 있었다",
        "우리 식구 모두가 충분한 양의 음식을 먹을 수 있었으나 다양한 종류의 음식은 먹지 못했다",
        "경제적으로 어려워서 가끔 먹을 것이 부족했다",
        "경제적으로 어려워서 자주 먹을 것이 부족했다",
    ]
    present = [v for v in order_food if v in df["식생활 형편"].dropna().unique()]
    fig_food = px.histogram(
        df,
        x="식생활 형편",
        category_orders={"식생활 형편": present},
    )
    st.plotly_chart(fig_food, use_container_width=True)

# --------------------
# 가구원수 (전체 / 19세 이상)
# --------------------
for label in ["가구원수 전체", "가구원수 만 19세 이상"]:
    if label in df.columns:
        st.subheader(f"{label} 분포")
        col_val = pd.to_numeric(df[label], errors="coerce").dropna()
        if len(col_val) > 0:
            fig_num = px.histogram(col_val, x=col_val, nbins=10)
            st.plotly_chart(fig_num, use_container_width=True)

# --------------------
# 가구소득 (이상치 제거)
# --------------------
if "가구소득" in df.columns:
    st.subheader("가구소득(0~20000만원, 이상치 제거) 분포")
    soc_col = pd.to_numeric(df["가구소득"], errors="coerce")
    outlier_vals = [90000, 99999, 77777, 88888, 9999]
    soc_clean = soc_col[~soc_col.isin(outlier_vals)]
    soc_clean = soc_clean[(soc_clean >= 0) & (soc_clean <= 20000)]
    if len(soc_clean) > 0:
        fig_soc = px.histogram(soc_clean, x=soc_clean, nbins=40)
        fig_soc.update_xaxes(range=[0, 20000])
        st.plotly_chart(fig_soc, use_container_width=True)

# --------------------
# 하루 평균 수면시간 (주중 / 주말)
# --------------------
for label in ["하루 평균 수면시간(주중)", "하루 평균 수면시간(주말)"]:
    if label in df.columns:
        st.subheader(f"{label} 분포")
        col_val = pd.to_numeric(df[label], errors="coerce").dropna()
        if len(col_val) > 0:
            fig_sleep = px.histogram(col_val, x=col_val, nbins=30)
            st.plotly_chart(fig_sleep, use_container_width=True)

# --------------------
# 수면 소요시간(분): 0~360분, 15분 구간
# --------------------
if "수면 소요시간(분)" in df.columns:
    st.subheader("수면 소요시간(분) 분포 (0~360분, 15분 단위, 이상치 제거)")
    sleep_dur = pd.to_numeric(df["수면 소요시간(분)"], errors="coerce").dropna()
    if len(sleep_dur) > 0:
        sleep_dur_clean = sleep_dur[(sleep_dur > 0) & (sleep_dur <= 360)]
        if len(sleep_dur_clean) > 0:
            bins = list(range(0, 361, 15))
            labels = [f"{b}-{b+15}" for b in bins[:-1]]
            sleep_bins = pd.cut(sleep_dur_clean, bins=bins, labels=labels, right=False)
            sleep_df = pd.DataFrame(
                {"수면 소요시간(15분 구간)": sleep_bins}
            ).dropna()
            fig_sd = px.histogram(
                sleep_df,
                x="수면 소요시간(15분 구간)",
                title="수면 소요시간(분) 분포 (0~360분, 15분 단위 구간)",
            )
            st.plotly_chart(fig_sd, use_container_width=True)

# --------------------
# 시간 변수: 잠자는 시각 / 기상 시각
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
    "nue_01z1(식생활 형편) 코드는 1~4까지이며, 설문지 문항 내용으로 라벨링하여 분포를 시각화했습니다. "
    "이 페이지를 데이터 구조 및 식생활 관련 특성 파악용으로 사용할 수 있습니다."
)
