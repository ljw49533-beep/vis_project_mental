import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --------------------
# 기본 설정
# --------------------
st.set_page_config(
    page_title="KCHS 우울 분석 대시보드",
    layout="wide",
)

st.title("KCHS 우울 관련 요인 대시보드")
st.markdown(
    "- 우울감 경험, 자살생각과 소득·나이·수면 등 사이의 관계를 탐색하는 대시보드입니다.\n"
    "- 좌측 사이드바에서 공통 필터를 적용하고, 상단 탭에서 세부 분석을 선택하세요."
)

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
    "수면 소요시간(분)": "수면 소요시간(분)",
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
        3: "아니다",
        7: "응답거부",
        9: "모름",
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

# 라벨 컬럼 생성
display_cols = []
for code, label in column_labels.items():
    if code in response_maps and code in df.columns:
        df[label] = df[code].map(response_maps[code])
        display_cols.append(label)
    elif code in df.columns:
        df[label] = df[code]
        display_cols.append(label)

# --------------------
# 나이 10살 구간 생성
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

# 만 나이는 이후 필터/그래프용 리스트에서 제외 (요약용으로만 사용)
if "만 나이" in display_cols:
    display_cols.remove("만 나이")

# --------------------
# 공통: 이진 우울/자살 변수 생성
# --------------------
def yes_no_to_binary(series):
    # 코드 1=예, 2=아니오, 그 외/결측은 NaN
    s = pd.to_numeric(series, errors="coerce")
    return np.where(s == 1, 1, np.where(s == 2, 0, np.nan))

if "우울감 경험 여부" in df.columns:
    df["우울_binary"] = yes_no_to_binary(df["우울감 경험 여부"])
else:
    df["우울_binary"] = np.nan

if "자살생각 경험 여부" in df.columns:
    df["자살생각_binary"] = yes_no_to_binary(df["자살생각 경험 여부"])
else:
    df["자살생각_binary"] = np.nan

# --------------------
# 사이드바: 전역 필터
# --------------------
st.sidebar.header("전역 필터")

# 시도
if "시도명" in df.columns:
    region_opts = sorted(df["시도명"].dropna().unique())
    selected_regions = st.sidebar.multiselect(
        "시도명", region_opts, default=region_opts
    )
else:
    selected_regions = None

# 성별
if "성별" in df.columns:
    sex_opts = sorted(df["성별"].dropna().unique())
    selected_sex = st.sidebar.multiselect("성별", sex_opts, default=sex_opts)
else:
    selected_sex = None

# 나이 구간(10살)
if age_bins_str:
    selected_age_bins = st.sidebar.multiselect(
        "나이 구간(10살 단위)", age_bins_str, default=age_bins_str
    )
else:
    selected_age_bins = None

# 가구소득 범위 (이상치 포함 상태에서 범위만 지정)
if "가구소득" in df.columns:
    income_series = pd.to_numeric(df["가구소득"], errors="coerce")
    inc_min = int(np.nanmin(income_series))
    inc_max = int(np.nanmax(income_series))
    selected_income = st.sidebar.slider(
        "가구소득(만원, 전체 범위 필터)",
        min_value=inc_min,
        max_value=inc_max,
        value=(inc_min, inc_max),
    )
else:
    selected_income = None

# --------------------
# 전역 필터 적용
# --------------------
filtered = df.copy()

if selected_regions is not None:
    filtered = filtered[filtered["시도명"].isin(selected_regions)]

if selected_sex is not None:
    filtered = filtered[filtered["성별"].isin(selected_sex)]

if selected_age_bins is not None and "나이 구간(10살 단위)_str" in filtered.columns:
    filtered = filtered[filtered["나이 구간(10살 단위)_str"].isin(selected_age_bins)]

if selected_income is not None and "가구소득" in filtered.columns:
    income_val = pd.to_numeric(filtered["가구소득"], errors="coerce")
    filtered = filtered[
        (income_val >= selected_income[0]) & (income_val <= selected_income[1])
    ]

st.caption(f"전역 필터 적용 후 표본 수: {len(filtered):,}명")

# --------------------
# 유틸 함수: 그룹별 우울률 계산
# --------------------
def depression_rate_by(group_col, df_in, target="우울_binary"):
    temp = df_in.copy()
    temp = temp[[group_col, target]].dropna()
    if temp.empty:
        return pd.DataFrame(columns=[group_col, "표본수", "우울률"])
    grp = (
        temp.groupby(group_col)[target]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "표본수", "mean": "우울률"})
    )
    grp["우울률"] = grp["우울률"] * 100
    return grp

# --------------------
# 탭 구성
# --------------------
(
    tab_overview,
    tab_income,
    tab_age,
    tab_sleep,
    tab_explorer,
    tab_raw,
) = st.tabs(
    [
        "요약",
        "소득 × 우울",
        "나이 × 우울",
        "수면 × 우울",
        "교차탐색",
        "원자료",
    ]
)

# ====================
# 1. 요약 탭
# ====================
with tab_overview:
    st.subheader("전체 우울·자살 지표 요약")

    total_n = len(filtered)
    dep_rate = np.nan
    sui_rate = np.nan
    if total_n > 0:
        dep_rate = np.nanmean(filtered["우울_binary"]) * 100
        sui_rate = np.nanmean(filtered["자살생각_binary"]) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("표본 수", f"{total_n:,}명")
    col2.metric("우울감 경험률", f"{dep_rate:.1f}%" if not np.isnan(dep_rate) else "데이터 없음")
    col3.metric("자살생각 경험률", f"{sui_rate:.1f}%" if not np.isnan(sui_rate) else "데이터 없음")

    st.markdown("---")

    # 나이 구간별 우울률
    if "나이 구간(10살 단위)_str" in filtered.columns:
        age_dep = depression_rate_by("나이 구간(10살 단위)_str", filtered)
        if not age_dep.empty:
            fig_age_dep = px.bar(
                age_dep,
                x="나이 구간(10살 단위)_str",
                y="우울률",
                title="나이 구간(10살 단위)별 우울감 경험률(%)",
            )
            st.plotly_chart(fig_age_dep, use_container_width=True)

    # 성별별 우울률
    if "성별" in filtered.columns:
        sex_dep = depression_rate_by("성별", filtered)
        if not sex_dep.empty:
            fig_sex_dep = px.bar(
                sex_dep,
                x="성별",
                y="우울률",
                title="성별별 우울감 경험률(%)",
            )
            st.plotly_chart(fig_sex_dep, use_container_width=True)

# ====================
# 2. 소득 × 우울 탭
# ====================
with tab_income:
    st.subheader("가구소득과 우울")

    if "가구소득" not in filtered.columns:
        st.warning("가구소득 변수가 없습니다.")
    else:
        # 이상치 제거용 옵션
        st.caption("아래 옵션으로 분석용 소득 구간을 정제합니다.")
        col_a, col_b = st.columns(2)
        with col_a:
            remove_outlier = st.checkbox(
                "이상치 코드(90000, 99999, 77777, 88888, 9999) 제거", value=True
            )
        with col_b:
            n_bins = st.slider("소득 구간 수", min_value=4, max_value=20, value=8)

        inc = pd.to_numeric(filtered["가구소득"], errors="coerce")
        if remove_outlier:
            out_vals = [90000, 99999, 77777, 88888, 9999]
            inc = inc.mask(inc.isin(out_vals), np.nan)

        tmp = filtered.copy()
        tmp["가구소득_clean"] = inc
        tmp = tmp.dropna(subset=["가구소득_clean", "우울_binary"])

        if tmp.empty:
            st.warning("소득·우울 데이터가 충분하지 않습니다.")
        else:
            # 등폭 구간
            tmp["소득구간"] = pd.cut(
                tmp["가구소득_clean"],
                bins=n_bins,
                include_lowest=True,
            )
            inc_dep = depression_rate_by("소득구간", tmp)
            if not inc_dep.empty:
                inc_dep = inc_dep.sort_values("소득구간")
                fig_inc = px.line(
                    inc_dep,
                    x="소득구간",
                    y="우울률",
                    markers=True,
                    title="가구소득 구간별 우울감 경험률(%)",
                )
                st.plotly_chart(fig_inc, use_container_width=True)
                st.dataframe(inc_dep)

# ====================
# 3. 나이 × 우울 탭
# ====================
with tab_age:
    st.subheader("나이 구간과 우울")

    if "나이 구간(10살 단위)_str" not in filtered.columns:
        st.warning("나이 구간 정보가 없습니다.")
    else:
        # 비교할 나이 구간 선택
        age_opts = sorted(
            filtered["나이 구간(10살 단위)_str"].dropna().unique()
        )
        selected_compare = st.multiselect(
            "비교할 나이 구간 선택 (예: 20대와 30대)",
            age_opts,
            default=age_opts,
        )

        tmp = filtered.copy()
        tmp = tmp[tmp["나이 구간(10살 단위)_str"].isin(selected_compare)]
        age_dep = depression_rate_by("나이 구간(10살 단위)_str", tmp)

        if age_dep.empty:
            st.warning("선택한 나이 구간에 데이터가 없습니다.")
        else:
            age_dep = age_dep.sort_values("나이 구간(10살 단위)_str")
            fig = px.bar(
                age_dep,
                x="나이 구간(10살 단위)_str",
                y="우울률",
                title="나이 구간별 우울감 경험률(%)",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(age_dep)

# ====================
# 4. 수면 × 우울 탭
# ====================
with tab_sleep:
    st.subheader("수면과 우울")

    sleep_col = st.selectbox(
        "수면 관련 변수 선택",
        [c for c in ["하루 평균 수면시간(주중)", "하루 평균 수면시간(주말)", "수면 소요시간(분)"] if c in filtered.columns],
    )

    if sleep_col is None:
        st.warning("수면 관련 변수가 없습니다.")
    else:
        tmp = filtered[[sleep_col, "우울_binary"]].copy()
        tmp[sleep_col] = pd.to_numeric(tmp[sleep_col], errors="coerce")
        tmp = tmp.dropna()

        if tmp.empty:
            st.warning("수면·우울 데이터가 충분하지 않습니다.")
        else:
            col_l, col_r = st.columns(2)
            with col_l:
                st.caption("산점도 + 추세선(형태만 보는 용도)")
                fig_scatter = px.scatter(
                    tmp,
                    x=sleep_col,
                    y="우울_binary",
                    trendline="ols",
                    opacity=0.2,
                    title=f"{sleep_col}과 우울(0/1) 관계",
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            with col_r:
                st.caption("수면시간 구간별 우울률")
                bins = st.slider("수면시간 구간 수", 4, 20, 8)
                tmp["수면구간"] = pd.cut(tmp[sleep_col], bins=bins, include_lowest=True)
                sleep_dep = depression_rate_by("수면구간", tmp)
                if not sleep_dep.empty:
                    fig_sleep = px.bar(
                        sleep_dep,
                        x="수면구간",
                        y="우울률",
                        title=f"{sleep_col} 구간별 우울감 경험률(%)",
                    )
                    st.plotly_chart(fig_sleep, use_container_width=True)
                    st.dataframe(sleep_dep)

# ====================
# 5. 교차탐색 탭
# ====================
with tab_explorer:
    st.subheader("사용자 지정 교차탐색")

    # 타깃(우울/자살 관련) 선택
    target_map = {
        "우울감 경험 여부": "우울_binary",
        "자살생각 경험 여부": "자살생각_binary",
    }
    target_label = st.selectbox("우울 관련 지표 선택", list(target_map.keys()))
    target_col = target_map[target_label]

    # 설명변수 선택
    candidate_explanatory = [
        c for c in ["가구소득", "나이 구간(10살 단위)_str", "성별", "가구유형", "시도명"]
        if c in filtered.columns
    ]
    expl = st.selectbox("설명변수 선택", candidate_explanatory)

    if expl is None:
        st.warning("설명변수를 선택하세요.")
    else:
        tmp = filtered[[expl, target_col]].copy()
        tmp = tmp.dropna()

        if tmp.empty:
            st.warning("선택한 변수 조합에 데이터가 없습니다.")
        else:
            if pd.api.types.is_numeric_dtype(tmp[expl]) and expl != "나이 구간(10살 단위)_str":
                # 수치형이면 구간으로 나눠서 그룹 분석
                bins = st.slider("설명변수 구간 수", 4, 20, 8)
                tmp["설명구간"] = pd.cut(
                    pd.to_numeric(tmp[expl], errors="coerce"),
                    bins=bins,
                    include_lowest=True,
                )
                grp_col = "설명구간"
            else:
                grp_col = expl

            res = depression_rate_by(grp_col, tmp, target=target_col)
            if res.empty:
                st.warning("그룹별 결과가 없습니다.")
            else:
                res = res.sort_values(grp_col)
                fig = px.bar(
                    res,
                    x=grp_col,
                    y="우울률",
                    title=f"{expl}에 따른 {target_label} 비율(%)",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(res)

# ====================
# 6. 원자료 탭
# ====================
with tab_raw:
    st.subheader("전역 필터 적용 후 원자료 미리보기")
    st.dataframe(filtered.head(50))

    csv = filtered.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="현재 필터 적용 데이터 CSV 다운로드",
        data=csv,
        file_name="kchs_filtered.csv",
        mime="text/csv",
    )
