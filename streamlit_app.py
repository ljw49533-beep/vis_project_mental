import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# ====================================================
# 0. 공통: 데이터 불러오기
# ====================================================
df = pd.read_csv("kchs_clean_ready.csv", encoding="utf-8")

# ====================================================
# 1. 코드 → 라벨 매핑
# ====================================================
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
    "mta_01z1": "주관적 스트레스 수준",
    "mta_02z1": "스트레스로 인한 정신상담 여부",
    "mtb_01z1": "우울감 경험 여부",
    "mtb_02z1": "우울감으로 인한 정신상담 여부",
    "mtd_01z1": "자살생각 경험 여부",
    "mtd_02z1": "자살생각으로 인한 정신상담 여부",
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
    "nue_01z1": {
        1: "우리 식구 모두가 원하는 만큼의 충분한 양과 다양한 종류의 음식을 먹을 수 있었다",
        2: "우리 식구 모두가 충분한 양의 음식을 먹을 수 있었으나 다양한 종류의 음식은 먹지 못했다",
        3: "경제적으로 어려워서 가끔 먹을 것이 부족했다",
        4: "경제적으로 어려워서 자주 먹을 것이 부족했다",
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
for code, label in column_labels.items():
    if code in df.columns and label not in df.columns:
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
    age_interval = pd.cut(df["만 나이"], bins=bin_edges, right=False)

    def interval_to_str(interval):
        if pd.isna(interval):
            return np.nan
        left = int(interval.left)
        right = int(interval.right)
        return f"{left}-{right}"

    df["나이 구간(10살 단위)"] = age_interval.apply(interval_to_str)
    age_bins_str = sorted(
        df["나이 구간(10살 단위)"].dropna().unique(),
        key=lambda s: int(s.split("-")[0]),
    )
else:
    age_bins_str = []

# 시간값 정렬
def time_order_sort(times):
    def time_to_minutes(s):
        if isinstance(s, str) and ":" in s:
            h, m = s.split(":")
            return int(h) * 60 + int(m)
        return float("inf")
    valid_times = [t for t in times if t is not None and pd.notnull(t)]
    return sorted(valid_times, key=time_to_minutes)

# 그룹별 비율/평균
def group_rate(df_in, group_col, target_col):
    temp = df_in[[group_col, target_col]].copy().dropna()
    if temp.empty:
        return pd.DataFrame(columns=[group_col, "표본수", "값"])
    grp = (
        temp.groupby(group_col)[target_col]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "표본수", "mean": "값"})
    )
    if set(np.unique(temp[target_col])) <= {0, 1}:
        grp["값"] = grp["값"] * 100
    return grp

# ====================================================
# 2. 탭 구성
# ====================================================
st.set_page_config(page_title="KCHS 우울 분석", layout="wide")
tab_dist, tab_1d, tab_2d, tab_help = st.tabs(
    ["데이터 분포", "요인별 1차원 비교", "요인별 2차원 비교", "상담 이용률 분석"]
)

# ====================================================
# 탭1: 데이터 분포
# ====================================================
with tab_dist:
    st.title("KCHS | 인구·수면·정신건강 분포")

    st.subheader("데이터 미리보기")
    preview_cols = [label for label in column_labels.values() if label in df.columns]
    st.dataframe(df[preview_cols].head(30))

    if "나이 구간(10살 단위)" in df.columns:
        st.subheader("나이 구간(10살 단위) 분포")
        fig_age_bin = px.histogram(
            df,
            x="나이 구간(10살 단위)",
            category_orders={"나이 구간(10살 단위)": age_bins_str},
        )
        st.plotly_chart(fig_age_bin, use_container_width=True)

    if "성별" in df.columns:
        st.subheader("성별 분포")
        sex_counts = df["성별"].value_counts(dropna=True).reset_index()
        sex_counts.columns = ["성별", "count"]
        fig_sex_pie = px.pie(
            sex_counts,
            names="성별",
            values="count",
            hole=0.3,
        )
        st.plotly_chart(fig_sex_pie, use_container_width=True)

    for label in ["시도명", "세대 유형", "기초생활수급자 여부"]:
        if label in df.columns:
            st.subheader(f"{label} 분포")
            fig_cat = px.histogram(df, x=label)
            st.plotly_chart(fig_cat, use_container_width=True)

    if "식생활 형편" in df.columns:
        st.subheader("최근 1년간 식생활 형편 분포")
        order_food = [
            response_maps["nue_01z1"][1],
            response_maps["nue_01z1"][2],
            response_maps["nue_01z1"][3],
            response_maps["nue_01z1"][4],
        ]
        present = [v for v in order_food if v in df["식생활 형편"].dropna().unique()]
        fig_food = px.histogram(
            df,
            x="식생활 형편",
            category_orders={"식생활 형편": present},
        )
        st.plotly_chart(fig_food, use_container_width=True)

    for label in ["가구원수 전체", "가구원수 만 19세 이상"]:
        if label in df.columns:
            st.subheader(f"{label} 분포")
            col_val = pd.to_numeric(df[label], errors="coerce").dropna()
            if len(col_val) > 0:
                fig_num = px.histogram(col_val, x=col_val, nbins=10)
                st.plotly_chart(fig_num, use_container_width=True)

    if "가구소득" in df.columns:
        st.subheader("가구소득 분포 (0~20000만원)")
        soc_col = pd.to_numeric(df["가구소득"], errors="coerce")
        outlier_vals = [90000, 99999, 77777, 88888, 9999]
        soc_clean = soc_col[~soc_col.isin(outlier_vals)]
        soc_clean = soc_clean[(soc_clean >= 0) & (soc_clean <= 20000)]
        if len(soc_clean) > 0:
            fig_soc = px.histogram(soc_clean, x=soc_clean, nbins=40)
            fig_soc.update_xaxes(range=[0, 20000])
            st.plotly_chart(fig_soc, use_container_width=True)

    for label in ["하루 평균 수면시간(주중)", "하루 평균 수면시간(주말)"]:
        if label in df.columns:
            st.subheader(f"{label} 분포")
            col_val = pd.to_numeric(df[label], errors="coerce").dropna()
            if len(col_val) > 0:
                fig_sleep = px.histogram(col_val, x=col_val, nbins=30)
                st.plotly_chart(fig_sleep, use_container_width=True)

    if "수면 소요시간(분)" in df.columns:
        st.subheader("수면 소요시간(분) 분포")
        sleep_dur = pd.to_numeric(df["수면 소요시간(분)"], errors="coerce").dropna()
        if len(sleep_dur) > 0:
            sleep_dur_clean = sleep_dur[(sleep_dur > 0) & (sleep_dur <= 360)]
            if len(sleep_dur_clean) > 0:
                bins = list(range(0, 361, 15))
                labels = [f"{b}-{b+15}" for b in bins[:-1]]
                sleep_bins = pd.cut(
                    sleep_dur_clean, bins=bins, labels=labels, right=False
                )
                sleep_df = pd.DataFrame(
                    {"수면 소요시간 구간": sleep_bins}
                ).dropna()
                fig_sd = px.histogram(
                    sleep_df,
                    x="수면 소요시간 구간",
                    title="수면 소요시간(분) 분포 (0~360분, 15분 단위 구간)",
                )
                st.plotly_chart(fig_sd, use_container_width=True)

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

# ====================================================
# 탭2: 요인별 1차원 비교
# ====================================================
with tab_1d:
    st.title("KCHS | 단일 요인별 정신건강 격차")

    if len(df) == 0:
        st.warning("데이터가 없습니다.")
    else:
        axis_candidates = []

        if "나이 구간(10살 단위)" in df.columns:
            axis_candidates.append("나이 구간(10살 단위)")

        for label in [
            "성별",
            "시도명",
            "세대 유형",
            "기초생활수급자 여부",
            "식생활 형편",
            "잠자는 시각",
            "기상 시각",
        ]:
            if label in df.columns and label not in axis_candidates:
                axis_candidates.append(label)

        for label in [
            "하루 평균 수면시간(주중)",
            "하루 평균 수면시간(주말)",
            "수면 소요시간(분)",
            "가구소득",
        ]:
            if label in df.columns and label not in axis_candidates:
                axis_candidates.append(label)

        x_label = st.selectbox("비교할 요인 선택", axis_candidates)

        dep_candidates = []
        for label in [
            "우울감 경험 여부",
            "우울감으로 인한 정신상담 여부",
            "자살생각 경험 여부",
            "자살생각으로 인한 정신상담 여부",
            "주관적 스트레스 수준",
            "스트레스로 인한 정신상담 여부",
        ]:
            if label in df.columns:
                dep_candidates.append(label)

        target_label = st.selectbox("정신건강 관련 지표 선택", dep_candidates)

        st.markdown("##### 결과")

        df_tmp = df[[x_label, target_label]].copy().dropna()
        if df_tmp.empty:
            st.warning("선택한 요인과 정신건강 지표 조합에 데이터가 없습니다.")
        else:
            if x_label in ["잠자는 시각", "기상 시각"]:
                group_col = x_label

            elif x_label == "수면 소요시간(분)":
                df_tmp[x_label] = pd.to_numeric(df_tmp[x_label], errors="coerce")
                df_tmp = df_tmp.dropna(subset=[x_label, target_label])
                s = df_tmp[x_label]
                s = s[(s > 0) & (s <= 360)]
                df_tmp = df_tmp.loc[s.index]
                if df_tmp.empty:
                    st.warning("0~360분 구간 내 수면 소요시간 데이터가 없습니다.")
                    st.stop()
                bins = list(range(0, 361, 15))
                labels = [f"{b}-{b+15}" for b in bins[:-1]]
                df_tmp["요인구간"] = pd.cut(s, bins=bins, labels=labels, right=False)
                df_tmp = df_tmp.dropna(subset=["요인구간"])
                group_col = "요인구간"

            elif x_label in ["하루 평균 수면시간(주중)", "하루 평균 수면시간(주말)"]:
                group_col = x_label

            elif x_label == "가구소득":
                df_tmp[x_label] = pd.to_numeric(df_tmp[x_label], errors="coerce")
                df_tmp = df_tmp.dropna(subset=[x_label, target_label])
                s = df_tmp[x_label]
                out_vals = [90000, 99999, 77777, 88888, 9999]
                s = s[~s.isin(out_vals)]
                s = s[(s >= 0) & (s <= 20000)]
                df_tmp = df_tmp.loc[s.index]
                if df_tmp.empty:
                    st.warning("0~20000만원 구간 내 가구소득 데이터가 없습니다.")
                    st.stop()
                bins = list(range(0, 20001, 2000))
                labels = [f"{b}-{b+2000}" for b in bins[:-1]]
                df_tmp["요인구간"] = pd.cut(s, bins=bins, labels=labels, right=False)
                df_tmp = df_tmp.dropna(subset=["요인구간"])
                group_col = "요인구간"

            else:
                x_col = df[x_label]
                if pd.api.types.is_numeric_dtype(x_col):
                    bins = st.slider("요인 구간 수", 4, 20, 8)
                    df_tmp[x_label] = pd.to_numeric(df_tmp[x_label], errors="coerce")
                    df_tmp = df_tmp.dropna(subset=[x_label, target_label])
                    df_tmp["요인구간"] = pd.cut(
                        df_tmp[x_label],
                        bins=bins,
                        include_lowest=True,
                    )
                    group_col = "요인구간"
                else:
                    group_col = x_label

            if df_tmp.empty:
                st.warning("선택한 요인과 정신건강 지표 조합에 데이터가 없습니다.")
            else:
                s = df_tmp[target_label].astype(str)
                unique_vals = set(s.unique())

                if unique_vals <= {"예", "아니오", "nan"}:
                    df_tmp["is_yes"] = (s == "예").astype(float)
                    res = group_rate(df_tmp, group_col, "is_yes")
                    if res.empty:
                        st.warning("그룹별 결과가 없습니다.")
                    else:
                        fig = px.bar(
                            res,
                            x=group_col,
                            y="값",
                            title=f"{group_col}별 {target_label} '예' 비율(%)",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(res)
                else:
                    df_tmp[target_label] = pd.to_numeric(
                        df_tmp[target_label], errors="coerce"
                    )
                    res = group_rate(df_tmp, group_col, target_label)
                    if res.empty:
                        st.warning("그룹별 결과가 없습니다.")
                    else:
                        fig = px.bar(
                            res,
                            x=group_col,
                            y="값",
                            title=f"{group_col}별 {target_label} 평균 코드값",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(res)

# ====================================================
# 탭3: 요인별 2차원 비교
# ====================================================
with tab_2d:
    st.title("KCHS | 복수 요인 교차에 따른 정신건강")

    if len(df) == 0:
        st.warning("데이터가 없습니다.")
    else:
        axis_candidates = []

        if "나이 구간(10살 단위)" in df.columns:
            axis_candidates.append("나이 구간(10살 단위)")

        for label in [
            "성별",
            "시도명",
            "세대 유형",
            "기초생활수급자 여부",
            "식생활 형편",
            "잠자는 시각",
            "기상 시각",
        ]:
            if label in df.columns and label not in axis_candidates:
                axis_candidates.append(label)

        for label in [
            "하루 평균 수면시간(주중)",
            "하루 평균 수면시간(주말)",
            "수면 소요시간(분)",
            "가구소득",
        ]:
            if label in df.columns and label not in axis_candidates:
                axis_candidates.append(label)

        if len(axis_candidates) < 2:
            st.warning("2차원 비교에 사용할 요인이 2개 이상 필요합니다.")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                axis1 = st.selectbox("요인 1 (행, Row)", axis_candidates, index=0, key="axis1_2d")
            with col_b:
                axis2 = st.selectbox("요인 2 (열, Column)", axis_candidates, index=1, key="axis2_2d")

            if axis1 == axis2:
                st.warning("요인 1과 요인 2는 서로 다른 변수를 선택해야 합니다.")
            else:
                dep_candidates = []
                for label in [
                    "우울감 경험 여부",
                    "우울감으로 인한 정신상담 여부",
                    "자살생각 경험 여부",
                    "자살생각으로 인한 정신상담 여부",
                    "주관적 스트레스 수준",
                    "스트레스로 인한 정신상담 여부",
                ]:
                    if label in df.columns:
                        dep_candidates.append(label)

                target_label = st.selectbox("정신건강 관련 지표 선택 (2D)", dep_candidates)

                df_tmp = df[[axis1, axis2, target_label]].copy().dropna()
                if df_tmp.empty:
                    st.warning("선택한 요인 1·2와 정신건강 지표 조합에 데이터가 없습니다.")
                else:
                    def make_binned_col(df_in, col):
                        series = df_in[col]
                        if col in ["잠자는 시각", "기상 시각"]:
                            return col, series

                        if col == "수면 소요시간(분)":
                            s = pd.to_numeric(series, errors="coerce")
                            s = s[(s > 0) & (s <= 360)]
                            df_sub = df_in.loc[s.index].copy()
                            bins = list(range(0, 361, 30))
                            labels = [f"{b}-{b+30}" for b in bins[:-1]]
                            df_sub["요인"] = pd.cut(s, bins=bins, labels=labels, right=False)
                            return "요인", df_sub["요인"]

                        if col in ["하루 평균 수면시간(주중)", "하루 평균 수면시간(주말)"]:
                            return col, series

                        if col == "가구소득":
                            s = pd.to_numeric(series, errors="coerce")
                            out_vals = [90000, 99999, 77777, 88888, 9999]
                            s = s[~s.isin(out_vals)]
                            s = s[(s >= 0) & (s <= 20000)]
                            df_sub = df_in.loc[s.index].copy()
                            bins = list(range(0, 20001, 2000))
                            labels = [f"{b}-{b+2000}" for b in bins[:-1]]
                            df_sub["요인"] = pd.cut(s, bins=bins, labels=labels, right=False)
                            return "요인", df_sub["요인"]

                        if pd.api.types.is_numeric_dtype(series):
                            s = pd.to_numeric(series, errors="coerce")
                            df_sub = df_in.copy()
                            df_sub["요인"] = pd.cut(s, bins=6, include_lowest=True).astype(str)
                            return "요인", df_sub["요인"]

                        return col, series

                    # 요인 1
                    new_col1, new_series1 = make_binned_col(df_tmp, axis1)
                    df_tmp = df_tmp.loc[new_series1.dropna().index]
                    df_tmp[new_col1] = new_series1.dropna()

                    # 요인 2
                    new_col2, new_series2 = make_binned_col(df_tmp, axis2)
                    df_tmp = df_tmp.loc[new_series2.dropna().index]
                    df_tmp[new_col2] = new_series2.dropna()

                    if df_tmp.empty:
                        st.warning("요인 1·2를 구간화한 뒤 남은 데이터가 없습니다.")
                    else:
                        s = df_tmp[target_label].astype(str)
                        unique_vals = set(s.unique())

                        if unique_vals <= {"예", "아니오", "nan"}:
                            df_tmp["is_yes"] = (s == "예").astype(float)
                            value_col = "is_yes"
                            is_binary = True
                        else:
                            df_tmp[target_label] = pd.to_numeric(
                                df_tmp[target_label], errors="coerce"
                            )
                            value_col = target_label
                            is_binary = False

                        tmp = df_tmp[[new_col1, new_col2, value_col]].dropna()
                        if tmp.empty:
                            st.warning("선택한 요인 조합에 대해 계산 가능한 데이터가 없습니다.")
                        else:
                            pivot = (
                                tmp.groupby([new_col1, new_col2])[value_col]
                                .mean()
                                .reset_index()
                            )
                            if is_binary:
                                pivot["값"] = pivot[value_col] * 100
                                z_label = "비율(%)"
                            else:
                                pivot["값"] = pivot[value_col]
                                z_label = "평균 코드값"

                            heat = pivot.pivot(index=new_col1, columns=new_col2, values="값")
                            st.markdown("#### 교차표 (행: 요인 1, 열: 요인 2)")
                            st.dataframe(heat)

                            fig = px.imshow(
                                heat,
                                text_auto=".1f",
                                aspect="auto",
                                color_continuous_scale="Reds",
                                labels=dict(color=z_label),
                                title=f"{axis1} × {axis2} 에 따른 {target_label} ({z_label})",
                            )
                            st.plotly_chart(fig, use_container_width=True)

# ====================================================
# 탭4: 상담 이용률 (우울/자살/스트레스 기준)
# ====================================================
with tab_help:
    st.title("KCHS | 정신건강 서비스 접근성과 자가진단")

    if len(df) == 0:
        st.warning("데이터가 없습니다.")
    else:
        mode = st.radio(
            "기준 집단 선택",
            ["우울감 경험자 기준", "자살생각 경험자 기준", "스트레스 고수준 기준"],
            horizontal=True,
        )

        if mode == "우울감 경험자 기준":
            base_col = "우울감 경험 여부"
            help_col = "우울감으로 인한 정신상담 여부"
        elif mode == "자살생각 경험자 기준":
            base_col = "자살생각 경험 여부"
            help_col = "자살생각으로 인한 정신상담 여부"
        else:  # 스트레스 고수준
            base_col = "주관적 스트레스 수준"
            help_col = "스트레스로 인한 정신상담 여부"

        if base_col not in df.columns or help_col not in df.columns:
            st.warning("해당 변수들이 데이터에 없습니다.")
        else:
            if mode in ["우울감 경험자 기준", "자살생각 경험자 기준"]:
                base_df = df[df[base_col] == "예"].copy()
            else:
                high_stress_vals = ["대단히 많이 느낀다", "많이 느끼는 편이다"]
                base_df = df[df[base_col].isin(high_stress_vals)].copy()

            st.caption(f"기준 집단 표본 수: {len(base_df):,}명")

            if len(base_df) == 0:
                st.warning("기준 집단에 포함되는 응답자가 없습니다.")
            else:
                axis_candidates = []

                if "나이 구간(10살 단위)" in base_df.columns:
                    axis_candidates.append("나이 구간(10살 단위)")

                for label in [
                    "성별",
                    "시도명",
                    "세대 유형",
                    "기초생활수급자 여부",
                    "식생활 형편",
                    "잠자는 시각",
                    "기상 시각",
                ]:
                    if label in base_df.columns and label not in axis_candidates:
                        axis_candidates.append(label)

                for label in [
                    "하루 평균 수면시간(주중)",
                    "하루 평균 수면시간(주말)",
                    "수면 소요시간(분)",
                    "가구소득",
                ]:
                    if label in base_df.columns and label not in axis_candidates:
                        axis_candidates.append(label)

                axis = st.selectbox(
                    "상담 이용률을 비교할 요인 선택", axis_candidates
                )

                tmp = base_df[[axis, help_col]].copy().dropna()
                if tmp.empty:
                    st.warning("선택한 요인과 상담 변수 조합에 데이터가 없습니다.")
                else:
                    if axis in ["잠자는 시각", "기상 시각"]:
                        group_col = axis

                    elif axis == "수면 소요시간(분)":
                        tmp[axis] = pd.to_numeric(tmp[axis], errors="coerce")
                        tmp = tmp.dropna(subset=[axis, help_col])
                        s = tmp[axis]
                        s = s[(s > 0) & (s <= 360)]
                        tmp = tmp.loc[s.index]
                        bins = list(range(0, 361, 30))
                        labels = [f"{b}-{b+30}" for b in bins[:-1]]
                        tmp["요인구간"] = pd.cut(s, bins=bins, labels=labels, right=False)
                        tmp = tmp.dropna(subset=["요인구간"])
                        group_col = "요인구간"

                    elif axis in ["하루 평균 수면시간(주중)", "하루 평균 수면시간(주말)"]:
                        group_col = axis

                    elif axis == "가구소득":
                        tmp[axis] = pd.to_numeric(tmp[axis], errors="coerce")
                        tmp = tmp.dropna(subset=[axis, help_col])
                        s = tmp[axis]
                        out_vals = [90000, 99999, 77777, 88888, 9999]
                        s = s[~s.isin(out_vals)]
                        s = s[(s >= 0) & (s <= 20000)]
                        tmp = tmp.loc[s.index]
                        bins = list(range(0, 20001, 2000))
                        labels = [f"{b}-{b+2000}" for b in bins[:-1]]
                        tmp["요인구간"] = pd.cut(s, bins=bins, labels=labels, right=False)
                        tmp = tmp.dropna(subset=["요인구간"])
                        group_col = "요인구간"

                    else:
                        x_col = base_df[axis]
                        if pd.api.types.is_numeric_dtype(x_col):
                            tmp[axis] = pd.to_numeric(tmp[axis], errors="coerce")
                            tmp = tmp.dropna(subset=[axis, help_col])
                            tmp["요인구간"] = pd.cut(
                                tmp[axis], bins=6, include_lowest=True
                            ).astype(str)
                            group_col = "요인구간"
                        else:
                            group_col = axis

                    if tmp.empty:
                        st.warning("구간화 후 남은 데이터가 없습니다.")
                    else:
                        s_help = tmp[help_col].astype(str)
                        tmp["is_yes"] = (s_help == "예").astype(float)
                        res = group_rate(tmp, group_col, "is_yes")
                        if res.empty:
                            st.warning("그룹별 상담 이용률을 계산할 수 없습니다.")
                        else:
                            title = f"{axis}별 {mode} 집단 중 {help_col} '예' 비율(%)"
                            fig = px.bar(
                                res,
                                x=group_col,
                                y="값",
                                title=title,
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.dataframe(res)
