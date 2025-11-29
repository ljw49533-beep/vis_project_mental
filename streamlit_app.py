import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import numpy as np

# ====================================================
# 0. 전역 스타일
# ====================================================
pio.templates.default = "plotly_white"
COLOR_CAT = px.colors.qualitative.Set2

# ====================================================
# 1. 데이터 불러오기
# ====================================================
df = pd.read_csv("kchs_clean_ready.csv", encoding="utf-8")

# ====================================================
# 2. 코드 → 라벨 매핑
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

    def interval_to_str(i):
        if pd.isna(i):
            return np.nan
        return f"{int(i.left)}-{int(i.right)}"

    df["나이 구간(10살 단위)"] = age_interval.apply(interval_to_str)
    age_bins_str = sorted(
        df["나이 구간(10살 단위)"].dropna().unique(),
        key=lambda s: int(s.split("-")[0]),
    )
else:
    age_bins_str = []

# 시간 정렬
def time_order_sort(times):
    def to_min(s):
        if isinstance(s, str) and ":" in s:
            h, m = s.split(":")
            return int(h) * 60 + int(m)
        return float("inf")
    valid = [t for t in times if t is not None and pd.notnull(t)]
    return sorted(valid, key=to_min)

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

# 공통 layout: trace 이름/legend/legend title 제거 + y축 제목 기본값
def style_fig(fig, y_title="count"):
    fig.for_each_trace(lambda t: t.update(name=""))  # trace name 비우기[web:46]
    fig.update_layout(
        font=dict(size=13),
        title_font_size=18,
        margin=dict(t=60, l=40, r=20, b=60),
        showlegend=False,
        legend_title_text="",
    )
    fig.update_yaxes(title_text=y_title)
    return fig

# ====================================================
# 3. 탭 구성
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
    st.dataframe(df[preview_cols].head(30), use_container_width=True)

    if "나이 구간(10살 단위)" in df.columns:
        st.subheader("나이 구간(10살 단위) 분포")
        fig_age = px.histogram(
            df,
            x="나이 구간(10살 단위)",
            category_orders={"나이 구간(10살 단위)": age_bins_str},
            labels={"나이 구간(10살 단위)": "나이 구간 (10살 단위)"},
            color_discrete_sequence=COLOR_CAT,
        )
        fig_age = style_fig(fig_age, y_title="인원 수")
        st.plotly_chart(fig_age, use_container_width=True)

    if "성별" in df.columns:
        st.subheader("성별 분포")
        sex_counts = df["성별"].value_counts(dropna=True).reset_index()
        sex_counts.columns = ["성별", "count"]
        fig_sex = px.pie(
            sex_counts,
            names="성별",
            values="count",
            color="성별",
            color_discrete_sequence=COLOR_CAT,
            hole=0.3,
        )
        fig_sex = style_fig(fig_sex, y_title="")
        st.plotly_chart(fig_sex, use_container_width=True)

    for label in ["시도명", "세대 유형", "기초생활수급자 여부"]:
        if label in df.columns:
            st.subheader(f"{label} 분포")
            fig_cat = px.histogram(
                df,
                x=label,
                labels={label: label},
                color_discrete_sequence=COLOR_CAT,
            )
            fig_cat = style_fig(fig_cat, y_title="인원 수")
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
            labels={"식생활 형편": "식생활 형편"},
            color_discrete_sequence=COLOR_CAT,
        )
        fig_food = style_fig(fig_food, y_title="인원 수")
        st.plotly_chart(fig_food, use_container_width=True)

    # --- 가구원수 전체 / 만 19세 이상 ---
    for label in ["가구원수 전체", "가구원수 만 19세 이상"]:
        if label in df.columns:
            st.subheader(f"{label} 분포")
            col_val = pd.to_numeric(df[label], errors="coerce").dropna()
            if len(col_val) > 0:
                tmp = pd.DataFrame({label: col_val})
                fig_num = px.histogram(
                    tmp,
                    x=label,
                    nbins=10,
                    labels={label: f"{label} (명)"},
                    color_discrete_sequence=COLOR_CAT,
                )
                fig_num = style_fig(fig_num, y_title="가구 수")
                st.plotly_chart(fig_num, use_container_width=True)

    # --- 가구소득 ---
    if "가구소득" in df.columns:
        st.subheader("가구소득 분포 (0~20000만원)")
        soc_col = pd.to_numeric(df["가구소득"], errors="coerce")
        outlier_vals = [90000, 99999, 77777, 88888, 9999]
        soc_clean = soc_col[~soc_col.isin(outlier_vals)]
        soc_clean = soc_clean[(soc_clean >= 0) & (soc_clean <= 20000)]
        if len(soc_clean) > 0:
            tmp = pd.DataFrame({"가구소득(만원)": soc_clean})
            fig_soc = px.histogram(
                tmp,
                x="가구소득(만원)",
                nbins=40,
                labels={"가구소득(만원)": "가구소득 (만원)"},
                color_discrete_sequence=COLOR_CAT,
            )
            fig_soc.update_xaxes(range=[0, 20000])
            fig_soc = style_fig(fig_soc, y_title="가구 수")
            st.plotly_chart(fig_soc, use_container_width=True)

    # --- 하루 평균 수면시간(주중/주말) ---
    for label in ["하루 평균 수면시간(주중)", "하루 평균 수면시간(주말)"]:
        if label in df.columns:
            st.subheader(f"{label} 분포")
            col_val = pd.to_numeric(df[label], errors="coerce").dropna()
            if len(col_val) > 0:
                tmp = pd.DataFrame({label: col_val})
                fig_sleep = px.histogram(
                    tmp,
                    x=label,
                    nbins=30,
                    labels={label: f"{label} (시간)"},
                    color_discrete_sequence=COLOR_CAT,
                )
                fig_sleep = style_fig(fig_sleep, y_title="인원 수")
                st.plotly_chart(fig_sleep, use_container_width=True)

    # --- 수면 소요시간(분) ---
    if "수면 소요시간(분)" in df.columns:
        st.subheader("수면 소요시간(분) 분포 (0~360분, 15분 단위)")
        sleep_dur = pd.to_numeric(df["수면 소요시간(분)"], errors="coerce").dropna()
        if len(sleep_dur) > 0:
            sleep_dur_clean = sleep_dur[(sleep_dur > 0) & (sleep_dur <= 360)]
            if len(sleep_dur_clean) > 0:
                bins = list(range(0, 361, 15))
                labels = [f"{b}-{b+15}" for b in bins[:-1]]
                sleep_bins = pd.cut(
                    sleep_dur_clean, bins=bins, labels=labels, right=False
                )
                sleep_df = pd.DataFrame({"수면 소요시간(분 구간)": sleep_bins}).dropna()
                fig_sd = px.histogram(
                    sleep_df,
                    x="수면 소요시간(분 구간)",
                    labels={"수면 소요시간(분 구간)": "수면 소요시간 (분, 15분 단위)"},
                    color_discrete_sequence=COLOR_CAT,
                )
                fig_sd = style_fig(fig_sd, y_title="인원 수")
                st.plotly_chart(fig_sd, use_container_width=True)

    # --- 잠자는 시각 / 기상 시각 ---
    for label in ["잠자는 시각", "기상 시각"]:
        if label in df.columns:
            times_sorted = time_order_sort(df[label].dropna().unique())
            if len(times_sorted) > 0:
                st.subheader(f"{label} 분포 (시간순 정렬)")
                fig_t = px.histogram(
                    df,
                    x=label,
                    category_orders={label: times_sorted},
                    labels={label: f"{label} (시:분)"},
                    color_discrete_sequence=COLOR_CAT,
                )
                fig_t = style_fig(fig_t, y_title="인원 수")
                st.plotly_chart(fig_t, use_container_width=True)
                
st.markdown("### 변수 코드 및 정의 출처 안내")

st.markdown(
    """
지역사회건강조사(KCHS) 2024년 원시자료에 포함된 각 변수의 공식 정의와 코드값은  
질병관리청에서 제공하는 **「지역사회건강조사 2024년 원시자료 이용지침서 및 참고자료」**를 참고해야 합니다.

1. 아래 링크를 통해 질병관리청 지역사회건강조사 누리집에 접속합니다.  
   - https://chs.kdca.go.kr/chs/mnl/mnlBoardMain.do
2. 화면에서 **「원시자료 이용지침서」** 메뉴 중  
   **「지역사회건강조사 2024년 원시자료 이용지침서 및 참고자료」**를 선택하여 PDF를 내려받습니다.
3. PDF의 **변수설명서(예: 50–57쪽, 96–115쪽)**에서  
   이 대시보드에서 사용하는 변수(세대 유형, 기초생활수급자 여부, 가구소득, 식생활 형편,  
   수면시간, 주관적 스트레스 수준, 우울감/자살생각 경험 여부, 상담 이용 여부 등)의  
   정확한 문항 내용과 코드값 의미를 확인할 수 있습니다.

※ 이 대시보드는 공식 지침서를 참고해 변수명을 한글 라벨로 정리해 보여주며,  
   **상세 코드북 내용과 조사 문항 원문은 반드시 질병관리청에서 제공하는 공식 PDF를 기준으로 해석해야 합니다.**
"""
)

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
                bins = list(range(0, 361, 30))
                labels = [f"{b}-{b+30}" for b in bins[:-1]]
                df_tmp["요인구간"] = pd.cut(s, bins=bins, labels=labels, right=False)
                df_tmp = df_tmp.dropna(subset=["요인구간"])
                group_col = "요인구간"

            elif x_label in ["하루 평균 수면시간(주중)", "하루 평균 수면시간(주말)"]:
                group_col = x_label

            elif x_label == "가구소득":
                df_tmp[x_label] = pd.to_numeric(df_tmp[x_label], errors="coerce")
                df_tmp = df_tmp.dropna(subset=[x_label, target_label])
                s = df_tmp[x_label]
                outlier_vals = [90000, 99999, 77777, 88888, 9999]
                s = s[~s.isin(outlier_vals)]
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
                        df_tmp[x_label], bins=bins, include_lowest=True
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
                    y_title = "비율(%)"
                else:
                    df_tmp[target_label] = pd.to_numeric(
                        df_tmp[target_label], errors="coerce"
                    )
                    res = group_rate(df_tmp, group_col, target_label)
                    y_title = "평균 코드값"

                if res.empty:
                    st.warning("그룹별 결과가 없습니다.")
                else:
                    fig = px.bar(
                        res,
                        x=group_col,
                        y="값",
                        color_discrete_sequence=COLOR_CAT,
                    )
                    fig.update_traces(texttemplate="%{y:.1f}", textposition="outside")
                    fig = style_fig(fig, y_title=y_title)
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(res, use_container_width=True)

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
                            outlier_vals = [90000, 99999, 77777, 88888, 9999]
                            s = s[~s.isin(outlier_vals)]
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

                    new_col1, new_series1 = make_binned_col(df_tmp, axis1)
                    df_tmp = df_tmp.loc[new_series1.dropna().index]
                    df_tmp[new_col1] = new_series1.dropna()

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

                        tmp2 = df_tmp[[new_col1, new_col2, value_col]].dropna()
                        if tmp2.empty:
                            st.warning("선택한 요인 조합에 대해 계산 가능한 데이터가 없습니다.")
                        else:
                            pivot = (
                                tmp2.groupby([new_col1, new_col2])[value_col]
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
                            fig_h = px.imshow(
                                heat,
                                text_auto=".1f",
                                aspect="auto",
                                labels=dict(color=z_label),
                                color_continuous_scale="Blues",
                                title=f"{axis1} × {axis2} 에 따른 {target_label} ({z_label})",
                            )
                            fig_h.update_coloraxes(colorbar_title_text="")
                            fig_h = style_fig(fig_h, y_title=z_label)
                            st.plotly_chart(fig_h, use_container_width=True)
                            st.dataframe(heat, use_container_width=True)

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
        else:
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
                        outlier_vals = [90000, 99999, 77777, 88888, 9999]
                        s = s[~s.isin(outlier_vals)]
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
                            fig = px.bar(
                                res,
                                x=group_col,
                                y="값",
                                color_discrete_sequence=COLOR_CAT,
                            )
                            fig.update_traces(texttemplate="%{y:.1f}", textposition="outside")
                            fig = style_fig(fig, y_title="상담 '예' 비율(%)")
                            st.plotly_chart(fig, use_container_width=True)
                            st.dataframe(res, use_container_width=True)
