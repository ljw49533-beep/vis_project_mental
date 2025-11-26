import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="KCHS 우울 분석 (최대 2차원 비교)", layout="wide")
st.title("KCHS 우울 분석 대시보드 (최대 2차원 비교)")

# =====================================================
# 1. 데이터 불러오기
# =====================================================
df = pd.read_csv("kchs_clean_ready.csv", encoding="utf-8")

# =====================================================
# 2. 공통 전처리 (한 번만)
# =====================================================
character_cols = {
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

depression_cols = {
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

# 한글 라벨 컬럼 생성
for code, label in {**character_cols, **depression_cols}.items():
    if code in df.columns and label not in df.columns:
        if code in response_maps:
            df[label] = df[code].map(response_maps[code])
        else:
            df[label] = df[code]

# 만 나이/나이 구간
if "age" in df.columns:
    if "만 나이" not in df.columns:
        df["만 나이"] = df["age"]
    age_min = int(np.nanmin(df["만 나이"]))
    age_max = int(np.nanmax(df["만 나이"]))
    bin_edges = list(range(age_min // 10 * 10, age_max + 10, 10))
    df["나이 구간(10살 단위)"] = pd.cut(df["만 나이"], bins=bin_edges, right=False)
    df["나이 구간(10살 단위)_str"] = df["나이 구간(10살 단위)"].astype(str)
else:
    age_min = age_max = None

# 우울/자살 이진 변수
def yes_no_to_binary_raw(series):
    s = pd.to_numeric(series, errors="coerce")
    return np.where(s == 1, 1, np.where(s == 2, 0, np.nan))

if "mtb_01z1" in df.columns:
    df["우울_binary"] = yes_no_to_binary_raw(df["mtb_01z1"])
if "mtd_01z1" in df.columns:
    df["자살생각_binary"] = yes_no_to_binary_raw(df["mtd_01z1"])

char_labels = [label for code, label in character_cols.items() if label in df.columns]

# 시간 정렬 함수 (구조 탭에서 사용)
def time_order_sort(times):
    def time_to_minutes(s):
        if isinstance(s, str) and ":" in s:
            h, m = s.split(":")
            return int(h) * 60 + int(m)
        return float("inf")
    valid_times = [t for t in times if t is not None and pd.notnull(t)]
    return sorted(valid_times, key=time_to_minutes)

# 그룹 요약 함수
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

# =====================================================
# 3. 탭 구성
# =====================================================
tab_struct, tab1, tab2, tab_raw = st.tabs(
    ["데이터 구조 보기", "1차원 비교", "2차원 비교", "원자료"]
)

# -----------------------------------------------------
# 탭0: 데이터 구조 보기 (네가 준 구조 코드 기반, df_struct만 사용)
# -----------------------------------------------------
with tab_struct:
    st.subheader("데이터 구조 보기 (필터 + 분포)")

    df_struct = df.copy()
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

    # 라벨 다시 한 번 보장
    display_cols = []
    for code, label in column_labels.items():
        if code in df_struct.columns:
            if code in response_maps:
                df_struct[label] = df_struct[code].map(response_maps[code])
            else:
                df_struct[label] = df_struct[code]
            display_cols.append(label)

    # 나이 구간
    if "만 나이" in df_struct.columns:
        age_min_s = int(np.nanmin(df_struct["만 나이"]))
        age_max_s = int(np.nanmax(df_struct["만 나이"]))
        bin_edges_s = list(range(age_min_s // 10 * 10, age_max_s + 10, 10))
        df_struct["나이 구간(10살 단위)"] = pd.cut(
            df_struct["만 나이"], bins=bin_edges_s, right=False
        )
        df_struct["나이 구간(10살 단위)_str"] = df_struct[
            "나이 구간(10살 단위)"
        ].astype(str)
        age_bins_str = [
            str(interval)
            for interval in sorted(
                df_struct["나이 구간(10살 단위)"].dropna().unique()
            )
        ]
        age_selected = st.sidebar.multiselect(
            "나이 구간(10살 단위)", age_bins_str, default=age_bins_str
        )
    else:
        age_selected = []

    if "만 나이" in display_cols:
        display_cols.remove("만 나이")

    st.sidebar.header("전체 설문문항 필터")
    filters = {}
    for label in display_cols:
        if label not in df_struct.columns:
            continue
        options = [v for v in df_struct[label].dropna().unique()]
        if label in ["잠자는 시각", "기상 시각"]:
            sorted_options = time_order_sort(options)
            filters[label] = st.sidebar.multiselect(
                label, sorted_options, default=sorted_options
            )
        elif (
            "분" in label
            and pd.api.types.is_numeric_dtype(df_struct[label])
            and len(options) > 10
        ):
            min_val, max_val = float(min(options)), float(max(options))
            filters[label] = st.sidebar.slider(
                label, min_val, max_val, (min_val, max_val)
            )
        else:
            filters[label] = st.sidebar.multiselect(
                label, sorted(options), default=sorted(options)
            )

    base_cols = display_cols + ["나이 구간(10살 단위)", "나이 구간(10살 단위)_str"]
    base_cols = [c for c in base_cols if c in df_struct.columns]
    filtered_struct = df_struct[base_cols].copy()

    if age_selected and "나이 구간(10살 단위)_str" in filtered_struct.columns:
        filtered_struct = filtered_struct[
            filtered_struct["나이 구간(10살 단위)_str"].isin(age_selected)
        ]

    for label, sel in filters.items():
        if label not in filtered_struct.columns:
            continue
        col = filtered_struct[label]
        if isinstance(sel, tuple) and pd.api.types.is_numeric_dtype(col):
            filtered_struct = filtered_struct[
                (col >= sel[0]) & (col <= sel[1])
            ]
        elif isinstance(sel, list) and len(sel) < len(
            df_struct[label].dropna().unique()
        ):
            filtered_struct = filtered_struct[col.isin(sel)]

    st.metric("필터 적용 후 응답자 수", filtered_struct.shape[0])
    st.dataframe(filtered_struct.head(30))

    if "나이 구간(10살 단위)_str" in filtered_struct.columns:
        fig_age = px.histogram(
            filtered_struct,
            x="나이 구간(10살 단위)_str",
            title="10살 단위 나이별 분포",
            category_orders={"나이 구간(10살 단위)_str": age_bins_str},
        )
        st.plotly_chart(fig_age, use_container_width=True)

    for label in display_cols:
        if label not in filtered_struct.columns:
            continue
        if label == "가구소득":
            soc_col = filtered_struct[label]
            outlier_vals = [90000, 99999, 77777, 88888, 9999, None, np.nan]
            minval, maxval = 0, 20000
            soc_clean = soc_col[~soc_col.isin(outlier_vals)]
            soc_clean = soc_clean[(soc_clean >= minval) & (soc_clean <= maxval)]
            if len(soc_clean) > 0:
                st.subheader("가구소득(0~20000만원, 이상치 제거) 분포")
                fig_soc = px.histogram(
                    soc_clean,
                    x=soc_clean,
                    title="가구소득 응답 분포(정상 구간, 이상치 제거)",
                    nbins=40,
                )
                fig_soc.update_xaxes(range=[minval, maxval])
                st.plotly_chart(fig_soc, use_container_width=True)
        elif label in ["잠자는 시각", "기상 시각"]:
            times_sorted = time_order_sort(
                filtered_struct[label].dropna().unique()
            )
            if len(times_sorted) > 0:
                fig = px.histogram(
                    filtered_struct,
                    x=label,
                    category_orders={label: times_sorted},
                    title=f"{label} (시간순 정렬)",
                )
                st.plotly_chart(fig, use_container_width=True)
        elif (
            filtered_struct[label].notna().sum() > 0
            and pd.api.types.is_numeric_dtype(filtered_struct[label])
        ):
            fig = px.histogram(
                filtered_struct,
                x=label,
                title=f"{label} 응답 분포",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.info(
        "만 나이는 10살 단위 구간만 사용해 필터/그래프를 제공하고, "
        "개별 만 나이(1살 단위)는 필터와 그래프에서 모두 제거했습니다."
    )

# -----------------------------------------------------
# 탭1: 1차원 비교
# -----------------------------------------------------
with tab1:
    st.subheader("1차원 비교 (축 1개)")
    if len(df) == 0:
        st.warning("데이터가 없습니다.")
    else:
        axis_candidates = []
        if "나이 구간(10살 단위)_str" in df.columns:
            axis_candidates.append("나이 구간(10살 단위)_str")
        for label in char_labels:
            if label in df.columns and label not in axis_candidates:
                axis_candidates.append(label)

        x_label = st.selectbox("비교할 축(캐릭터 변수)", axis_candidates)

        dep_candidates = list(depression_cols.values()) + ["우울_binary", "자살생각_binary"]
        dep_candidates = [c for c in dep_candidates if c in df.columns]
        target_label = st.selectbox("우울 지표 선택", dep_candidates)

        if x_label and target_label:
            if target_label in ["우울_binary", "자살생각_binary"]:
                grp = group_rate(df, x_label, target_label)
                if grp.empty:
                    st.warning("선택한 축과 우울 지표 조합에 데이터가 없습니다.")
                else:
                    y_title = "우울감 경험률(%)" if target_label == "우울_binary" else "자살생각 경험률(%)"
                    fig = px.bar(
                        grp,
                        x=x_label,
                        y="값",
                        title=f"{x_label}별 {y_title}",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(grp)
            else:
                tmp = df[[x_label, target_label]].copy().dropna()
                tmp[target_label] = tmp[target_label].astype(str)
                tmp["is_yes"] = (tmp[target_label] == "예").astype(float)
                grp = group_rate(tmp, x_label, "is_yes")
                if grp.empty:
                    st.warning("그룹별 결과가 없습니다.")
                else:
                    fig = px.bar(
                        grp,
                        x=x_label,
                        y="값",
                        title=f"{x_label}별 {target_label} '예' 비율(%)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(grp)

# -----------------------------------------------------
# 탭2: 2차원 비교
# -----------------------------------------------------
with tab2:
    st.subheader("2차원 비교 (축 2개)")
    if len(df) == 0:
        st.warning("데이터가 없습니다.")
    else:
        axis_candidates = []
        if "나이 구간(10살 단위)_str" in df.columns:
            axis_candidates.append("나이 구간(10살 단위)_str")
        for label in char_labels:
            if (
                label in df.columns
                and not pd.api.types.is_numeric_dtype(df[label])
                and label not in axis_candidates
            ):
                axis_candidates.append(label)

        if len(axis_candidates) < 2:
            st.warning("2차원 비교에 사용할 범주형 캐릭터 변수가 2개 이상 필요합니다.")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                axis1 = st.selectbox("축 1 (행, Row)", axis_candidates, index=0)
            with col_b:
                axis2 = st.selectbox("축 2 (열, Column)", axis_candidates, index=1)

            dep_candidates = list(depression_cols.values()) + ["우울_binary", "자살생각_binary"]
            dep_candidates = [c for c in dep_candidates if c in df.columns]
            target_label = st.selectbox("우울 지표 선택", dep_candidates, key="dep2d")

            if axis1 == axis2:
                st.warning("축 1과 축 2는 서로 다른 변수를 선택해야 합니다.")
            else:
                tmp = df[[axis1, axis2]].copy()
                if target_label in ["우울_binary", "자살생각_binary"]:
                    tmp[target_label] = df[target_label]
                    value_col = target_label
                else:
                    tmp[target_label] = df[target_label].astype(str)
                    tmp["is_yes"] = (tmp[target_label] == "예").astype(float)
                    value_col = "is_yes"

                tmp = tmp[[axis1, axis2, value_col]].dropna()
                if tmp.empty:
                    st.warning("선택한 변수 조합에 데이터가 없습니다.")
                else:
                    pivot = (
                        tmp.groupby([axis1, axis2])[value_col]
                        .mean()
                        .reset_index()
                    )
                    pivot["값"] = pivot[value_col] * 100
                    heat = pivot.pivot(index=axis1, columns=axis2, values="값")

                    st.markdown("#### 교차표 (행: 축 1, 열: 축 2, 값: 비율 %)")
                    st.dataframe(heat)

                    fig = px.imshow(
                        heat,
                        text_auto=".1f",
                        aspect="auto",
                        color_continuous_scale="Reds",
                        labels=dict(color="비율(%)"),
                        title=f"{axis1} × {axis2} 에 따른 {target_label} 비율(%)",
                    )
                    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------
# 탭3: 원자료
# -----------------------------------------------------
with tab_raw:
    st.subheader("전체 원자료 미리보기 (라벨 컬럼 포함)")
    st.dataframe(df.head(50))

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="전체 데이터 CSV 다운로드",
        data=csv,
        file_name="kchs_all.csv",
        mime="text/csv",
    )
