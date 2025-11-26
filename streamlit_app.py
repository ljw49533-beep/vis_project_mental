import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ====================
# 기본 설정
# ====================
st.set_page_config(page_title="KCHS 우울 분석 (최대 2차원 비교)", layout="wide")
st.title("KCHS 우울 분석 대시보드 (최대 2차원 비교)")

# ====================
# 데이터 불러오기
# ====================
df = pd.read_csv("kchs_clean_ready.csv", encoding="utf-8")

# ====================
# 코드 → 라벨 매핑
# ====================
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

# 한글 라벨 컬럼 만들기
label_map = {}  # 한글라벨 → 원래 코드
for code, label in {**character_cols, **depression_cols}.items():
    if code in df.columns:
        if code in response_maps:
            df[label] = df[code].map(response_maps[code])
        else:
            df[label] = df[code]
        label_map[label] = code

# 만 나이를 10살 구간으로 추가
if "age" in df.columns:
    df["만 나이"] = df["age"]
    age_min = int(np.nanmin(df["만 나이"]))
    age_max = int(np.nanmax(df["만 나이"]))
    bin_edges = list(range(age_min // 10 * 10, age_max + 10, 10))
    df["나이 구간(10살 단위)"] = pd.cut(df["만 나이"], bins=bin_edges, right=False)
    df["나이 구간(10살 단위)_str"] = df["나이 구간(10살 단위)"].astype(str)

# 우울/자살 이진 변수(원시 코드 기준)
def yes_no_to_binary_raw(series):
    s = pd.to_numeric(series, errors="coerce")
    return np.where(s == 1, 1, np.where(s == 2, 0, np.nan))

if "mtb_01z1" in df.columns:
    df["우울_binary"] = yes_no_to_binary_raw(df["mtb_01z1"])
else:
    df["우울_binary"] = np.nan

if "mtd_01z1" in df.columns:
    df["자살생각_binary"] = yes_no_to_binary_raw(df["mtd_01z1"])
else:
    df["자살생각_binary"] = np.nan

# 캐릭터 라벨 목록
char_labels = [label for code, label in character_cols.items() if label in df.columns]

# ====================
# 유틸: 그룹별 값/비율
# ====================
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

# ====================
# 탭 구성: 구조 / 1D / 2D / 원자료
# ====================
tab_struct, tab1, tab2, tab_raw = st.tabs(
    ["데이터 구조 보기", "1차원 비교", "2차원 비교", "원자료"]
)

# --------------------
# 탭0: 데이터 구조 보기
# (이전에 쓰던 전체 분포 시각화 느낌으로 최소만 구현)
# --------------------
with tab_struct:
    st.subheader("전체 데이터 구조 보기")

    # 만 나이 분포 (1살 단위, 구조 파악용)
    if "만 나이" in df.columns:
        fig_age = px.histogram(
            df,
            x="만 나이",
            nbins=50,
            title="만 나이 응답 분포 (구조 파악용)",
        )
        st.plotly_chart(fig_age, use_container_width=True)

    # 가구소득 분포 (이상치 포함, 구조 파악용)
    if "가구소득" in df.columns:
        fig_inc = px.histogram(
            df,
            x="가구소득",
            nbins=60,
            title="가구소득 응답 분포 (구조 파악용)",
        )
        st.plotly_chart(fig_inc, use_container_width=True)

    # 수면시간 분포
    for c in ["하루 평균 수면시간(주중)", "하루 평균 수면시간(주말)", "수면 소요시간(분)"]:
        if c in df.columns:
            fig_sleep = px.histogram(
                df, x=c, nbins=40, title=f"{c} 분포 (구조 파악용)"
            )
            st.plotly_chart(fig_sleep, use_container_width=True)

    st.markdown("위 그래프들은 **필터 없이 전체 데이터 구조**를 보는 용도입니다.")

# --------------------
# 탭1: 1차원 비교 (축 1개만)
# --------------------
with tab1:
    st.subheader("1차원 비교 (축 1개)")

    if len(df) == 0:
        st.warning("데이터가 없습니다.")
    else:
        # 축 후보: 나이 구간 + 캐릭터 라벨
        axis_candidates = []
        if "나이 구간(10살 단위)_str" in df.columns:
            axis_candidates.append("나이 구간(10살 단위)_str")
        for label in char_labels:
            if label in df.columns and label not in axis_candidates:
                axis_candidates.append(label)

        x_label = st.selectbox("비교할 축(캐릭터 변수)", axis_candidates)

        # 우울 지표 후보
        dep_candidates = list(depression_cols.values()) + ["우울_binary", "자살생각_binary"]
        dep_candidates = [c for c in dep_candidates if c in df.columns]
        target_label = st.selectbox("우울 지표 선택", dep_candidates)

        # 1차원 비교에서는 **추가 필터 없음** → 선택한 축만 사용

        if x_label and target_label:
            if target_label in ["우울_binary", "자살생각_binary"]:
                grp = group_rate(df, x_label, target_label)
                if grp.empty:
                    st.warning("선택한 축과 우울 지표 조합에 데이터가 없습니다.")
                else:
                    if target_label == "우울_binary":
                        y_title = "우울감 경험률(%)"
                    else:
                        y_title = "자살생각 경험률(%)"
                    fig = px.bar(
                        grp,
                        x=x_label,
                        y="값",
                        title=f"{x_label}별 {y_title}",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(grp)
            else:
                # 예/아니오 응답의 '예' 비율
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

# --------------------
# 탭2: 2차원 비교 (축 2개만)
# --------------------
with tab2:
    st.subheader("2차원 비교 (축 2개)")

    if len(df) == 0:
        st.warning("데이터가 없습니다.")
    else:
        # 축 후보: 범주형 캐릭터 변수
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

            # 2차원 비교에서도 **축 2개 외의 필터는 없음**

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

# --------------------
# 탭3: 원자료
# --------------------
with tab_raw:
    st.subheader("원자료 미리보기 (라벨 컬럼 포함)")
    st.dataframe(df.head(50))

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="전체 데이터 CSV 다운로드",
        data=csv,
        file_name="kchs_all.csv",
        mime="text/csv",
    )
