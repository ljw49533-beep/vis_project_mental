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

# ====================
# 캐릭터 필터 (좌측 사이드바)
# ====================
st.sidebar.header("캐릭터 필터")

# 만 나이
if "만 나이" in df.columns:
    age_min = int(np.nanmin(df["만 나이"]))
    age_max = int(np.nanmax(df["만 나이"]))
    selected_age_range = st.sidebar.slider(
        "만 나이 범위", min_value=age_min, max_value=age_max, value=(age_min, age_max)
    )
else:
    selected_age_range = None

# 나이 구간
if "나이 구간(10살 단위)_str" in df.columns:
    age_bins = sorted(df["나이 구간(10살 단위)_str"].dropna().unique())
    selected_age_bins = st.sidebar.multiselect(
        "나이 구간(10살 단위)", age_bins, default=age_bins
    )
else:
    selected_age_bins = None

# 그 외 캐릭터 필터
character_filters = {}

# 필터에 쓸 캐릭터 라벨 목록
char_labels = []
for code, label in character_cols.items():
    if label in df.columns:
        char_labels.append(label)

for label in char_labels:
    if label in ["만 나이", "나이 구간(10살 단위)_str"]:
        continue
    col_data = df[label]
    unique_vals = sorted(col_data.dropna().unique())
    if len(unique_vals) <= 1:
        continue

    # 수면 소요시간은 슬라이더, 나머지는 멀티셀렉트
    if pd.api.types.is_numeric_dtype(col_data) and "수면 소요시간" in label:
        min_v = float(np.nanmin(col_data))
        max_v = float(np.nanmax(col_data))
        character_filters[label] = st.sidebar.slider(
            label, min_value=min_v, max_value=max_v, value=(min_v, max_v)
        )
    else:
        character_filters[label] = st.sidebar.multiselect(
            label, unique_vals, default=unique_vals
        )

# ====================
# 캐릭터 필터 적용
# ====================
filtered = df.copy()

# 만 나이 범위
if selected_age_range is not None and "만 나이" in filtered.columns:
    filtered = filtered[
        (filtered["만 나이"] >= selected_age_range[0])
        & (filtered["만 나이"] <= selected_age_range[1])
    ]

# 나이 구간
if selected_age_bins is not None and "나이 구간(10살 단위)_str" in filtered.columns:
    filtered = filtered[filtered["나이 구간(10살 단위)_str"].isin(selected_age_bins)]

# 나머지 캐릭터 필터
for label, cond in character_filters.items():
    if label not in filtered.columns:
        continue
    col = filtered[label]
    if isinstance(cond, tuple) and pd.api.types.is_numeric_dtype(col):
        filtered = filtered[(col >= cond[0]) & (col <= cond[1])]
    elif isinstance(cond, list) and len(cond) < len(col.dropna().unique()):
        filtered = filtered[col.isin(cond)]

st.caption(f"캐릭터 필터 적용 후 표본 수: {len(filtered):,}명")

# ====================
# 공통 유틸: 그룹별 우울 지표 요약
# ====================
def summarize_depression(df_in, target_label):
    """
    target_label: 우울 관련 한글 라벨 또는 '우울_binary' / '자살생각_binary'
    """
    if target_label in ["우울_binary", "자살생각_binary"]:
        target_col = target_label
        series = df_in[target_col]
        if series.dropna().empty:
            return np.nan, np.nan
        rate = np.nanmean(series) * 100
        n = series.notna().sum()
        return rate, n

    # 범주형 응답(예/아니오 등)의 '예' 비율
    if target_label not in df_in.columns:
        return np.nan, np.nan

    s = df_in[target_label].astype(str)
    if s.dropna().empty:
        return np.nan, np.nan
    yes_mask = s == "예"
    if yes_mask.sum() == 0 and s.nunique() > 2:
        # 스트레스 수준처럼 다범주인 경우, 평균값(코드) 반환은 여기서는 생략
        return np.nan, s.notna().sum()
    rate = yes_mask.mean() * 100
    n = s.notna().sum()
    return rate, n

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
    # target이 이진이면 퍼센트로 해석
    if set(np.unique(temp[target_col])) <= {0, 1}:
        grp["값"] = grp["값"] * 100
    return grp

# ====================
# 탭 구성: 1차원 / 2차원 / 원자료
# ====================
tab1, tab2, tab3 = st.tabs(["1차원 비교", "2차원 비교", "원자료"])

# --------------------
# 탭1: 1차원 비교
# --------------------
with tab1:
    st.subheader("1차원 비교 (축 1개)")
    if len(filtered) == 0:
        st.warning("필터 조건에 맞는 데이터가 없습니다.")
    else:
        # 축 후보: 캐릭터 변수 라벨 + 나이 구간
        axis_candidates = []
        if "나이 구간(10살 단위)_str" in filtered.columns:
            axis_candidates.append("나이 구간(10살 단위)_str")
        for label in char_labels:
            if label in filtered.columns and label not in axis_candidates:
                axis_candidates.append(label)

        x_label = st.selectbox("비교할 축(캐릭터 변수)", axis_candidates)

        # 우울 지표 후보
        dep_candidates = list(depression_cols.values()) + ["우울_binary", "자살생각_binary"]
        dep_candidates = [c for c in dep_candidates if c in filtered.columns]
        target_label = st.selectbox("우울 지표 선택", dep_candidates)

        if x_label and target_label:
            # 이진 우울 변수 선택 시: 비율 계산
            if target_label in ["우울_binary", "자살생각_binary"]:
                grp = group_rate(filtered, x_label, target_label)
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
                # 범주형 우울 지표(예/아니오) → '예' 비율을 따로 계산
                tmp = filtered[[x_label, target_label]].copy().dropna()
                if tmp.empty:
                    st.warning("선택한 축과 우울 지표 조합에 데이터가 없습니다.")
                else:
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
# 탭2: 2차원 비교
# --------------------
with tab2:
    st.subheader("2차원 비교 (축 2개)")

    if len(filtered) == 0:
        st.warning("필터 조건에 맞는 데이터가 없습니다.")
    else:
        # 축 후보 (범주형 위주)
        axis_candidates = []
        if "나이 구간(10살 단위)_str" in filtered.columns:
            axis_candidates.append("나이 구간(10살 단위)_str")
        for label in char_labels:
            if label in filtered.columns and not pd.api.types.is_numeric_dtype(filtered[label]):
                if label not in axis_candidates:
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
            dep_candidates = [c for c in dep_candidates if c in filtered.columns]
            target_label = st.selectbox("우울 지표 선택", dep_candidates, key="dep2d")

            if axis1 == axis2:
                st.warning("축 1과 축 2는 서로 다른 변수를 선택해야 합니다.")
            else:
                tmp = filtered[[axis1, axis2]].copy()
                if target_label in ["우울_binary", "자살생각_binary"]:
                    tmp[target_label] = filtered[target_label]
                    value_col = target_label
                    is_binary = True
                else:
                    tmp[target_label] = filtered[target_label].astype(str)
                    tmp["is_yes"] = (tmp[target_label] == "예").astype(float)
                    value_col = "is_yes"
                    is_binary = True

                tmp = tmp[[axis1, axis2, value_col]].dropna()
                if tmp.empty:
                    st.warning("선택한 변수 조합에 데이터가 없습니다.")
                else:
                    # 피벗 테이블 (평균값)
                    pivot = (
                        tmp.groupby([axis1, axis2])[value_col]
                        .mean()
                        .reset_index()
                    )
                    pivot["값"] = pivot[value_col] * 100 if is_binary else pivot[value_col]

                    heat = pivot.pivot(index=axis1, columns=axis2, values="값")

                    st.markdown("#### 교차표 (행: 축 1, 열: 축 2)")
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
with tab3:
    st.subheader("캐릭터 필터 적용 후 원자료 미리보기")
    st.dataframe(filtered.head(50))

    csv = filtered.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="현재 필터 적용 데이터 CSV 다운로드",
        data=csv,
        file_name="kchs_filtered.csv",
        mime="text/csv",
    )
