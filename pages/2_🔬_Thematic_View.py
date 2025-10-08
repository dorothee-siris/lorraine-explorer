# pages/2_🔬_Topic_View.py
# Streamlit "Topic View" — two perspectives: Domains and Fields
# - Reads:
#     data/ul_domains_indicators.parquet
#     data/ul_fields_indicators.parquet
# - Self-contained: no app-level dependencies beyond Streamlit, pandas, numpy, altair
# - Visuals:
#     * Overview tables with progress bars
#     * FWCI whisker charts (min–Q1–median–Q3–max)
#     * Drilldowns: labs, partners, authors, distributions (fields/subfields)
# - Optional: upload alternative parquet files or a benchmark file to compare shares

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="🔬 Topic View — Domains & Fields",
    page_icon="🔬",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_parquet(path_or_buffer):
    try:
        return pd.read_parquet(path_or_buffer)
    except Exception as e:
        st.error(f"Could not read parquet: {e}")
        return pd.DataFrame()


def _ensure_fraction(series: pd.Series) -> pd.Series:
    """Ensure percentage-looking columns are fractions in [0,1].
    Accepts values already in [0,1] or [0,100]."""
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().max() is not None and s.dropna().max() > 1.000001:
        s = s / 100.0
    return s


def _pipe_split(cell: object) -> List[str]:
    """Split 'a|b|c' -> list[str]; robust to NaN/None."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    text = str(cell).strip()
    if not text:
        return []
    return [p.strip() for p in text.split("|") if p.strip()]


def _pipe_to_numeric(cell: object, dtype=float) -> List:
    return [dtype(x) if str(x).strip() else np.nan for x in _pipe_split(cell)]


_kv_re = re.compile(r"^(?P<key>.+?)\s*\(\s*(?P<val>-?\d+(?:\.\d+)?)\s*\)\s*$")


def parse_kv_list(cell: object) -> List[Tuple[str, float]]:
    """
    Parse 'key(value) | key2(value2)' into list[(key, value)].
    Works for counts or percentages (values are floats).
    """
    items = []
    for part in _pipe_split(cell):
        m = _kv_re.match(part)
        if m:
            key = m.group("key").strip()
            val = float(m.group("val"))
            # If the value is an integer (like counts), cast to int for display niceness
            if abs(val - int(val)) < 1e-9:
                val = int(val)
            items.append((key, val))
    return items


def kv_to_df(cell: object, key_name="Key", value_name="Value") -> pd.DataFrame:
    pairs = parse_kv_list(cell)
    if not pairs:
        return pd.DataFrame(columns=[key_name, value_name])
    return pd.DataFrame(pairs, columns=[key_name, value_name])


def _progress_cols_to_config(percent_cols: Iterable[str]) -> Dict[str, st.column_config.ProgressColumn]:
    cfg = {}
    for c in percent_cols:
        cfg[c] = st.column_config.ProgressColumn(
            c,
            help="Percentage",
            format="%.1f%%",
            min_value=0.0,
            max_value=1.0,
        )
    return cfg


def _number_cols_to_config(cols: Iterable[str], fmt="%.0f") -> Dict[str, st.column_config.NumberColumn]:
    cfg = {}
    for c in cols:
        cfg[c] = st.column_config.NumberColumn(c, format=fmt)
    return cfg


def whisker_chart(
    df_stats: pd.DataFrame,
    label_col: str,
    min_col: str,
    q1_col: str,
    med_col: str,
    q3_col: str,
    max_col: str,
    title: str,
    log_scale: bool = False,
    height: int = 28,
):
    """
    Build a horizontal min–Q1–median–Q3–max whisker using Altair.
    df_stats: rows per entity (e.g., domain or lab)
    """
    if df_stats.empty:
        return alt.LayerChart()

    scale = alt.Scale(type="log") if log_scale else alt.Scale(type="linear", nice=True)

    base = alt.Chart(df_stats).transform_calculate(
        label=f"datum['{label_col}']"
    )

    rule = base.mark_rule().encode(
        y=alt.Y(f"{label_col}:N", sort='-x', title=None),
        x=alt.X(f"{min_col}:Q", scale=scale, title="FWCI (France)"),
        x2=f"{max_col}:Q",
        tooltip=[
            alt.Tooltip(f"{label_col}:N", title="Name"),
            alt.Tooltip(f"{min_col}:Q", title="Min", format=".2f"),
            alt.Tooltip(f"{q1_col}:Q", title="Q1", format=".2f"),
            alt.Tooltip(f"{med_col}:Q", title="Median", format=".2f"),
            alt.Tooltip(f"{q3_col}:Q", title="Q3", format=".2f"),
            alt.Tooltip(f"{max_col}:Q", title="Max", format=".2f"),
        ],
    )

    box = base.mark_bar(height=height).encode(
        y=alt.Y(f"{label_col}:N", sort='-x', title=None),
        x=alt.X(f"{q1_col}:Q", scale=scale, title="FWCI (France)"),
        x2=f"{q3_col}:Q",
    )

    median_tick = base.mark_tick(thickness=2, size=18).encode(
        y=alt.Y(f"{label_col}:N", sort='-x', title=None),
        x=alt.X(f"{med_col}:Q", scale=scale, title="FWCI (France)"),
    )

    chart = (rule + box + median_tick).properties(
        title=title, height=max(240, height * (len(df_stats) + 2))
    )
    return chart


def _download_button(df: pd.DataFrame, label: str, file_name: str):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
        use_container_width=True,
        type="secondary",
    )


def _maybe_fractionize(df: pd.DataFrame, percent_cols: List[str]) -> pd.DataFrame:
    for c in percent_cols:
        if c in df.columns:
            df[c] = _ensure_fraction(df[c])
    return df


def _topn(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    if df.empty:
        return df
    return df.head(n).copy()


# -----------------------------------------------------------------------------
# Data inputs (defaults + optional uploads)
# -----------------------------------------------------------------------------
st.sidebar.subheader("📥 Data")
default_domains_path = Path("data/ul_domains_indicators.parquet")
default_fields_path = Path("data/ul_fields_indicators.parquet")

domains_file = st.sidebar.file_uploader("ul_domains_indicators.parquet", type=["parquet"], key="domains_upload")
fields_file = st.sidebar.file_uploader("ul_fields_indicators.parquet", type=["parquet"], key="fields_upload")
benchmark_file = st.sidebar.file_uploader("Optional benchmark file (same schema) to compare shares", type=["parquet"], key="benchmark_upload")

df_domains = load_parquet(domains_file if domains_file else default_domains_path)
df_fields = load_parquet(fields_file if fields_file else default_fields_path)
df_benchmark = load_parquet(benchmark_file) if benchmark_file else pd.DataFrame()

if df_domains.empty and df_fields.empty:
    st.warning("No data loaded. Place parquet files in `data/` or upload them from the sidebar.")
    st.stop()

# Normalize percentage columns to fractions [0,1]
DOMAIN_PERCENT_COLS = [
    "% Pubs (uni level)",
    "% Pubs LUE (domain level)",
    "% Pubs LUE (uni level)",
    "% PPtop10% (domain level)",
    "% PPtop10% (uni level)",
    "% PPtop1% (domain level)",
    "% PPtop1% (uni level)",
    "% internal collaboration",
    "% international",
    "% industrial",
]
FIELD_PERCENT_COLS = [
    "% Pubs (uni level)",
    "% Pubs LUE (field level)",
    "% Pubs LUE (uni level)",
    "% PPtop10% (field level)",
    "% PPtop10% (uni level)",
    "% PPtop1% (field level)",
    "% PPtop1% (uni level)",
    "% internal collaboration",
    "% international",
    "% industrial",
]
df_domains = _maybe_fractionize(df_domains, DOMAIN_PERCENT_COLS)
df_fields = _maybe_fractionize(df_fields, FIELD_PERCENT_COLS)
if not df_benchmark.empty:
    # Try to align expected distribution columns for benchmark comparisons
    for cols in (DOMAIN_PERCENT_COLS, FIELD_PERCENT_COLS):
        df_benchmark = _maybe_fractionize(df_benchmark, cols)


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("🔬 Topic View")
st.caption(
    "Explain **Domains** and **Fields** through labs' contribution, citation impact (FWCI, top10%, top1%), and collaboration (internal / international / industrial)."
)

tab_domains, tab_fields = st.tabs(["🧭 Domains", "🧩 Fields"])

# -----------------------------------------------------------------------------
# TAB 1: DOMAINS
# -----------------------------------------------------------------------------
with tab_domains:
    st.subheader("Domain Overview")

    # Filters
    colA, colB, colC = st.columns([1.5, 1, 1])
    domain_search = colA.text_input("Search domain name", placeholder="e.g., Physical Sciences")
    sort_by = colB.selectbox(
        "Sort by",
        ["Pubs", "Avg FWCI (France)", "% PPtop10% (domain level)", "% PPtop1% (domain level)"],
        index=0,
    )
    ascending = colC.toggle("Ascending sort", value=False)

    dfD = df_domains.copy()
    if domain_search:
        dfD = dfD[dfD["Domain name"].str.contains(domain_search, case=False, na=False)]

    dfD = dfD.sort_values(sort_by, ascending=ascending)

    # Render table with progress bars
    domain_overview_cols = [
        "Domain name",
        "Pubs",
        "% Pubs (uni level)",
        "Pubs LUE",
        "% Pubs LUE (domain level)",
        "% Pubs LUE (uni level)",
        "PPtop10%",
        "% PPtop10% (domain level)",
        "% PPtop10% (uni level)",
        "PPtop1%",
        "% PPtop1% (domain level)",
        "% PPtop1% (uni level)",
        "Avg FWCI (France)",
        "FWCI_FR min",
        "FWCI_FR Q1",
        "FWCI_FR Q2",
        "FWCI_FR Q3",
        "FWCI_FR max",
        "% internal collaboration",
        "% international",
        "% industrial",
    ]
    present_cols = [c for c in domain_overview_cols if c in dfD.columns]
    st.dataframe(
        dfD[present_cols],
        hide_index=True,
        use_container_width=True,
        height=480,
        column_config={
            **_number_cols_to_config(["Pubs", "Pubs LUE", "PPtop10%", "PPtop1%"], fmt="%.0f"),
            **_number_cols_to_config(["Avg FWCI (France)", "FWCI_FR min", "FWCI_FR Q1", "FWCI_FR Q2", "FWCI_FR Q3", "FWCI_FR max"], fmt="%.2f"),
            **_progress_cols_to_config([c for c in present_cols if c in DOMAIN_PERCENT_COLS]),
        },
    )
    _download_button(dfD[present_cols], "⬇️ Download overview (CSV)", "domains_overview.csv")

    # Whisker chart (FWCI by domain)
    st.markdown("### FWCI Distribution by Domain")
    log_scale_domains = st.toggle("Log scale (FWCI)", value=False, key="domains_log")
    stats_cols = ["Domain name", "FWCI_FR min", "FWCI_FR Q1", "FWCI_FR Q2", "FWCI_FR Q3", "FWCI_FR max"]
    stats_df = dfD[[c for c in stats_cols if c in dfD.columns]].rename(
        columns={"Domain name": "Name"}
    )
    stats_df = stats_df.dropna(subset=["FWCI_FR min", "FWCI_FR Q1", "FWCI_FR Q2", "FWCI_FR Q3", "FWCI_FR max"])
    chart = whisker_chart(
        stats_df.rename(columns={"Name": "Domain"}),
        label_col="Domain",
        min_col="FWCI_FR min",
        q1_col="FWCI_FR Q1",
        med_col="FWCI_FR Q2",
        q3_col="FWCI_FR Q3",
        max_col="FWCI_FR max",
        title="FWCI (France baseline): min–Q1–median–Q3–max",
        log_scale=log_scale_domains,
    )
    st.altair_chart(chart, use_container_width=True)

    st.markdown("---")
    st.subheader("🔎 Drilldown by Domain")

    # Domain selector for drilldown
    all_domains = df_domains["Domain name"].dropna().unique().tolist()
    selected_domain = st.selectbox("Select a domain", options=all_domains)

    if selected_domain:
        rowD = df_domains[df_domains["Domain name"] == selected_domain].head(1)
        if rowD.empty:
            st.info("No data for selected domain.")
        else:
            left, right = st.columns([1, 1])

            # Field distribution within the domain
            with left:
                st.markdown("#### Field distribution in domain")
                df_fields_counts = kv_to_df(rowD["By field: count"].iloc[0], key_name="Field ID", value_name="Pubs")
                df_fields_share = kv_to_df(rowD["By field: % of domain pubs"].iloc[0], key_name="Field ID", value_name="Share")

                df_fields_dist = pd.merge(df_fields_counts, df_fields_share, on="Field ID", how="outer")
                # Enrich with field names if available
                if "Field ID" in df_fields.columns and "Field name" in df_fields.columns:
                    field_names = df_fields[["Field ID", "Field name"]].drop_duplicates()
                    df_fields_dist = df_fields_dist.merge(field_names, on="Field ID", how="left")

                if not df_fields_dist.empty:
                    df_fields_dist["Share"] = _ensure_fraction(df_fields_dist["Share"])
                    df_fields_dist = df_fields_dist.sort_values("Pubs", ascending=False)
                    # Bar chart
                    chart_fields = (
                        alt.Chart(df_fields_dist)
                        .mark_bar()
                        .encode(
                            y=alt.Y("Field name:N", sort="-x", title=None),
                            x=alt.X("Pubs:Q", title="Publications"),
                            tooltip=[
                                alt.Tooltip("Field name:N", title="Field"),
                                alt.Tooltip("Pubs:Q", title="Pubs", format=",.0f"),
                                alt.Tooltip("Share:Q", title="Share", format=".1%"),
                            ],
                        )
                        .properties(height=min(480, 24 * len(df_fields_dist) + 40), title=f"Fields within {selected_domain}")
                    )
                    st.altair_chart(chart_fields, use_container_width=True)
                    _download_button(df_fields_dist, "⬇️ Download field distribution (CSV)", f"{selected_domain}_field_distribution.csv")
                else:
                    st.info("No field distribution data.")

            # Labs contribution within the domain
            with right:
                st.markdown("#### Labs contribution")
                df_lab_counts = kv_to_df(rowD["By lab: count"].iloc[0], key_name="Lab", value_name="Pubs")
                df_lab_share = kv_to_df(rowD["By lab: % of domain pubs"].iloc[0], key_name="Lab", value_name="Share")
                df_lab = pd.merge(df_lab_counts, df_lab_share, on="Lab", how="outer")
                df_lab["Share"] = _ensure_fraction(df_lab["Share"])
                df_lab = df_lab.sort_values("Pubs", ascending=False)

                if not df_lab.empty:
                    chart_labs = (
                        alt.Chart(df_lab)
                        .mark_bar()
                        .encode(
                            y=alt.Y("Lab:N", sort="-x", title=None),
                            x=alt.X("Pubs:Q", title="Publications"),
                            tooltip=[
                                alt.Tooltip("Lab:N"),
                                alt.Tooltip("Pubs:Q", format=",.0f"),
                                alt.Tooltip("Share:Q", format=".1%"),
                            ],
                        )
                        .properties(height=min(480, 24 * len(df_lab) + 40))
                    )
                    st.altair_chart(chart_labs, use_container_width=True)
                    _download_button(df_lab, "⬇️ Download lab contributions (CSV)", f"{selected_domain}_labs.csv")
                else:
                    st.info("No lab contribution data.")

            st.markdown("#### FWCI Whiskers by Lab")
            # FWCI by lab within domain
            labs_fwci_cols = [
                "By lab: FWCI_FR min",
                "By lab: FWCI_FR Q1",
                "By lab: FWCI_FR Q2",
                "By lab: FWCI_FR Q3",
                "By lab: FWCI_FR max",
            ]
            lab_fwci_frames = []
            for c in labs_fwci_cols:
                df_tmp = kv_to_df(rowD[c].iloc[0], key_name="Lab", value_name=c)
                lab_fwci_frames.append(df_tmp)
            if lab_fwci_frames:
                df_lfwci = lab_fwci_frames[0]
                for extra in lab_fwci_frames[1:]:
                    df_lfwci = df_lfwci.merge(extra, on="Lab", how="outer")
                df_lfwci = df_lfwci.dropna()
                st.altair_chart(
                    whisker_chart(
                        df_lfwci,
                        label_col="Lab",
                        min_col="By lab: FWCI_FR min",
                        q1_col="By lab: FWCI_FR Q1",
                        med_col="By lab: FWCI_FR Q2",
                        q3_col="By lab: FWCI_FR Q3",
                        max_col="By lab: FWCI_FR max",
                        title=f"FWCI by lab — {selected_domain}",
                        log_scale=False,
                        height=22,
                    ),
                    use_container_width=True,
                )
                _download_button(df_lfwci, "⬇️ Download labs FWCI stats (CSV)", f"{selected_domain}_labs_fwci.csv")
            else:
                st.info("No FWCI-by-lab stats.")

            # Partners
            st.markdown("#### Top partners in this domain")
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                st.markdown("##### 🇫🇷 French partners")
                fr_names = _pipe_split(rowD["Top 20 FR partners (name)"].iloc[0])
                fr_counts = _pipe_to_numeric(rowD["Top 20 FR partners (totals copubs in this domain)"].iloc[0], int)
                df_fr = pd.DataFrame({"Partner": fr_names, "Co-pubs": fr_counts}).sort_values("Co-pubs", ascending=False)
                st.dataframe(_topn(df_fr, 20), hide_index=True, use_container_width=True)
                _download_button(df_fr, "⬇️ Download FR partners", f"{selected_domain}_partners_FR.csv")
            with pcol2:
                st.markdown("##### 🌍 International partners")
                int_names = _pipe_split(rowD["Top 20 int partners (name)"].iloc[0])
                int_counts = _pipe_to_numeric(rowD["Top 20 int partners (totals copubs in this domain)"].iloc[0], int)
                int_countries = _pipe_split(rowD.get("Top 20 int partners (country)", [""] * len(int_names)).iloc[0] if "Top 20 int partners (country)" in rowD else "")
                df_int = pd.DataFrame({"Partner": int_names, "Country": int_countries, "Co-pubs": int_counts}).sort_values("Co-pubs", ascending=False)
                st.dataframe(_topn(df_int, 20), hide_index=True, use_container_width=True)
                _download_button(df_int, "⬇️ Download INT partners", f"{selected_domain}_partners_INT.csv")

            # Authors
            st.markdown("#### Top authors in this domain")
            a_names = _pipe_split(rowD["Top 20 authors (name)"].iloc[0])
            a_pubs = _pipe_to_numeric(rowD["Top 20 authors (pubs)"].iloc[0], int)
            a_fwci = _pipe_to_numeric(rowD["Top 20 authors (Average FWCI_FR)"].iloc[0], float)
            df_auth = pd.DataFrame({"Author": a_names, "Pubs": a_pubs, "Avg FWCI_FR": a_fwci}).sort_values(["Pubs", "Avg FWCI_FR"], ascending=[False, False])
            st.dataframe(_topn(df_auth, 20), hide_index=True, use_container_width=True, column_config=_number_cols_to_config(["Pubs"], "%.0f") | _number_cols_to_config(["Avg FWCI_FR"], "%.2f"))
            _download_button(df_auth, "⬇️ Download authors", f"{selected_domain}_authors.csv")

            # Optional comparison of field share within domain vs benchmark
            if not df_benchmark.empty:
                st.markdown("#### 📊 Compare field shares with benchmark (optional)")
                # Find same domain in benchmark
                bench_row = df_benchmark[df_benchmark.get("Domain name", "") == selected_domain].head(1)
                if not bench_row.empty and "By field: % of domain pubs" in bench_row.columns:
                    df_me_share = kv_to_df(rowD["By field: % of domain pubs"].iloc[0], key_name="Field ID", value_name="UL Share")
                    df_bench_share = kv_to_df(bench_row["By field: % of domain pubs"].iloc[0], key_name="Field ID", value_name="Benchmark Share")
                    comp = df_me_share.merge(df_bench_share, on="Field ID", how="outer")
                    # Map field names if available
                    if "Field ID" in df_fields.columns and "Field name" in df_fields.columns:
                        comp = comp.merge(df_fields[["Field ID", "Field name"]].drop_duplicates(), on="Field ID", how="left")
                    comp["UL Share"] = _ensure_fraction(comp["UL Share"])
                    comp["Benchmark Share"] = _ensure_fraction(comp["Benchmark Share"])
                    comp = comp.fillna(0.0)
                    chart_comp = (
                        alt.Chart(comp.melt(id_vars=["Field ID", "Field name"], value_vars=["UL Share", "Benchmark Share"], var_name="Who", value_name="Share"))
                        .mark_bar()
                        .encode(
                            y=alt.Y("Field name:N", sort="-x", title=None),
                            x=alt.X("Share:Q", axis=alt.Axis(format="%")),
                            color=alt.Color("Who:N"),
                            tooltip=[alt.Tooltip("Field name:N"), alt.Tooltip("Who:N"), alt.Tooltip("Share:Q", format=".1%")],
                        )
                        .properties(height=min(500, 26 * comp.shape[0] + 40), title=f"Share within {selected_domain}: UL vs Benchmark")
                    )
                    st.altair_chart(chart_comp, use_container_width=True)
                    _download_button(comp, "⬇️ Download share comparison", f"{selected_domain}_field_share_vs_benchmark.csv")
                else:
                    st.info("No matching domain in benchmark or missing share column.")

# -----------------------------------------------------------------------------
# TAB 2: FIELDS
# -----------------------------------------------------------------------------
with tab_fields:
    st.subheader("Field Overview")

    # Filters
    c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
    # Optional domain filter to subset fields by domain membership, if joinable via "By field: ..." is not present here.
    # We assume "Field name" is unique; domain filter not strictly needed unless the dataset includes it.
    field_search = c1.text_input("Search field name", placeholder="e.g., Engineering")
    sort_field = c2.selectbox(
        "Sort by",
        ["Pubs", "Avg FWCI (France)", "% PPtop10% (field level)", "% PPtop1% (field level)"],
        index=0,
    )
    asc_field = c3.toggle("Ascending sort", value=False, key="fields_sort_asc")
    log_scale_fields = c4.toggle("Log scale (FWCI whiskers)", value=True, key="fields_log")

    dfF = df_fields.copy()
    if field_search:
        dfF = dfF[dfF["Field name"].str.contains(field_search, case=False, na=False)]
    dfF = dfF.sort_values(sort_field, ascending=asc_field)

    field_overview_cols = [
        "Field name",
        "Pubs",
        "% Pubs (uni level)",
        "Pubs LUE",
        "% Pubs LUE (field level)",
        "% Pubs LUE (uni level)",
        "PPtop10%",
        "% PPtop10% (field level)",
        "% PPtop10% (uni level)",
        "PPtop1%",
        "% PPtop1% (field level)",
        "% PPtop1% (uni level)",
        "Avg FWCI (France)",
        "FWCI_FR min",
        "FWCI_FR Q1",
        "FWCI_FR Q2",
        "FWCI_FR Q3",
        "FWCI_FR max",
        "% internal collaboration",
        "% international",
        "% industrial",
    ]
    present_cols_f = [c for c in field_overview_cols if c in dfF.columns]
    st.dataframe(
        dfF[present_cols_f],
        hide_index=True,
        use_container_width=True,
        height=480,
        column_config={
            **_number_cols_to_config(["Pubs", "Pubs LUE", "PPtop10%", "PPtop1%"], fmt="%.0f"),
            **_number_cols_to_config(["Avg FWCI (France)", "FWCI_FR min", "FWCI_FR Q1", "FWCI_FR Q2", "FWCI_FR Q3", "FWCI_FR max"], fmt="%.2f"),
            **_progress_cols_to_config([c for c in present_cols_f if c in FIELD_PERCENT_COLS]),
        },
    )
    _download_button(dfF[present_cols_f], "⬇️ Download overview (CSV)", "fields_overview.csv")

    # FWCI whiskers by field (log-scale default)
    st.markdown("### FWCI Distribution by Field")
    stats_cols_f = ["Field name", "FWCI_FR min", "FWCI_FR Q1", "FWCI_FR Q2", "FWCI_FR Q3", "FWCI_FR max"]
    stats_ff = dfF[[c for c in stats_cols_f if c in dfF.columns]].rename(columns={"Field name": "Field"})
    stats_ff = stats_ff.dropna(subset=["FWCI_FR min", "FWCI_FR Q1", "FWCI_FR Q2", "FWCI_FR Q3", "FWCI_FR max"])
    chartF = whisker_chart(
        stats_ff,
        label_col="Field",
        min_col="FWCI_FR min",
        q1_col="FWCI_FR Q1",
        med_col="FWCI_FR Q2",
        q3_col="FWCI_FR Q3",
        max_col="FWCI_FR max",
        title="FWCI (France baseline): min–Q1–median–Q3–max",
        log_scale=log_scale_fields,
    )
    st.altair_chart(chartF, use_container_width=True)

    st.markdown("---")
    st.subheader("🔎 Drilldown by Field")

    # Field selector
    all_fields = df_fields["Field name"].dropna().unique().tolist()
    selected_field = st.selectbox("Select a field", options=all_fields)

    if selected_field:
        rowF = df_fields[df_fields["Field name"] == selected_field].head(1)
        if rowF.empty:
            st.info("No data for selected field.")
        else:
            lcol, rcol = st.columns([1, 1])

            # Subfield distribution
            with lcol:
                st.markdown("#### Subfield distribution in field")
                sf_counts = kv_to_df(rowF["By subfield: count"].iloc[0], key_name="Subfield ID", value_name="Pubs")
                sf_share = kv_to_df(rowF["By subfield: % of field pubs"].iloc[0], key_name="Subfield ID", value_name="Share")
                sf = pd.merge(sf_counts, sf_share, on="Subfield ID", how="outer")
                sf["Share"] = _ensure_fraction(sf["Share"])
                sf = sf.sort_values("Pubs", ascending=False)

                if not sf.empty:
                    # We might not have subfield names in this file; show ID as fallback
                    sf["Label"] = sf["Subfield ID"].astype(str)
                    chart_sf = (
                        alt.Chart(sf)
                        .mark_bar()
                        .encode(
                            y=alt.Y("Label:N", sort="-x", title=None),
                            x=alt.X("Pubs:Q", title="Publications"),
                            tooltip=[
                                alt.Tooltip("Label:N", title="Subfield"),
                                alt.Tooltip("Pubs:Q", format=",.0f"),
                                alt.Tooltip("Share:Q", format=".1%"),
                            ],
                        )
                        .properties(height=min(480, 24 * len(sf) + 40), title=f"Subfields within {selected_field}")
                    )
                    st.altair_chart(chart_sf, use_container_width=True)
                    _download_button(sf, "⬇️ Download subfield distribution", f"{selected_field}_subfields.csv")
                else:
                    st.info("No subfield distribution data.")

            # Labs contribution within field
            with rcol:
                st.markdown("#### Labs contribution")
                lf_counts = kv_to_df(rowF["By lab: count"].iloc[0], key_name="Lab", value_name="Pubs")
                lf_share = kv_to_df(rowF["By lab: % of field pubs"].iloc[0], key_name="Lab", value_name="Share")
                lf = pd.merge(lf_counts, lf_share, on="Lab", how="outer")
                lf["Share"] = _ensure_fraction(lf["Share"])
                lf = lf.sort_values("Pubs", ascending=False)

                if not lf.empty:
                    chart_lf = (
                        alt.Chart(lf)
                        .mark_bar()
                        .encode(
                            y=alt.Y("Lab:N", sort="-x", title=None),
                            x=alt.X("Pubs:Q", title="Publications"),
                            tooltip=[
                                alt.Tooltip("Lab:N"),
                                alt.Tooltip("Pubs:Q", format=",.0f"),
                                alt.Tooltip("Share:Q", format=".1%"),
                            ],
                        )
                        .properties(height=min(480, 24 * len(lf) + 40))
                    )
                    st.altair_chart(chart_lf, use_container_width=True)
                    _download_button(lf, "⬇️ Download lab contributions", f"{selected_field}_labs.csv")
                else:
                    st.info("No lab contribution data.")

            # FWCI by lab within field
            st.markdown("#### FWCI Whiskers by Lab")
            labs_fwci_cols_f = [
                "By lab: FWCI_FR min",
                "By lab: FWCI_FR Q1",
                "By lab: FWCI_FR Q2",
                "By lab: FWCI_FR Q3",
                "By lab: FWCI_FR max",
            ]
            lab_fwci_frames_f = []
            for c in labs_fwci_cols_f:
                df_tmp = kv_to_df(rowF[c].iloc[0], key_name="Lab", value_name=c)
                lab_fwci_frames_f.append(df_tmp)
            if lab_fwci_frames_f:
                df_lfwci_f = lab_fwci_frames_f[0]
                for extra in lab_fwci_frames_f[1:]:
                    df_lfwci_f = df_lfwci_f.merge(extra, on="Lab", how="outer")
                df_lfwci_f = df_lfwci_f.dropna()
                st.altair_chart(
                    whisker_chart(
                        df_lfwci_f,
                        label_col="Lab",
                        min_col="By lab: FWCI_FR min",
                        q1_col="By lab: FWCI_FR Q1",
                        med_col="By lab: FWCI_FR Q2",
                        q3_col="By lab: FWCI_FR Q3",
                        max_col="By lab: FWCI_FR max",
                        title=f"FWCI by lab — {selected_field}",
                        log_scale=False,
                        height=22,
                    ),
                    use_container_width=True,
                )
                _download_button(df_lfwci_f, "⬇️ Download labs FWCI stats (CSV)", f"{selected_field}_labs_fwci.csv")
            else:
                st.info("No FWCI-by-lab stats.")

            # Partners (Top 10)
            st.markdown("#### Top partners in this field")
            pf1, pf2 = st.columns(2)
            with pf1:
                st.markdown("##### 🇫🇷 French partners")
                fr_names = _pipe_split(rowF["Top 10 FR partners (name)"].iloc[0])
                fr_counts = _pipe_to_numeric(rowF["Top 10 FR partners (totals copubs in this field)"].iloc[0], int)
                df_fr = pd.DataFrame({"Partner": fr_names, "Co-pubs": fr_counts}).sort_values("Co-pubs", ascending=False)
                st.dataframe(df_fr, hide_index=True, use_container_width=True)
                _download_button(df_fr, "⬇️ Download FR partners", f"{selected_field}_partners_FR.csv")
            with pf2:
                st.markdown("##### 🌍 International partners")
                int_names = _pipe_split(rowF["Top 10 int partners (name)"].iloc[0])
                int_counts = _pipe_to_numeric(rowF["Top 10 int partners (totals copubs in this field)"].iloc[0], int)
                int_countries = _pipe_split(rowF.get("Top 10 int partners (country)", [""] * len(int_names)).iloc[0] if "Top 10 int partners (country)" in rowF else "")
                df_int = pd.DataFrame({"Partner": int_names, "Country": int_countries, "Co-pubs": int_counts}).sort_values("Co-pubs", ascending=False)
                st.dataframe(df_int, hide_index=True, use_container_width=True)
                _download_button(df_int, "⬇️ Download INT partners", f"{selected_field}_partners_INT.csv")

            # Authors (Top 10)
            st.markdown("#### Top authors in this field")
            a_names = _pipe_split(rowF["Top 10 authors (name)"].iloc[0])
            a_pubs = _pipe_to_numeric(rowF["Top 10 authors (pubs)"].iloc[0], int)
            a_fwci = _pipe_to_numeric(rowF["Top 10 authors (Average FWCI_FR)"].iloc[0], float)
            df_auth = pd.DataFrame({"Author": a_names, "Pubs": a_pubs, "Avg FWCI_FR": a_fwci}).sort_values(["Pubs", "Avg FWCI_FR"], ascending=[False, False])
            st.dataframe(df_auth, hide_index=True, use_container_width=True, column_config=_number_cols_to_config(["Pubs"], "%.0f") | _number_cols_to_config(["Avg FWCI_FR"], "%.2f"))
            _download_button(df_auth, "⬇️ Download authors", f"{selected_field}_authors.csv")

            # Optional: OpenAlex link if present
            if "See in OpenAlex" in rowF.columns and pd.notna(rowF["See in OpenAlex"].iloc[0]):
                st.link_button("🔗 See in OpenAlex", url=str(rowF["See in OpenAlex"].iloc[0]), use_container_width=False)

# -----------------------------------------------------------------------------
# Footer help
# -----------------------------------------------------------------------------
with st.expander("ℹ️ Tips & Notes"):
    st.markdown(
        """
- **Progress bars** assume percent columns are either in `[0,1]` or `[0,100]` and normalize automatically.
- **Whisker plots** show FWCI_FR range and quartiles. Toggle **log scale** for right-skewed distributions.
- Use the **optional benchmark upload** in the sidebar to compare field shares within a selected domain (e.g., *“within Physical Sciences, 23% vs 50%”*).
- All tables are downloadable as CSV.
"""
    )
