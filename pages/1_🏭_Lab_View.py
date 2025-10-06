from __future__ import annotations

"""
Lab View — LUE Portfolio Explorer (self‑contained, new data feed)
-----------------------------------------------------------------
This page relies only on:
  • data/pubs_final.parquet                 (publication level)
  • data/ul_units_indicators.parquet        (lab-level indicators)
  • data/ul_authors_indicators.parquet      (author-level indicators)
  • lib/taxonomy.py                         (optional; for names/order/colors)

It does NOT depend on other internal helpers. If taxonomy isn't available,
charts still render with a reasonable fallback ordering.

Important expected columns (case/spacing tolerant):
  pubs_final.parquet
    - OpenAlex ID | DOI | Title | Publication Year | Publication Type
    - Authors | Authors ID | Authors ORCID
    - Institution Types | Institution Countries | Institutions ROR
    - FWCI_FR | FWCI_all | Citation Count
    - Primary Topic | Primary Subfield ID | Primary Field ID | Primary Domain ID
    - In_LUE | Labs_RORs | Is_international | Is_company

  ul_units_indicators.parquet
    - ROR | Unit Name | Pubs | % Pubs (uni level) | Pubs LUE | % Pubs LUE (lab level)
    - % international | % industrial | Avg FWCI (France) | See in OpenAlex

  ul_authors_indicators.parquet
    - Author Name | Author ID | ORCID | Publications (unique) | Average FWCI_FR
    - Is Lorraine (optional) | Lab(s)/labs columns (optional)

Notes
-----
• Year filter controls plots and the dynamically generated OpenAlex links.
• Field distributions & co-publications are computed from publication-level data.
• Robust column normalization makes the app resilient to header variants.
"""

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Optional taxonomy import (safe if missing)
try:
    from lib import taxonomy  # type: ignore
except Exception:  # pragma: no cover
    taxonomy = None  # graceful fallback

# ---------------------------------------------------------------------
# Page config (must be before any output)
# ---------------------------------------------------------------------
st.set_page_config(page_title="Lab View — LUE Portfolio Explorer", page_icon="🏭", layout="wide")
st.title("🏭 Lab View")

# ---------------------------------------------------------------------
# Helpers — paths, caching, normalization
# ---------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

@st.cache_data(show_spinner=False)
def _read_parquet_safe(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # unify all column names for tolerant matching (but keep originals too)
    df.columns = [c.strip() for c in df.columns]
    return df

# --- pubs loader (normalize to snake_case) -----------------------------------

PUBS_RENAME = {
    # id/meta
    "OpenAlex ID": "openalex_id",
    "DOI": "doi",
    "Title": "title",
    "Publication Year": "year",
    "Publication Type": "pub_type",
    # authors
    "Authors": "authors",
    "Authors ID": "authors_id",
    "Authors ORCID": "authors_orcid",
    # institutions
    "Institutions ROR": "inst_ror",
    "Institution Types": "inst_types",
    "Institution Countries": "inst_countries",
    # metrics
    "FWCI_FR": "fwci_fr",
    "FWCI_all": "fwci_all",
    "Citation Count": "citation_count",
    "Citations per Year": "cites_per_year",
    # classification (primary)
    "Primary Topic": "primary_topic_id",
    "Primary Subfield ID": "primary_subfield_id",
    "Primary Field ID": "primary_field_id",
    "Primary Domain ID": "primary_domain_id",
    # flags
    "In_LUE": "in_lue",
    "Labs_RORs": "labs_rors",
    "Is_international": "is_international",
    "Is_company": "is_company",
}

@st.cache_data(show_spinner=False)
def load_pubs(path: Optional[Path] = None) -> pd.DataFrame:
    """Load publication-level data and normalize key columns."""
    path = path or (DATA_DIR / "pubs_final.parquet")
    df = _read_parquet_safe(path)

    # Case/variant tolerant renaming
    rename = {}
    for k, v in PUBS_RENAME.items():
        if k in df.columns:
            rename[k] = v
        else:
            # accept relaxed keys (common variants)
            alt_key = k.replace("_", " ")
            if alt_key in df.columns:
                rename[alt_key] = v
    df = df.rename(columns=rename)

    # type coercions
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    for c in ("fwci_fr", "fwci_all"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "citation_count" in df.columns:
        df["citation_count"] = pd.to_numeric(df["citation_count"], errors="coerce").astype("Int64")

    return df

# --- labs/units loader --------------------------------------------------------

UNITS_RENAME = {
    "ROR": "lab_ror",
    "OpenAlex ID": "lab_openalex_id",
    "Unit Name": "lab_name",
    "Department": "department",
    "Type": "unit_type",
    "Pubs": "pubs_19_23",
    "% Pubs (uni level)": "share_uni",
    "Pubs LUE": "pubs_lue",
    "% Pubs LUE (lab level)": "lue_pct_lab",
    "% Pubs LUE (uni level)": "lue_pct_uni",
    "% international": "intl_pct",
    "% industrial": "company_pct",
    "Avg FWCI (France)": "avg_fwci_fr",
    "See in OpenAlex": "openalex_ui_url",
}

@st.cache_data(show_spinner=False)
def load_labs(path: Optional[Path] = None) -> pd.DataFrame:
    """Load lab/unit indicators and normalize column names."""
    path = path or (DATA_DIR / "ul_units_indicators.parquet")
    df = _read_parquet_safe(path).rename(columns={k: v for k, v in UNITS_RENAME.items() if k in _read_parquet_safe(path).columns})

    # Coerce numerics when present
    for c in ("pubs_19_23", "share_uni", "pubs_lue", "lue_pct_lab", "lue_pct_uni", "intl_pct", "company_pct", "avg_fwci_fr"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Add ROR link
    if "lab_ror" in df.columns:
        df["ror_url"] = df["lab_ror"].apply(lambda r: f"https://ror.org/{str(r).strip()}" if pd.notna(r) and str(r).strip() else None)

    return df

# --- authors loader -----------------------------------------------------------

AUTH_RENAME = {
    "Author Name": "author_name",
    "Normalized Name": "normalized_name",
    "Author ID": "author_id_raw",  # may contain multiple IDs "A... | A..."
    "ORCID": "orcid",
    "Publications (unique)": "pubs_unique",
    "Average FWCI_FR": "avg_fwci_fr",
    "Is Lorraine": "is_lorraine",
}

@st.cache_data(show_spinner=False)
def load_authors(path: Optional[Path] = None) -> pd.DataFrame:
    path = path or (DATA_DIR / "ul_authors_indicators.parquet")
    df = _read_parquet_safe(path)
    df = df.rename(columns={k: v for k, v in AUTH_RENAME.items() if k in df.columns})

    # explode potential multiple author IDs per row into tidy rows
    if "author_id_raw" in df.columns:
        rows = []
        for _, r in df.iterrows():
            ids = [x.strip() for x in str(r.get("author_id_raw", "")).split("|") if x.strip()]
            if not ids:
                rows.append({**r, "author_id": None})
            else:
                for aid in ids:
                    rows.append({**r, "author_id": aid})
        df = pd.DataFrame(rows)
    else:
        df["author_id"] = None

    # keep only relevant columns
    keep = [c for c in ("author_id", "author_name", "orcid", "pubs_unique", "avg_fwci_fr", "is_lorraine") if c in df.columns]
    return df[keep].drop_duplicates()

# ---------------------------------------------------------------------
# Minimal transforms used on-page
# ---------------------------------------------------------------------
LEAD_NUM = re.compile(r"^\s*\[\d+\]\s*")


def clean_pipe_list(s: object) -> List[str]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    return [LEAD_NUM.sub("", x).strip() for x in str(s).replace(";", "|").split("|") if x and str(x).strip()]


def explode_labs(pubs: pd.DataFrame) -> pd.DataFrame:
    """Explode Labs_RORs into one row per (work, lab_ror)."""
    req = ["openalex_id", "year", "labs_rors"]
    for c in req:
        if c not in pubs.columns:
            raise ValueError(f"Missing required column in pubs: {c}")

    rows = []
    for _, r in pubs[req].iterrows():
        for lab in clean_pipe_list(r["labs_rors"]):
            rows.append({"openalex_id": r.openalex_id, "year": r.year, "lab_ror": lab})
    return pd.DataFrame(rows)


def openalex_works_url_for_lab(ror: str, year_min: int, year_max: int) -> str:
    
    # OA expects full dates
    return (
        "https://openalex.org/works?"
        f"filter=authorships.institutions.ror:{ror},"
        f"from_publication_date:{year_min}-01-01,"
        f"to_publication_date:{year_max}-12-31"
        "&sort=publication_year:desc"
    )


# Field ID -> name using taxonomy if available

def field_name_for_id(field_id: object) -> str:
    if pd.isna(field_id):
        return "Unknown"
    fid = str(int(field_id)) if str(field_id).isdigit() else str(field_id)
    if taxonomy is not None:
        return taxonomy.build_taxonomy_lookups()["id2name"].get(fid, fid)
    return fid


def canonical_field_order() -> List[str]:
    if taxonomy is not None:
        return taxonomy.canonical_field_order()
    # fallback from present data (sorted by name)
    return []


def color_for_field(field: str) -> str:
    if taxonomy is not None:
        return taxonomy.get_field_color(field)
    return "#7f7f7f"


# Bar chart used for field mixes (volume or percent)

def field_mix_bars(df: pd.DataFrame, *, value_col: str, percent: bool, enforce_order: Optional[List[str]] = None,
                   width: int = 560, height: int = 340) -> alt.Chart:
    data = df.copy()
    data["value"] = pd.to_numeric(data[value_col], errors="coerce").fillna(0.0)

    # ordering & colors
    cats = enforce_order or list(data["field_name"].unique())
    color_range = [color_for_field(f) for f in cats]

    y = alt.Y("field_name:N", sort=cats, title="Field")
    x_title = "% of lab publications" if percent else "Publications"
    x = alt.X("value:Q", title=x_title, scale=alt.Scale(domain=[0, 1]) if percent else alt.Undefined)

    tooltip = [alt.Tooltip("field_name:N", title="Field"), alt.Tooltip("value:Q", title=x_title, format=".0%" if percent else ",")]

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            y=y,
            x=x,
            color=alt.Color("field_name:N", sort=cats, scale=alt.Scale(domain=cats, range=color_range), legend=None),
            tooltip=tooltip,
        )
        .properties(width=width, height=height)
    )
    return chart

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
with st.spinner("Loading data…"):
    try:
        pubs = load_pubs()
    except Exception as e:
        st.error(f"Could not load pubs_final.parquet — {e}")
        st.stop()

    try:
        labs = load_labs()
    except Exception as e:
        st.error(f"Could not load ul_units_indicators.parquet — {e}")
        st.stop()

    try:
        authors_idx = load_authors()
    except Exception:
        authors_idx = pd.DataFrame(columns=["author_id", "author_name", "orcid", "pubs_unique", "avg_fwci_fr", "is_lorraine"])  # optional

# Derive default years from data
YEARS = sorted([int(y) for y in pubs["year"].dropna().unique()]) if "year" in pubs else list(range(2019, 2024))
if YEARS:
    YEAR_START, YEAR_END = min(YEARS), max(YEARS)
else:
    YEAR_START, YEAR_END = 2019, 2023

st.caption(f"Default period: {YEAR_START}–{YEAR_END}")

# ---------------------------------------------------------------------
# Topline metrics (from pubs)
# ---------------------------------------------------------------------
try:
    n_labs = int(pd.Series(sum(bool(clean_pipe_list(x)) for x in pubs.get("labs_rors", []))).sum())  # just to trigger dtype; unused
except Exception:
    n_labs = None

# unique labs present in labs file (more robust)
lab_count = int(labs["lab_ror"].nunique()) if "lab_ror" in labs else 0

p19_23 = pubs[(pubs["year"] >= YEAR_START) & (pubs["year"] <= YEAR_END)] if "year" in pubs else pubs.copy()

total_pubs = int(p19_23["openalex_id"].nunique()) if "openalex_id" in p19_23 else len(p19_23)

with_lab = 0
if "labs_rors" in p19_23:
    with_lab = int(p19_23.loc[p19_23["labs_rors"].map(lambda s: len(clean_pipe_list(s)) > 0), "openalex_id"].nunique())

coverage = (with_lab / total_pubs) if total_pubs else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Number of labs", f"{lab_count:,}")
k2.metric("Publications", f"{total_pubs:,}")
k3.metric("Publications with a lab", f"{with_lab:,}")
k4.metric("% covered by labs", f"{coverage*100:.1f}%")

st.divider()

# ---------------------------------------------------------------------
# Per‑lab overview (2019–2023 baseline from ul_units_indicators)
# ---------------------------------------------------------------------
st.subheader("Per‑lab overview")

summary = labs.copy()

# Display-friendly % columns (ul_units_indicators already stores fractions 0–1)
if "share_uni" in summary:
    summary["share_pct_display"] = summary["share_uni"] * 100.0
else:
    # fallback: compute from pubs
    lab_totals = (
        explode_labs(pubs)
        .merge(pubs[["openalex_id", "year"]], on="openalex_id", how="left")
        .query("@YEAR_START <= year <= @YEAR_END")
        .groupby("lab_ror")["openalex_id"].nunique()
    )
    overall = max(total_pubs, 1)
    summary["share_pct_display"] = summary["lab_ror"].map(lambda r: (lab_totals.get(r, 0) / overall) * 100.0)

for src, dst in (
    ("lue_pct_lab", "lue_pct_display"),
    ("intl_pct", "intl_pct_display"),
    ("company_pct", "company_pct_display"),
):
    if src in summary:
        summary[dst] = pd.to_numeric(summary[src], errors="coerce") * 100.0

# Metrics bounds for progress bars
max_share = float(summary.get("share_pct_display", pd.Series([0])).max() or 1.0)
max_lue   = float(summary.get("lue_pct_display", pd.Series([0])).max()   or 1.0)
max_intl  = float(summary.get("intl_pct_display", pd.Series([0])).max()  or 1.0)
max_comp  = float(summary.get("company_pct_display", pd.Series([0])).max() or 1.0)

# Ensure link columns
if "openalex_ui_url" not in summary and {"lab_ror"}.issubset(summary.columns):
    summary["openalex_ui_url"] = summary["lab_ror"].map(lambda r: openalex_works_url_for_lab(str(r), YEAR_START, YEAR_END))
if "ror_url" not in summary and "lab_ror" in summary:
    summary["ror_url"] = summary["lab_ror"].map(lambda r: f"https://ror.org/{r}")

# Column selection & config
default_cols = [
    "lab_name", "pubs_19_23", "share_pct_display",
    "lue_pct_display", "intl_pct_display", "company_pct_display",
    "avg_fwci_fr",
]
column_order = [c for c in default_cols + ["openalex_ui_url", "ror_url"] if c in summary.columns]

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True,
    column_order=column_order,
    column_config={
        "lab_name": st.column_config.TextColumn("Lab"),
        "pubs_19_23": st.column_config.NumberColumn("Publications", format="%.0f"),
        "share_pct_display":    st.column_config.ProgressColumn("% Université de Lorraine", format="%.1f %%", min_value=0.0, max_value=max_share),
        "lue_pct_display":      st.column_config.ProgressColumn("% of pubs LUE",            format="%.1f %%", min_value=0.0, max_value=max_lue),
        "intl_pct_display":     st.column_config.ProgressColumn("% international",          format="%.1f %%", min_value=0.0, max_value=max_intl),
        "company_pct_display":  st.column_config.ProgressColumn("% with company",           format="%.1f %%", min_value=0.0, max_value=max_comp),
        "avg_fwci_fr": st.column_config.NumberColumn("Avg. FWCI (France)", format="%.3f"),
        "openalex_ui_url": st.column_config.LinkColumn("See in OpenAlex"),
        "ror_url":         st.column_config.LinkColumn("See in ROR"),
    },
)

# ---------------------------------------------------------------------
# Year filter (affects plots + OpenAlex links)
# ---------------------------------------------------------------------
st.markdown("### Year filter")
years_all = YEARS
years_sel = st.multiselect("Filter years (affects the plots and OpenAlex links)", years_all, default=years_all)
if not years_sel:
    st.warning("Select at least one year.")
    st.stop()

year_min, year_max = min(years_sel), max(years_sel)

# Rebuild OpenAlex links for the chosen window
if {"lab_name", "lab_ror"}.issubset(summary.columns):
    links = (
        summary[["lab_name", "lab_ror"]]
        .drop_duplicates()
        .assign(openalex_ui_url=lambda d: d["lab_ror"].map(lambda r: openalex_works_url_for_lab(str(r), year_min, year_max)))
    )
    summary = summary.drop(columns=["openalex_ui_url"], errors="ignore").merge(links, on=["lab_name", "lab_ror"], how="left")

st.divider()

# ---------------------------------------------------------------------
# Compare two labs (field mixes from pubs)
# ---------------------------------------------------------------------
st.subheader("Compare two labs")

lab_options = summary[["lab_name", "lab_ror"]].dropna().drop_duplicates().sort_values("lab_name")
labels = lab_options["lab_name"].tolist()
name_to_ror = dict(zip(lab_options["lab_name"], lab_options["lab_ror"]))

c1, c2 = st.columns(2)
with c1:
    left_label = st.selectbox("Left lab", labels, index=0 if labels else None)
with c2:
    right_label = st.selectbox("Right lab", labels, index=(1 if len(labels) > 1 else 0) if labels else None)

if not labels:
    st.info("No labs available.")
    st.stop()

left_ror = str(name_to_ror[left_label])
right_ror = str(name_to_ror[right_label])

# Build field distributions from pubs for selected labs/years
pp = pubs.copy()
pp = pp[pp["year"].isin(years_sel)] if "year" in pp.columns else pp

# explode labs and join primary field id/name
el = explode_labs(pp)
if "primary_field_id" in pp.columns:
    el = el.merge(pp[["openalex_id", "primary_field_id"]], on="openalex_id", how="left")
    el["field_name"] = el["primary_field_id"].map(field_name_for_id)
else:
    el["field_name"] = "Unknown"

lf = el.groupby(["lab_ror", "field_name"], as_index=False).agg(count=("openalex_id", "nunique"))

left_df  = lf[lf["lab_ror"].eq(left_ror)].copy()
right_df = lf[lf["lab_ror"].eq(right_ror)].copy()

# percent within lab
if not left_df.empty:
    left_total = max(float(left_df["count"].sum()), 1.0)
    left_df["pct"] = left_df["count"] / left_total
if not right_df.empty:
    right_total = max(float(right_df["count"].sum()), 1.0)
    right_df["pct"] = right_df["count"] / right_total

catalogue = canonical_field_order()

pL, pR = st.columns(2, gap="large")
for side, title, df_lab in [(pL, left_label, left_df), (pR, right_label, right_df)]:
    with side:
        st.markdown(f"### {title}")
        if df_lab.empty:
            st.info("No publications for this lab in the selected period.")
            continue

        st.markdown("**Field distribution (volume)**")
        st.altair_chart(
            field_mix_bars(
                df_lab.rename(columns={"count": "value"}), value_col="value", percent=False,
                enforce_order=catalogue or None, width=560
            ),
            use_container_width=True,
        )

        st.markdown("**Field distribution (% of lab works)**")
        st.altair_chart(
            field_mix_bars(
                df_lab.rename(columns={"pct": "value"}), value_col="value", percent=True,
                enforce_order=catalogue or None, width=560
            ),
            use_container_width=True,
        )

# ---------------------------------------------------------------------
# Collaboration between selected labs (from pubs)
# ---------------------------------------------------------------------
st.divider()
st.subheader("Collaboration between the selected labs")

# publications that include BOTH labs in the selected years
elabs = explode_labs(pp)
e2 = elabs[elabs["lab_ror"].isin([left_ror, right_ror])]
both_works = e2.groupby("openalex_id")["lab_ror"].nunique().reset_index(name="n_labs")
both_ids = set(both_works.loc[both_works["n_labs"] == 2, "openalex_id"]) if not both_works.empty else set()

copubs = pp[pp["openalex_id"].isin(both_ids)].copy()

if copubs.empty:
    st.info("No co-publications between these labs for the selected years.")
else:
    # KPIs
    total = int(copubs["openalex_id"].nunique())
    lue_count = int(pd.to_numeric(copubs.get("in_lue"), errors="coerce").fillna(0).astype(bool).sum()) if "in_lue" in copubs else 0
    lue_pct = (lue_count / total) if total else 0.0

    avg_fwci = float(pd.to_numeric(copubs.get("fwci_fr"), errors="coerce").mean() or 0.0)

    def _has_non_fr(c):
        toks = [t.strip().upper() for t in clean_pipe_list(c)]
        return any(t and t != "FR" for t in toks)

    def _has_company(t):
        toks = [t.strip().lower() for t in clean_pipe_list(t)]
        return any("company" in t for t in toks)

    intl = int(copubs.get("inst_countries", pd.Series([None]*len(copubs))).map(_has_non_fr).sum()) if "inst_countries" in copubs else 0
    intl_pct = (intl / total) if total else 0.0
    comp = int(copubs.get("inst_types", pd.Series([None]*len(copubs))).map(_has_company).sum()) if "inst_types" in copubs else 0
    comp_pct = (comp / total) if total else 0.0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Co-publications", f"{total:,}")
    k2.metric("ISITE (count / %)", f"{lue_count:,} / {lue_pct*100:.1f}%")
    k3.metric("Avg. FWCI (FR)", f"{avg_fwci:.2f}")
    k4.metric("% international", f"{intl_pct*100:.1f}%")
    k5.metric("% with company", f"{comp_pct*100:.1f}%")

    # Stacked vertical bars: document types by year
    cop = copubs.copy()
    type_col = "pub_type" if "pub_type" in cop.columns else None
    if type_col is not None:
        cop["doc_type"] = cop[type_col].str.lower().map({
            "article": "article",
            "review": "review",
            "book-chapter": "book-chapter",
            "chapter": "book-chapter",
            "book": "book",
        }).fillna("other")
    else:
        cop["doc_type"] = "other"

    doc_order = ["article", "review", "book-chapter", "book", "other"]
    year_counts = cop.groupby(["year", "doc_type"], dropna=False)["openalex_id"].nunique().reset_index(name="count")

    chart = (
        alt.Chart(year_counts)
        .mark_bar()
        .encode(
            x=alt.X("year:O", title="Year", sort=sorted(years_sel)),
            y=alt.Y("count:Q", title="Co-publications"),
            color=alt.Color("doc_type:N", title="Type", sort=doc_order),
            tooltip=[alt.Tooltip("year:O"), alt.Tooltip("doc_type:N"), alt.Tooltip("count:Q", format=",")],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)

    # ---------- TOP AUTHORS (co-pubs) + enrich ----------
    rows = []
    need = ["openalex_id", "authors", "authors_id", "fwci_fr"]
    for col in need:
        if col not in copubs.columns:
            copubs[col] = None

    for _, r in copubs[need].iterrows():
        names = clean_pipe_list(r["authors"]) or [""] * len(clean_pipe_list(r["authors_id"]))
        ids   = clean_pipe_list(r["authors_id"]) or [""] * len(names)
        # pad to match lengths
        if len(names) < len(ids):
            names += [""] * (len(ids) - len(names))
        if len(ids) < len(names):
            ids += [""] * (len(names) - len(ids))
        for nm, aid in zip(names, ids):
            if not aid and not nm:
                continue
            rows.append({"author_id": aid, "Author": nm, "openalex_id": r["openalex_id"], "fwci_fr": r.get("fwci_fr")})
    ea = pd.DataFrame(rows)

    top_counts = (
        ea.groupby(["author_id", "Author"], as_index=False)
          .agg(Publications=("openalex_id", "nunique"))
          .sort_values("Publications", ascending=False)
    )

    # enrich from author indicators
    g = authors_idx.rename(columns={
        "author_id": "author_id",
        "author_name": "Author",
        "pubs_unique": "Total publications",
        "avg_fwci_fr": "Avg. FWCI (overall)",
        "is_lorraine": "Is Lorraine",
    }) if not authors_idx.empty else pd.DataFrame(columns=["author_id"])  # noqa: E501

    top_authors = top_counts.merge(g, on=["author_id", "Author"], how="left") if not g.empty else top_counts
    top_authors = top_authors.sort_values(["Publications", "Avg. FWCI (overall)"], ascending=[False, False]).head(25)

    st.markdown("**Top authors in these co-publications**")
    st.dataframe(
        top_authors[[c for c in ("Author", "author_id", "Total publications", "Avg. FWCI (overall)", "Is Lorraine", "Publications") if c in top_authors.columns]],
        use_container_width=True, hide_index=True,
        column_config={
            "author_id": st.column_config.TextColumn("Author ID"),
            "Publications": st.column_config.NumberColumn(format="%.0f"),
            "Total publications": st.column_config.NumberColumn(format="%.0f"),
            "Avg. FWCI (overall)": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    # ---------- ALL CO-PUBLICATIONS (exportable) ----------
    want = [
        ("openalex_id", "OpenAlex ID"),
        ("doi", "DOI"),
        ("pub_type", "Publication Type"),
        ("year", "Publication Year"),
        ("title", "Title"),
        ("citation_count", "Citation Count"),
        ("fwci_fr", "FWCI_FR"),
        ("in_lue", "In LUE"),
        ("primary_topic_id", "Primary Topic ID"),
        ("primary_subfield_id", "Primary Subfield ID"),
        ("primary_field_id", "Primary Field ID"),
        ("primary_domain_id", "Primary Domain ID"),
    ]
    cols = [c for c, _ in want if c in copubs.columns]
    rename = {c: new for c, new in want if c in copubs.columns}
    copub_table = copubs[cols].rename(columns=rename).drop_duplicates()

    # add readable names for field/domain if present via taxonomy
    if taxonomy is not None and {"Primary Field ID"}.issubset(copub_table.columns):
        copub_table["Primary Field Name"] = copub_table["Primary Field ID"].map(field_name_for_id)

    st.markdown("**All co-publications (exportable)**")
    st.dataframe(copub_table, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV",
        data=copub_table.to_csv(index=False).encode("utf-8"),
        file_name=f"copubs_{left_label}_{right_label}_{year_min}-{year_max}.csv",
        mime="text/csv",
    )

st.divider()
st.markdown("Use the sidebar to return to **Home** and switch dashboards.")
