# pages/2_🔬_Topic_View.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import altair as alt
import pandas as pd
import streamlit as st

# ---- the ONLY shared import you need ----
from lib.taxonomy import (
    build_taxonomy_lookups,
    get_domain_color,
    get_field_color,
)

# ------------------------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------------------------
st.set_page_config(page_title="🔬 Topic View", layout="wide")
st.title("🔬 Topic View")
st.caption(
    "Colors follow domain palette; labels are always visible (charts auto-grow); "
    "count labels appear to the left of % bars; FWCI whiskers use min/Q1/median/Q3/max on a log x-axis."
)

# ------------------------------------------------------------------------------------
# Paths / loaders (local, no other shared modules)
# ------------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]  # repo root (one up from /pages)
DATA_DIR = REPO_ROOT / "data"

@st.cache_data(show_spinner=False)
def _load(name: str) -> pd.DataFrame:
    """Load a parquet from /data (by name or filename)."""
    path = DATA_DIR / (name if name.endswith(".parquet") else f"{name}.parquet")
    return pd.read_parquet(path)

domains = _load("ul_domains_indicators")      # domain-level indicators
fields  = _load("ul_fields_indicators")       # field-level indicators
partners = _load("ul_partners_indicators")    # partners indicators
authors = _load("ul_authors_indicators")      # author-level indicators
topics = _load("all_topics")                  # taxonomy (ids & names)

lookups = build_taxonomy_lookups()            # domain/field/subfield ordering + id<->name
domain_order = (
    domains.sort_values("Domain ID")["Domain name"].tolist()
    if "Domain ID" in domains.columns else lookups["domain_order"]
)

# ------------------------------------------------------------------------------------
# Small helpers (local to this page)
# ------------------------------------------------------------------------------------
def _pct01_to_pct100(x) -> float:
    try:
        return float(x) * 100.0
    except Exception:
        return float("nan")

def _explode_listlike(cell: str, sep="|") -> List[str]:
    if pd.isna(cell):
        return []
    return [c.strip() for c in str(cell).replace(";", sep).split(sep) if c.strip()]

def _parse_pairs(cell, id_type=str, val_type=float) -> pd.DataFrame:
    """
    Parse a single cell containing 'id (value) | id (value) | ...'
    → DataFrame[id, value]
    Robust to extra spaces/commas.
    """
    import re
    pat = re.compile(r"\s*([^\|]+?)\s*\(\s*([^)]+?)\s*\)\s*")
    out = []
    if pd.isna(cell):
        return pd.DataFrame(columns=["id", "value"])
    for chunk in str(cell).split("|"):
        m = pat.search(chunk)
        if not m:
            continue
        i, v = m.group(1).strip(), m.group(2).strip()
        try:
            v = val_type(str(v).replace(",", "").replace(" ", ""))
        except Exception:
            v = math.nan
        try:
            i = id_type(i)
        except Exception:
            i = str(i)
        out.append({"id": i, "value": v})
    return pd.DataFrame(out)

def _progress_cols(df: pd.DataFrame, cols: List[str]) -> Dict[str, "object"]:
    """Build Streamlit ProgressColumn config for the given % columns (0..100)."""
    try:
        import streamlit as st  # local import for type
    except Exception:
        return {}
    cfg: Dict[str, "object"] = {}
    for c in cols:
        if c not in df.columns:
            continue
        vmax = float(pd.to_numeric(df[c], errors="coerce").max())
        vmax = vmax if vmax > 0 else 100.0
        cfg[c] = st.column_config.ProgressColumn(
            c, format="%0.1f%%", min_value=0.0, max_value=vmax
        )
    return cfg

# ---- Altair helpers (count gutter + bars + whiskers) ------------------------------
_LEFT_GUTTER_PX = 80
_BAR_WIDTH_PX   = 720

def _dynamic_height(n: int) -> int:
    return int(max(1, n) * 48 + 80)

def _apply_color(df: pd.DataFrame, color_series: pd.Series) -> pd.DataFrame:
    b = df.copy()
    b["__color__"] = color_series
    return b

def bar_with_counts(df: pd.DataFrame, label_col: str, value_col: str, count_col: str,
                    color_series: Optional[pd.Series], title: str, order: Optional[List[str]] = None) -> alt.Chart:
    base = df.copy()
    base[label_col] = base[label_col].astype(str)
    base[value_col] = pd.to_numeric(base[value_col], errors="coerce").fillna(0.0)
    base[count_col] = pd.to_numeric(base[count_col], errors="coerce").fillna(0)

    if order:
        base[label_col] = pd.Categorical(base[label_col], categories=[x for x in order if x in base[label_col].tolist()], ordered=True)
        base = base.sort_values(label_col)

    if color_series is None:
        base["__color__"] = "#7f7f7f"
    else:
        base = _apply_color(base, color_series)

    height = _dynamic_height(len(base))

    left = alt.Chart(base, width=_LEFT_GUTTER_PX, height=height).mark_text(align="right", dx=-6).encode(
        y=alt.Y(f"{label_col}:N", sort=None, title=""),
        text=alt.Text(f"{count_col}:Q", format=",.0f"),
        color=alt.value("#444"),
        tooltip=[count_col],
    )

    bars = alt.Chart(base, width=_BAR_WIDTH_PX, height=height).mark_bar().encode(
        y=alt.Y(f"{label_col}:N", sort=None, title=""),
        x=alt.X(f"{value_col}:Q", title=title),
        color=alt.Color("__color__:N", scale=None, legend=None),
        tooltip=list(base.columns),
    )
    return left | bars

def whisker(df: pd.DataFrame, label_col: str, qmin: str, q1: str, q2: str, q3: str, qmax: str,
            color_series: Optional[pd.Series], title: str, order: Optional[List[str]] = None, log_x: bool = True) -> alt.Chart:
    base = df.copy()
    base[label_col] = base[label_col].astype(str)
    for c in (qmin, q1, q2, q3, qmax):
        base[c] = pd.to_numeric(base[c], errors="coerce")

    if order:
        base[label_col] = pd.Categorical(base[label_col], categories=[x for x in order if x in base[label_col].tolist()], ordered=True)
        base = base.sort_values(label_col)

    if color_series is None:
        base["__color__"] = "#7f7f7f"
    else:
        base = _apply_color(base, color_series)

    if log_x:
        eps = 1e-3
        for c in (qmin, q1, q2, q3, qmax):
            base[c] = base[c].where(base[c] > eps, eps)

    height = _dynamic_height(len(base))
    xscale = alt.Scale(type="log") if log_x else alt.Scale()

    rules = alt.Chart(base, width=_BAR_WIDTH_PX, height=height).mark_rule().encode(
        y=alt.Y(f"{label_col}:N", sort=None, title=""),
        x=alt.X(f"{qmin}:Q", title=title, scale=xscale),
        x2=f"{qmax}:Q",
        color=alt.Color("__color__:N", scale=None, legend=None),
        tooltip=list(base.columns),
    )
    boxes = alt.Chart(base, width=_BAR_WIDTH_PX, height=height).mark_bar(opacity=0.35).encode(
        y=alt.Y(f"{label_col}:N", sort=None, title=""),
        x=alt.X(f"{q1}:Q", title=title, scale=xscale),
        x2=f"{q3}:Q",
        color=alt.Color("__color__:N", scale=None, legend=None),
    )
    med = alt.Chart(base, width=_BAR_WIDTH_PX, height=height).mark_tick(size=18, thickness=2).encode(
        y=alt.Y(f"{label_col}:N", sort=None, title=""),
        x=f"{q2}:Q",
        color=alt.Color("__color__:N", scale=None, legend=None),
    )
    return rules + boxes + med

# ------------------------------------------------------------------------------------
# 1) Domain overview table
#    Visible columns only: Domain, Publications, % UL, Pubs LUE, % Pubs LUE, % PPtop10%, % PPtop1%, % internal collaboration, % international
#    Hidden-by-default: Avg. FWCI (France), % Pubs LUE (uni level), % PPtop10% (uni level), % PPtop1% (uni level), See in OpenAlex
# ------------------------------------------------------------------------------------
st.subheader("Domain overview")

dom_tbl = pd.DataFrame({
    "Domain": domains["Domain name"],
    "Publications": pd.to_numeric(domains["Pubs"], errors="coerce"),
    "% UL": pd.to_numeric(domains["% Pubs (uni level)"], errors="coerce").apply(_pct01_to_pct100),
    "Pubs LUE": pd.to_numeric(domains.get("Pubs LUE", pd.Series([math.nan]*len(domains))), errors="coerce"),
    "% Pubs LUE": pd.to_numeric(domains["% Pubs LUE (domain level)"], errors="coerce").apply(_pct01_to_pct100),
    "% PPtop10%": pd.to_numeric(domains["% PPtop10% (domain level)"], errors="coerce").apply(_pct01_to_pct100),
    "% PPtop1%": pd.to_numeric(domains["% PPtop1% (domain level)"], errors="coerce").apply(_pct01_to_pct100),
    "% internal collaboration": pd.to_numeric(domains["% internal collaboration"], errors="coerce").apply(_pct01_to_pct100),
    "% international": pd.to_numeric(domains["% international"], errors="coerce").apply(_pct01_to_pct100),
    # Hidden by default
    "Avg. FWCI (France)": pd.to_numeric(domains.get("Avg FWCI (France)", pd.Series([math.nan]*len(domains))), errors="coerce"),
    "% Pubs LUE (uni level)": pd.to_numeric(domains.get("% Pubs LUE (uni level)", pd.Series([math.nan]*len(domains))), errors="coerce").apply(_pct01_to_pct100),
    "% PPtop10% (uni level)": pd.to_numeric(domains.get("% PPtop10% (uni level)", pd.Series([math.nan]*len(domains))), errors="coerce").apply(_pct01_to_pct100),
    "% PPtop1% (uni level)": pd.to_numeric(domains.get("% PPtop1% (uni level)", pd.Series([math.nan]*len(domains))), errors="coerce").apply(_pct01_to_pct100),
    "See in OpenAlex": domains.get("See in OpenAlex", ""),
})

# sort by Domain ID ascending (if present)
if "Domain ID" in domains.columns:
    order_names = domains.sort_values("Domain ID")["Domain name"].tolist()
else:
    order_names = domain_order

dom_tbl = dom_tbl.set_index("Domain").loc[order_names].reset_index()

visible_cols = [
    "Domain", "Publications", "% UL", "Pubs LUE", "% Pubs LUE",
    "% PPtop10%", "% PPtop1%", "% internal collaboration", "% international",
]
all_cols = visible_cols + [
    "Avg. FWCI (France)", "% Pubs LUE (uni level)", "% PPtop10% (uni level)", "% PPtop1% (uni level)", "See in OpenAlex"
]

show_advanced = st.toggle("Show advanced columns", value=False)

cfg = {}
cfg.update(_progress_cols(dom_tbl, ["% UL", "% Pubs LUE", "% PPtop10%", "% PPtop1%", "% internal collaboration", "% international"]))
if show_advanced:
    cfg.update(_progress_cols(dom_tbl, ["% Pubs LUE (uni level)", "% PPtop10% (uni level)", "% PPtop1% (uni level)"]))

st.dataframe(
    dom_tbl[all_cols if show_advanced else visible_cols],
    use_container_width=True,
    hide_index=True,
    column_config=cfg,
)

# ------------------------------------------------------------------------------------
# 2) Domain FWCI whiskers (log scale)
# ------------------------------------------------------------------------------------
st.subheader("FWCI (France) distribution by domain")
qmin, q1, q2, q3, qmax = "FWCI_FR min", "FWCI_FR Q1", "FWCI_FR Q2", "FWCI_FR Q3", "FWCI_FR max"
color_series = dom_tbl["Domain"].map(lambda d: get_domain_color(d))
st.altair_chart(
    whisker(
        domains.rename(columns={"Domain name": "Domain"}).set_index("Domain").loc[order_names].reset_index(),
        label_col="Domain",
        qmin=qmin, q1=q1, q2=q2, q3=q3, qmax=qmax,
        color_series=color_series, title="FWCI (France) — min / Q1 / median / Q3 / max (log scale)",
        order=order_names, log_x=True
    ),
    use_container_width=True
)

# ------------------------------------------------------------------------------------
# 3) Drilldown by domain
# ------------------------------------------------------------------------------------
st.subheader("Drill down by domain")

sel_domain = st.selectbox("Pick a domain", options=order_names, index=0)
drow = domains.loc[domains["Domain name"] == sel_domain]
if drow.empty:
    st.warning("Selected domain not found in the data.")
    st.stop()
drow = drow.iloc[0]

# === Labs contribution (bars with counts) + the same labs whiskers
st.markdown("##### Labs contribution and impact")

labs_count = _parse_pairs(drow["By lab: count"], id_type=str, val_type=int).rename(columns={"id": "ROR", "value": "count"})
labs_share = _parse_pairs(drow["By lab: % of domain pubs"], id_type=str, val_type=float).rename(columns={"id": "ROR", "value": "share"})
labs = labs_count.merge(labs_share, on="ROR", how="outer").fillna(0.0)
labs["share_pct"] = labs["share"].apply(_pct01_to_pct100)

# Only labs >= 2% of domain
labs = labs[labs["share_pct"] >= 2.0].copy()

# Attach lab names
if not labs.empty:
    name_map = _load("ul_units_indicators").set_index("ROR")["Unit Name"].to_dict()
    labs["Lab"] = labs["ROR"].map(name_map).fillna(labs["ROR"])
    labs = labs.sort_values("share", ascending=False).reset_index(drop=True)
    lab_order = labs["Lab"].tolist()

    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(
            bar_with_counts(
                labs.rename(columns={"Lab": "Label"}),
                label_col="Label", value_col="share_pct", count_col="count",
                color_series=pd.Series([get_domain_color(sel_domain)] * len(labs)),
                title="% of domain publications", order=lab_order
            ),
            use_container_width=True
        )

    # Lab FWCI whiskers for same order (if provided)
    if all(k in drow for k in ["By lab: FWCI_FR min", "By lab: FWCI_FR Q1", "By lab: FWCI_FR Q2", "By lab: FWCI_FR Q3", "By lab: FWCI_FR max"]):
        def _lab_q(col):  # returns Series indexed by ROR
            dfq = _parse_pairs(drow[col], id_type=str, val_type=float).rename(columns={"id": "ROR", "value": col})
            return dfq.set_index("ROR")[col]
        qdf = pd.DataFrame({"ROR": labs["ROR"]})
        for col in ["By lab: FWCI_FR min", "By lab: FWCI_FR Q1", "By lab: FWCI_FR Q2", "By lab: FWCI_FR Q3", "By lab: FWCI_FR max"]:
            qdf = qdf.join(_lab_q(col), on="ROR")
        qdf = qdf.join(labs[["ROR", "Lab"]].set_index("ROR"), on="ROR").reset_index(drop=True)
        with c2:
            st.altair_chart(
                whisker(
                    qdf.rename(columns={"Lab": "Label",
                                        "By lab: FWCI_FR min": "min",
                                        "By lab: FWCI_FR Q1": "q1",
                                        "By lab: FWCI_FR Q2": "q2",
                                        "By lab: FWCI_FR Q3": "q3",
                                        "By lab: FWCI_FR max": "max"}),
                    label_col="Label", qmin="min", q1="q1", q2="q2", q3="q3", qmax="max",
                    color_series=pd.Series([get_domain_color(sel_domain)] * len(qdf)),
                    title="FWCI (France) by lab (log scale)", order=lab_order, log_x=True
                ),
                use_container_width=True
            )
    else:
        with c2:
            st.info("FWCI whiskers by lab unavailable for this domain.")
else:
    st.info("No labs reach the 2% threshold for this domain.")

# === Top partners — FR then International (NOT side-by-side)
st.markdown("##### Top partners")

# French partners
st.markdown("**Top 20 French partners (no parent institution)**")

fr_names = _explode_listlike(drow["Top 20 FR partners (name)"])
fr_types = _explode_listlike(drow["Top 20 FR partners (type)"])
fr_copubs = [int(x) if x else 0 for x in _explode_listlike(drow["Top 20 FR partners (totals copubs in this domain)"])]
# Correct percentage source:
fr_pct_ul = [float(x) * 100.0 for x in _explode_listlike(drow["Top 20 FR partners (% of UL total copubs)"])]

fr_df = pd.DataFrame({
    "Partner": fr_names,
    "Type": fr_types,
    "Copubs in this domain": fr_copubs,
    # A clear header name (progress bar):
    "Share of UL–partner copubs for this domain": fr_pct_ul,
})
st.dataframe(
    fr_df,
    use_container_width=True,
    hide_index=True,
    column_config=_progress_cols(fr_df, ["Share of UL–partner copubs for this domain"]),
)

# International partners
st.markdown("**Top 20 international partners**")

int_names = _explode_listlike(drow["Top 20 int partners (name)"])
int_types = _explode_listlike(drow["Top 20 int partners (type)"])
int_countries = _explode_listlike(drow["Top 20 int partners (country)"])
int_copubs = [int(x) if x else 0 for x in _explode_listlike(drow["Top 20 int partners (totals copubs in this domain)"])]
int_pct_ul = [float(x) * 100.0 for x in _explode_listlike(drow["Top 20 int partners (% of UL total copubs)"])]

int_df = pd.DataFrame({
    "Partner": int_names,
    "Type": int_types,
    "Country": int_countries,
    "Copubs in this domain": int_copubs,
    "Share of UL–partner copubs for this domain": int_pct_ul,
})
st.dataframe(
    int_df,
    use_container_width=True,
    hide_index=True,
    column_config=_progress_cols(int_df, ["Share of UL–partner copubs for this domain"]),
)

# === Top 20 authors
st.markdown("##### Top 20 authors")

def _author_enrichment_table(drow: pd.Series) -> pd.DataFrame:
    names   = _explode_listlike(drow["Top 20 authors (name)"])
    pubs    = [int(x) if x else 0 for x in _explode_listlike(drow["Top 20 authors (pubs)"])]
    fwci    = [float(x) if x else float("nan") for x in _explode_listlike(drow["Top 20 authors (Average FWCI_FR)"])]
    top10   = [int(x) if x else 0 for x in _explode_listlike(drow["Top 20 authors (PPtop10% Count)"])]
    top1    = [int(x) if x else 0 for x in _explode_listlike(drow["Top 20 authors (PPtop1% Count)"])]
    lorraine= _explode_listlike(drow.get("Top 20 authors (Is Lorraine)", ""))
    orcids  = _explode_listlike(drow.get("Top 20 authors (Orcid)", ""))
    aids    = _explode_listlike(drow.get("Top 20 authors (ID)", ""))

    df = pd.DataFrame({
        "Author": names,
        "Pubs in this domain": pubs,
        "Avg. FWCI (France)": fwci,                        # renamed
        "PPtop10% Count": top10,
        "PPtop1% Count": top1,
        "Is Lorraine": lorraine,
        "ORCID": orcids,
        "Author ID": aids,
    })

    # Build key -> total pubs mapping from ul_authors_indicators
    a = authors.copy()
    a["ORCID_list"] = a["ORCID"].apply(lambda s: _explode_listlike(s, sep="|"))
    a["AuthorID_list"] = a["Author ID"].apply(lambda s: _explode_listlike(s, sep="|"))
    key_to_total: Dict[str, int] = {}
    for _, r in a.iterrows():
        total = int(pd.to_numeric(r["Publications (unique)"], errors="coerce"))
        for o in r["ORCID_list"]:
            if o and o not in key_to_total:
                key_to_total[o] = total
        for k in r["AuthorID_list"]:
            if k and k not in key_to_total:
                key_to_total[k] = total

    # Enrich df with Total pubs (ORCID preferred, then Author ID)
    total_list = []
    for i, row in df.iterrows():
        tot = None
        if row["ORCID"]:
            tot = key_to_total.get(row["ORCID"])
        if tot is None and row["Author ID"]:
            tot = key_to_total.get(row["Author ID"])
        total_list.append(tot if tot is not None else math.nan)
    df["Total pubs (at UL)"] = total_list

    # % Pubs in this domain
    df["% Pubs in this domain"] = (df["Pubs in this domain"] / pd.to_numeric(df["Total pubs (at UL)"], errors="coerce")) * 100.0
    return df

auth_df = _author_enrichment_table(drow)

# Order & show (hide ORCID/Author ID by default)
auth_basic_cols = ["Author", "Pubs in this domain", "Total pubs (at UL)", "% Pubs in this domain",
                   "Avg. FWCI (France)", "PPtop10% Count", "PPtop1% Count", "Is Lorraine"]
auth_advanced_cols = auth_basic_cols + ["ORCID", "Author ID"]

show_ids = st.toggle("Show ORCID and Author ID", value=False, key="authors_ids_toggle")
st.dataframe(
    auth_df[auth_advanced_cols if show_ids else auth_basic_cols],
    use_container_width=True,
    hide_index=True,
    column_config=_progress_cols(auth_df, ["% Pubs in this domain"]),
)

# === Field distribution within this domain
st.markdown("##### Thematic shape — fields within this domain")

fld_counts = _parse_pairs(drow["By field: count"], id_type=str, val_type=int).rename(columns={"id": "Field ID", "value": "count"})
fld_pct = _parse_pairs(drow["By field: % of domain pubs"], id_type=str, val_type=float).rename(columns={"id": "Field ID", "value": "pct"})
fld = fld_counts.merge(fld_pct, on="Field ID", how="outer").fillna(0.0)

# Map Field ID -> Field name
id_to_field = topics.drop_duplicates(["field_id", "field_name"]).set_index("field_id")["field_name"].to_dict()
fld["Field"] = fld["Field ID"].map(lambda x: id_to_field.get(int(str(x)), str(x)))
fld["pct"] = fld["pct"].apply(_pct01_to_pct100)

# Order fields within this domain using taxonomy lookups
canon_fields = [f for f in lookups["fields_by_domain"].get(sel_domain, []) if f in fld["Field"].tolist()]
st.altair_chart(
    bar_with_counts(
        fld.rename(columns={"Field": "Label"}),
        label_col="Label", value_col="pct", count_col="count",
        color_series=pd.Series([get_domain_color(sel_domain)] * len(fld)),
        title="% of domain publications by field", order=canon_fields
    ),
    use_container_width=True
)

# ------------------------------------------------------------------------------------
# 4) Compare subfields within a field — across institutions
# ------------------------------------------------------------------------------------
st.subheader("Compare subfield shape within a field")

# Field picker (26 fields)
all_fields_sorted = topics.drop_duplicates(["field_id", "field_name"]).sort_values("field_id")["field_name"].tolist()
sel_field = st.selectbox("Select a field", options=all_fields_sorted, index=0)

# Institution pickers (left defaults to UL)
cL, cR = st.columns(2)
with cL:
    inst_left = st.selectbox("Institution (left)", options=["Université de Lorraine"] + sorted(partners["Institution name"].unique()), index=0, key="inst_left")
with cR:
    default_right = "Université de Strasbourg"
    opts_right = ["Université de Lorraine"] + sorted(partners["Institution name"].unique())
    default_idx = opts_right.index(default_right) if default_right in opts_right else 0
    inst_right = st.selectbox("Institution (right)", options=opts_right, index=default_idx, key="inst_right")

mode = st.radio("Scale", options=["Relative to selected field", "Absolute (institution-level)"], index=0, horizontal=True)

# Helpers to build subfield distributions
def ul_subfields_for_field(field_name: str, as_relative: bool) -> pd.DataFrame:
    row = fields.loc[fields["Field name"] == field_name]
    if row.empty:
        return pd.DataFrame(columns=["Subfield", "count", "pct"])
    r = row.iloc[0]
    sub_counts = _parse_pairs(r["By subfield: count"], id_type=str, val_type=int).rename(columns={"id": "subfield_id", "value": "count"})
    sub_pct_field = _parse_pairs(r["By subfield: % of field pubs"], id_type=str, val_type=float).rename(columns={"id": "subfield_id", "value": "pct_rel"})
    df = sub_counts.merge(sub_pct_field, on="subfield_id", how="outer").fillna(0.0)

    sid_to_name = topics.drop_duplicates(["subfield_id", "subfield_name"]).set_index("subfield_id")["subfield_name"].to_dict()
    df["Subfield"] = df["subfield_id"].map(lambda x: sid_to_name.get(int(str(x)), str(x)))

    if as_relative:
        df["pct"] = df["pct_rel"].apply(_pct01_to_pct100)
    else:
        total_ul = int(pd.to_numeric(domains["Pubs"], errors="coerce").sum())
        df["pct"] = df["count"].astype(float) / max(total_ul, 1) * 100.0
    return df[["Subfield", "count", "pct"]]

def partner_subfields_for_field(partner_name: str, field_name: str, as_relative: bool) -> pd.DataFrame:
    prow = partners.loc[partners["Institution name"] == partner_name]
    if prow.empty:
        return pd.DataFrame(columns=["Subfield", "count", "pct"])
    pr = prow.iloc[0]
    sub_counts = _parse_pairs(pr["By subfield: count"], id_type=str, val_type=int).rename(columns={"id": "subfield_id", "value": "count"})

    # Keep only subfields that belong to selected field
    field_sid = topics.loc[topics["field_name"] == field_name, "subfield_id"].unique().tolist()
    sub_counts = sub_counts[sub_counts["subfield_id"].astype(int).isin([int(x) for x in field_sid])].copy()

    if as_relative:
        denom = max(int(sub_counts["count"].sum()), 1)
    else:
        denom = max(int(pd.to_numeric(pr["Copublications"], errors="coerce")), 1)
    sub_counts["pct"] = sub_counts["count"].astype(float) / denom * 100.0

    sid_to_name = topics.drop_duplicates(["subfield_id", "subfield_name"]).set_index("subfield_id")["subfield_name"].to_dict()
    sub_counts["Subfield"] = sub_counts["subfield_id"].map(lambda x: sid_to_name.get(int(str(x)), str(x)))
    return sub_counts[["Subfield", "count", "pct"]]

def _subfield_chart(df: pd.DataFrame, field_name: str) -> alt.Chart:
    df = df.copy()
    # Color by the field's parent domain color
    df["__color__"] = get_field_color(field_name)
    sf_order = [s for s in lookups["subfields_by_field"].get(field_name, []) if s in df["Subfield"].tolist()]
    return bar_with_counts(
        df.rename(columns={"Subfield": "Label"}),
        label_col="Label", value_col="pct", count_col="count",
        color_series=df["__color__"], title=f"{field_name} — subfield distribution (%)", order=sf_order
    )

left_df = ul_subfields_for_field(sel_field, as_relative=(mode == "Relative to selected field")) if inst_left == "Université de Lorraine" else partner_subfields_for_field(inst_left, sel_field, as_relative=(mode == "Relative to selected field"))
right_df = ul_subfields_for_field(sel_field, as_relative=(mode == "Relative to selected field")) if inst_right == "Université de Lorraine" else partner_subfields_for_field(inst_right, sel_field, as_relative=(mode == "Relative to selected field"))

cc1, cc2 = st.columns(2)
with cc1:
    st.subheader(inst_left)
    st.altair_chart(_subfield_chart(left_df, sel_field), use_container_width=True)
with cc2:
    st.subheader(inst_right)
    st.altair_chart(_subfield_chart(right_df, sel_field), use_container_width=True)

# Optional: tiny collaboration snapshot for selected field (from field indicators)
st.caption("Collaboration snapshot (UL, selected field)")
frow = fields.loc[fields["Field name"] == sel_field]
if not frow.empty:
    rr = frow.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("% internal collaboration", f"{_pct01_to_pct100(rr['% internal collaboration']):.1f}%")
    c2.metric("% international", f"{_pct01_to_pct100(rr['% international']):.1f}%")
    c3.metric("% industrial", f"{_pct01_to_pct100(rr['% industrial']):.1f}%")
else:
    st.info("Field-level collaboration indicators unavailable for this field.")
