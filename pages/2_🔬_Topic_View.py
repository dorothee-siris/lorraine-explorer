# pages/2_🔬_Thematic_View.py
from __future__ import annotations

import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# plotting
try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
except ModuleNotFoundError:
    st.error("matplotlib is not installed. Add it to requirements.txt or `pip install matplotlib`.")
    st.stop()

# Make lib/taxonomy.py importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "lib"))

from lib.taxonomy import (
    build_taxonomy_lookups,
    canonical_field_order,
    get_domain_color,
    get_field_color,
    get_subfield_color,
)

# ============================== constants ==============================

YEAR_START, YEAR_END = 2019, 2023

DOMAINS_PATH = REPO_ROOT / "data" / "ul_domains_indicators.parquet"
FIELDS_PATH  = REPO_ROOT / "data" / "ul_fields_indicators.parquet"
PUBS_PATH    = REPO_ROOT / "data" / "pubs_final.parquet"   # optional
UNITS_PATH   = REPO_ROOT / "data" / "ul_units_indicators.parquet"  # optional (for ROR -> lab name)

FIELD_PAIR_RE = re.compile(r"^\s*(.*?)\s*\(([^)]*)\)\s*$")

# ============================== page config ==============================

st.set_page_config(page_title="🔬 Thematic view · Domains", layout="wide")
st.title("🔬 Thematic view — Domains")

# ============================== caching & loaders ==============================

@st.cache_data(show_spinner=False)
def get_lookups() -> Dict:
    return build_taxonomy_lookups()

@st.cache_data(show_spinner=False)
def load_domains() -> pd.DataFrame:
    df = pd.read_parquet(DOMAINS_PATH)
    # normalize simple casing/spacing
    df = df.rename(columns={c: c.strip() for c in df.columns})
    return df

@st.cache_data(show_spinner=False)
def load_fields() -> pd.DataFrame:
    df = pd.read_parquet(FIELDS_PATH)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    return df

@st.cache_data(show_spinner=False)
def load_pubs() -> Optional[pd.DataFrame]:
    try:
        df = pd.read_parquet(PUBS_PATH)
        return df
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_units_optional() -> Optional[pd.DataFrame]:
    try:
        df = pd.read_parquet(UNITS_PATH)
        df = df.rename(columns={c: c.strip() for c in df.columns})
        return df
    except Exception:
        return None

look = get_lookups()
df_domains = load_domains()
df_fields  = load_fields()
df_pubs    = load_pubs()          # not strictly required here
df_units   = load_units_optional() # optional: used for ROR->lab name

# ============================== tiny utils ==============================

def _to_int_safe(s) -> int:
    try:
        m = re.match(r"^\s*([0-9]+)", str(s))
        return int(m.group(1)) if m else 0
    except Exception:
        return 0

def _to_float_safe(s) -> float:
    try:
        return float(str(s).replace(",", "."))
    except Exception:
        return np.nan

def darken_hex(hex_color: str, factor: float = 0.65) -> str:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join([c*2 for c in h])
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    r, g, b = int(r*factor), int(g*factor), int(b*factor)
    r, g, b = max(0,r), max(0,g), max(0,b)
    return f"#{r:02x}{g:02x}{b:02x}"

def parse_pipe_number_list(blob: str) -> List[int]:
    if pd.isna(blob) or not str(blob).strip():
        return []
    return [_to_int_safe(x) for x in str(blob).split("|")]

def parse_id_count_blob(blob: str) -> pd.DataFrame:
    """'id_or_name (count) | id (count)' -> DataFrame[id,count]"""
    if pd.isna(blob) or not str(blob).strip():
        return pd.DataFrame(columns=["id","count"])
    rows = []
    for p in str(blob).split("|"):
        m = FIELD_PAIR_RE.match(p.strip())
        if not m:
            continue
        tok = m.group(1).strip()
        cnt = _to_int_safe(m.group(2))
        rows.append((tok, cnt))
    df = pd.DataFrame(rows, columns=["id","count"])
    # aggregate if duplicates
    return df.groupby("id", as_index=False)["count"].sum().sort_values("count", ascending=False)

def parse_id_value_blob(blob: str) -> pd.DataFrame:
    """'id_or_name (value) | id (value)' -> DataFrame[id,value(float)]"""
    if pd.isna(blob) or not str(blob).strip():
        return pd.DataFrame(columns=["id","value"])
    rows = []
    for p in str(blob).split("|"):
        m = FIELD_PAIR_RE.match(p.strip())
        if not m:
            continue
        tok = m.group(1).strip()
        val = _to_float_safe(m.group(2))
        rows.append((tok, val))
    return pd.DataFrame(rows, columns=["id","value"])

def parse_parallel_lists(names_blob: str, *more_blobs: str) -> pd.DataFrame:
    def split_clean(s: str) -> List[str]:
        if pd.isna(s) or not str(s).strip():
            return []
        return [x.strip() for x in str(s).split("|")]
    cols = [split_clean(names_blob)]
    for b in more_blobs:
        cols.append(split_clean(b))
    n = max((len(c) for c in cols), default=0)
    cols = [c + [""]*(n-len(c)) for c in cols]
    df = pd.DataFrame({"name": cols[0]})
    if len(cols) > 1: df["type"] = cols[1]
    if len(cols) > 2: df["country"] = cols[2]
    if len(cols) > 3: df["count"] = [ _to_int_safe(x) for x in cols[3] ]
    if len(cols) > 4: df["pct_ul"] = [ _to_float_safe(x) for x in cols[4] ]
    return df

def pad_table_rows(df: pd.DataFrame, n_rows: int, numeric_cols: Optional[List[str]] = None) -> pd.DataFrame:
    d = df.copy()
    if len(d) >= n_rows:
        return d.head(n_rows)
    missing = n_rows - len(d)
    filler = {}
    for col in d.columns:
        if numeric_cols and col in numeric_cols:
            filler[col] = np.nan
        else:
            filler[col] = ""
    filler_df = pd.DataFrame([filler] * missing)
    return pd.concat([d, filler_df], ignore_index=True)

def ror_to_lab_name_map() -> Dict[str,str]:
    if df_units is None:
        return {}
    cols = {"ROR","Unit Name"}
    if not cols.issubset(df_units.columns):
        return {}
    ref = df_units[["ROR","Unit Name"]].drop_duplicates()
    return dict(zip(ref["ROR"].astype(str), ref["Unit Name"].astype(str)))

# ============================== plotting (Matplotlib, Lab View style) ==============================

def plot_year_counts_bar(values: List[int], title: str, ymax: int | None = None) -> plt.Figure:
    years = [YEAR_START + i for i in range(len(values))]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    ax.bar(years, values, width=0.6, color="#8190FF", edgecolor="none", alpha=0.9)
    ax.set_title(title, fontsize=12, pad=6)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], fontsize=10)
    ax.set_ylabel("Publications", fontsize=11)
    if ymax is not None and ymax > 0:
        ax.set_ylim(0, ymax * 1.05)
    ax.grid(axis="y", color="#eeeeee")
    for spine in ("top","right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return fig

def plot_barh_with_gutter(df: pd.DataFrame,
                          order_labels: List[str],
                          value_col: str,
                          count_col: Optional[str],
                          title: str,
                          color_col: Optional[str] = None,
                          base_color: Optional[str] = None,
                          overlay_value_col: Optional[str] = None,
                          overlay_darkening: float = 0.65,
                          x_label: str = "% of domain publications",
                          share_max: Optional[float] = None,
                          bar_height: float = 0.8) -> plt.Figure:
    """
    Generic left-gutter barh with optional overlay (e.g., LUE share).
    Expects df to have columns: Label, value_col, (optional) count_col, (optional) color_col
    """
    base = pd.DataFrame({"Label": order_labels})
    d = base.merge(df, on="Label", how="left")

    # defaults
    if count_col and count_col not in d.columns:
        d[count_col] = 0
    if value_col not in d.columns:
        d[value_col] = 0.0
    if overlay_value_col and overlay_value_col not in d.columns:
        d[overlay_value_col] = 0.0

    if color_col and color_col in d.columns:
        colors = d[color_col].fillna("#7f7f7f").tolist()
    else:
        colors = [base_color or "#7f7f7f"] * len(d)

    y = np.arange(len(d))
    vmax = float(share_max if share_max is not None else max(0.05, float(d[value_col].max() or 0.0)))
    fig_h = max(1.0, 0.42 * len(d) + 0.8)
    fig, ax = plt.subplots(figsize=(7.2, fig_h))

    # left gutter
    left_pad_px, offset_px = (72, 6)
    ax.set_xlim(0, vmax)
    fig.canvas.draw()
    bb = ax.get_window_extent(renderer=fig.canvas.get_renderer())
    ax_width_px = bb.width if bb.width > 0 else 600
    data_per_px = (vmax - 0.0) / ax_width_px if ax_width_px else 0.0001
    left_pad_data = left_pad_px * data_per_px
    offset_data   = offset_px * data_per_px
    ax.set_xlim(-left_pad_data, vmax)

    # bars
    for i, row in d.iterrows():
        ax.barh(y[i], width=float(row[value_col] or 0.0), left=0.0, height=bar_height,
                edgecolor="none", color=colors[i], alpha=0.95, zorder=2)
        if overlay_value_col:
            ov = float(row[overlay_value_col] or 0.0)
            if ov > 0:
                ax.barh(y[i], width=ov, left=0.0, height=bar_height*0.7,
                        edgecolor="none", color=darken_hex(colors[i], overlay_darkening), alpha=1.0, zorder=3)

    # gutter counts
    if count_col:
        for yi, cnt in enumerate(pd.to_numeric(d[count_col], errors="coerce").fillna(0).astype(int).tolist()):
            ax.text(-left_pad_data + offset_data, yi, f"{cnt:,}".replace(",", " "),
                    va="center", ha="left", fontsize=9, color="#444")

    # axes
    ax.set_title(title, fontsize=12, pad=6)
    ax.set_yticks(y)
    ax.set_yticklabels(d["Label"], fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#eeeeee")
    ax.set_axisbelow(True)

    # xticks every 5%
    max_tick = np.ceil(vmax * 20) / 20.0
    xticks = np.arange(0.0, max(0.05, max_tick) + 1e-9, 0.05)
    xticks = xticks[xticks <= vmax + 1e-9]
    ax.set_xticks(xticks)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_xlim(-left_pad_data, vmax)
    ax.set_xticklabels([f"{int(x*100)}%" for x in xticks], fontsize=10)

    for spine in ("top","right","left"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    return fig

def plot_fwci_whiskers(df: pd.DataFrame,
                       order_labels: List[str],
                       qcols: Dict[str,str],  # keys among {'min','p5','q1','q2','q3','p95','max'}
                       xmax: float,
                       title: str,
                       count_col: Optional[str] = None) -> plt.Figure:
    base = pd.DataFrame({"Label": order_labels})
    d = base.merge(df, on="Label", how="left")

    y = np.arange(len(d))
    fig_h = max(1.0, 0.40 * len(d) + 0.8)
    fig, ax = plt.subplots(figsize=(7.2, fig_h))

    # left gutter
    left_pad_px, offset_px = (72, 6)
    ax.set_xlim(0, xmax)
    fig.canvas.draw()
    bb = ax.get_window_extent(renderer=fig.canvas.get_renderer())
    ax_width_px = bb.width if bb.width > 0 else 600
    data_per_px = (xmax - 0.0) / ax_width_px if ax_width_px else 0.0001
    left_pad_data = left_pad_px * data_per_px
    offset_data   = offset_px * data_per_px
    ax.set_xlim(-left_pad_data, xmax)

    # draw rows
    for i, r in d.iterrows():
        c = r.get("color", "#7f7f7f")
        # whisker span
        x_min = r.get(qcols.get("min","min"))
        x_max = r.get(qcols.get("max","max"))
        # Allow p5/p95 when min/max are absent
        x_min = x_min if pd.notna(x_min) else r.get(qcols.get("p5","p5"))
        x_max = x_max if pd.notna(x_max) else r.get(qcols.get("p95","p95"))

        q1 = r.get(qcols.get("q1","q1"))
        q2 = r.get(qcols.get("q2","q2"))
        q3 = r.get(qcols.get("q3","q3"))

        stats = [x for x in [x_min,q1,q2,q3,x_max] if pd.notna(x)]
        if len(stats) == 0:
            continue

        if pd.notna(x_min) and pd.notna(x_max):
            ax.hlines(y[i], xmin=float(x_min), xmax=float(x_max), color=c, linewidth=1.2, zorder=2)
        if pd.notna(q1) and pd.notna(q3) and float(q3) >= float(q1):
            ax.barh(y[i], width=float(q3)-float(q1), left=float(q1), height=0.5,
                    color=c, alpha=0.25, edgecolor="none", zorder=3)
        if pd.notna(q2):
            ax.vlines(float(q2), ymin=y[i]-0.25, ymax=y[i]+0.25, color=c, linewidth=2.0, zorder=4)

    # gutter counts
    if count_col and count_col in d.columns:
        for yi, cnt in enumerate(pd.to_numeric(d[count_col], errors="coerce").fillna(0).astype(int).tolist()):
            if cnt > 0:
                ax.text(-left_pad_data + offset_data, yi, f"{cnt:,}".replace(",", " "),
                        va="center", ha="left", fontsize=9, color="#444")

    # axes
    ax.set_title(title, fontsize=12, pad=6)
    ax.set_yticks(y)
    ax.set_yticklabels(d["Label"], fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#eeeeee")
    ax.set_axisbelow(True)
    ax.set_xlabel("FWCI (France)", fontsize=11)
    for spine in ("top","right","left"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return fig

# ============================== domain overview table ==============================

st.subheader("Domain overview (2019–2023)")

# Prepare overview table
def pct_to_100(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce") * 100.0

overview = pd.DataFrame({
    "Domain": df_domains.get("Domain name", pd.Series([""]*len(df_domains))),
    "Publications": pd.to_numeric(df_domains.get("Pubs"), errors="coerce"),
    "% UL pubs": pct_to_100(df_domains.get("% Pubs (uni level)", 0)),
    "Pubs LUE": pd.to_numeric(df_domains.get("Pubs LUE"), errors="coerce"),
    "% LUE (domain)": pct_to_100(df_domains.get("% Pubs LUE (domain level)", 0)),
    "% Top 10% (domain)": pct_to_100(df_domains.get("% PPtop10% (domain level)", 0)),
    "% Top 1% (domain)": pct_to_100(df_domains.get("% PPtop1% (domain level)", 0)),
    "% internal": pct_to_100(df_domains.get("% internal collaboration", 0)),
    "% international": pct_to_100(df_domains.get("% international", 0)),
    "Avg FWCI (FR)": pd.to_numeric(df_domains.get("Avg FWCI (France)"), errors="coerce"),
    "OpenAlex": df_domains.get("See in OpenAlex", ""),
})

# Canonical order of domains
present = overview["Domain"].astype(str).tolist()
order = [d for d in look.get("domain_order", []) if d in present]
if order:
    cat = pd.Categorical(overview["Domain"], categories=order, ordered=True)
    overview = overview.sort_values("Domain", key=lambda s: cat)

# progress maxes
max_ul   = float(overview["% UL pubs"].max() or 1.0)
max_lue  = float(overview["% LUE (domain)"].max() or 1.0)
max_t10  = float(overview["% Top 10% (domain)"].max() or 1.0)
max_t1   = float(overview["% Top 1% (domain)"].max() or 1.0)
max_int  = float(overview["% international"].max() or 1.0)
max_intl = float(overview["% internal"].max() or 1.0)

st.dataframe(
    overview,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Domain": st.column_config.TextColumn("Domain"),
        "Publications": st.column_config.NumberColumn("Publications", format="%.0f"),
        "% UL pubs": st.column_config.ProgressColumn("% Université de Lorraine", format="%.1f %%", min_value=0.0, max_value=max_ul),
        "Pubs LUE": st.column_config.NumberColumn("Pubs LUE", format="%.0f"),
        "% LUE (domain)": st.column_config.ProgressColumn("% of domain pubs LUE", format="%.1f %%", min_value=0.0, max_value=max_lue),
        "% Top 10% (domain)": st.column_config.ProgressColumn("% Top 10% (domain)", format="%.1f %%", min_value=0.0, max_value=max_t10),
        "% Top 1% (domain)": st.column_config.ProgressColumn("% Top 1% (domain)", format="%.1f %%", min_value=0.0, max_value=max_t1),
        "% internal": st.column_config.ProgressColumn("% internal", format="%.1f %%", min_value=0.0, max_value=max_intl),
        "% international": st.column_config.ProgressColumn("% international", format="%.1f %%", min_value=0.0, max_value=max_int),
        "Avg FWCI (FR)": st.column_config.NumberColumn("Avg FWCI (FR)", format="%.3f"),
        "OpenAlex": st.column_config.LinkColumn("See in OpenAlex"),
    },
)

st.divider()

# ============================== drill-down by domain ==============================

st.subheader("Drill-down by domain")
domains_list = overview["Domain"].dropna().astype(str).tolist()
if not domains_list:
    st.info("No domains found in ul_domains_indicators.parquet.")
    st.stop()

sel_domain = st.selectbox("Pick a domain", options=domains_list, index=0)
drow = df_domains.loc[df_domains["Domain name"] == sel_domain]
if drow.empty:
    st.warning("Selected domain not found in the domain indicators file.")
    st.stop()
drow = drow.iloc[0]

# --- KPIs row
k1, k2, k3, k4 = st.columns(4)
k1.metric("Pubs (2019–2023)", f"{_to_int_safe(drow.get('Pubs')):,}".replace(",", " "))
k2.metric("… incl. LUE", f"{_to_int_safe(drow.get('Pubs LUE')):,}".replace(",", " "))
k3.metric("… Top 10% (count)", f"{_to_int_safe(drow.get('PPtop10%')):,}".replace(",", " "))
k4.metric("… Top 1% (count)", f"{_to_int_safe(drow.get('PPtop1%')):,}".replace(",", " "))

# --- Yearly distribution
year_counts = parse_pipe_number_list(drow.get("Year distribution (2019-2023)", ""))
if year_counts:
    fig_years = plot_year_counts_bar(year_counts, "Yearly publications (totals)")
    st.pyplot(fig_years, use_container_width=True)
else:
    st.info("No yearly distribution data for this domain.")

st.markdown("---")

# ------------------------ Labs contribution + FWCI ------------------------

st.markdown("#### Labs contribution and impact")

# Labs share/counts
labs_cnt = parse_id_count_blob(drow.get("By lab: count", ""))
labs_shr = parse_id_value_blob(drow.get("By lab: % of domain pubs", ""))

labs = pd.merge(labs_cnt, labs_shr, on="id", how="outer").fillna({"count":0, "value":0.0})
labs["share"] = pd.to_numeric(labs["value"], errors="coerce").fillna(0.0)  # 0..1
labs = labs.drop(columns=["value"])

# Optional: map ROR -> lab name
name_map = ror_to_lab_name_map()
labs["Label"] = labs["id"].astype(str).map(name_map).fillna(labs["id"].astype(str))

# Filter: show labs >= 2% (adjustable)
min_share = st.slider("Show labs with share ≥", 0.0, 10.0, 2.0, 0.5, help="% of domain publications")
labs = labs[labs["share"]*100.0 >= min_share].copy()

# order & colors
labs = labs.sort_values("share", ascending=False).reset_index(drop=True)
lab_labels = labs["Label"].tolist()
dom_color = get_domain_color(sel_domain)
labs["color"] = dom_color

# paired charts
share_max = float(max(labs["share"].max() if not labs.empty else 0.0, 0.05))
left_fig = plot_barh_with_gutter(
    df=labs.assign(**{"%share": labs["share"]}),   # keep share in 0..1
    order_labels=lab_labels,
    value_col="%share",
    count_col="count",
    title="% of domain publications (lab)",
    color_col="color",
    x_label="% of domain publications",
    share_max=share_max,
)

# FWCI whiskers per lab (percentiles provided at domain level)
q_needed = ["By lab: FWCI_FR p5", "By lab: FWCI_FR Q1", "By lab: FWCI_FR Q2", "By lab: FWCI_FR Q3", "By lab: FWCI_FR p95"]
have_all_q = all(c in df_domains.columns for c in q_needed)
right_fig = None
if have_all_q and not labs.empty:
    # Build qdf aligned to labs by their id token
    def _lab_q(col: str) -> pd.Series:
        s = parse_id_value_blob(drow.get(col, "")).set_index("id")["value"]
        return s

    qdf = pd.DataFrame({
        "id": labs["id"].astype(str),
        "Label": labs["Label"].astype(str),
        "p5":  _lab_q("By lab: FWCI_FR p5").reindex(labs["id"].astype(str)).values,
        "q1":  _lab_q("By lab: FWCI_FR Q1").reindex(labs["id"].astype(str)).values,
        "q2":  _lab_q("By lab: FWCI_FR Q2").reindex(labs["id"].astype(str)).values,
        "q3":  _lab_q("By lab: FWCI_FR Q3").reindex(labs["id"].astype(str)).values,
        "p95": _lab_q("By lab: FWCI_FR p95").reindex(labs["id"].astype(str)).values,
        "color": dom_color,
        "count": labs["count"].values,
    })
    xmax = float(np.nanmax(qdf[["p95","q3","q2","q1"]].values)) if not qdf.empty else 1.0
    xmax = max(1.0, xmax)
    right_fig = plot_fwci_whiskers(
        df=qdf,
        order_labels=lab_labels,
        qcols={"p5":"p5","q1":"q1","q2":"q2","q3":"q3","p95":"p95"},
        xmax=xmax,
        title="FWCI (France) by lab",
        count_col="count",
    )

# render side-by-side
cA, cB = st.columns(2)
with cA:
    st.pyplot(left_fig, use_container_width=True)
with cB:
    if right_fig is not None:
        st.pyplot(right_fig, use_container_width=True)
    else:
        st.info("FWCI percentiles per lab not available in this file.")

st.markdown("---")

# ------------------------ Field composition inside domain ------------------------

st.markdown("#### Thematic shape — fields within this domain")

fld_cnt = parse_id_count_blob(drow.get("By field: count", ""))
fld_lue = parse_id_count_blob(drow.get("By field: LUE count", ""))
fld_pct = parse_id_value_blob(drow.get("By field: % of domain pubs", ""))

# Merge & derive
fields_df = pd.merge(fld_cnt, fld_lue, on="id", how="outer", suffixes=("_count","_lue")).merge(
    fld_pct, on="id", how="outer"
).fillna({"count_count":0, "count_lue":0, "value":0.0})

# id can be numeric id or name; resolve to name via taxonomy when possible
def field_token_to_name(tok: str) -> str:
    t = str(tok).strip()
    if t.isdigit():
        # try exact id->name
        return look["id2name"].get(t, t)
    # already a name
    return t

fields_df["Field"] = fields_df["id"].astype(str).apply(field_token_to_name)
fields_df["Label"] = fields_df["Field"]
fields_df["share"] = pd.to_numeric(fields_df["value"], errors="coerce").fillna(0.0)  # 0..1
fields_df["lue_share"] = (pd.to_numeric(fields_df["count_lue"], errors="coerce").fillna(0).astype(float) /
                          pd.to_numeric(drow.get("Pubs", 0), errors="coerce").clip(lower=1).astype(float))
# color by domain (all fields belong to sel_domain); but we keep field-specific color for consistency
fields_df["color"] = fields_df["Field"].apply(get_field_color)

# Canonical order restricted to fields present in this domain per taxonomy
canon_fields_for_domain = [f for f in look["fields_by_domain"].get(sel_domain, []) if f in fields_df["Field"].tolist()]
if not canon_fields_for_domain:
    # fallback: keep rows we have
    canon_fields_for_domain = fields_df["Field"].tolist()

share_max_fields = float(max(fields_df["share"].max() if not fields_df.empty else 0.0, 0.05))
fig_fields = plot_barh_with_gutter(
    df=fields_df.rename(columns={"count_count":"count"})[["Label","share","lue_share","count","color"]],
    order_labels=canon_fields_for_domain,
    value_col="share",
    count_col="count",
    title="% of domain publications by field (LUE overlay in darker tint)",
    color_col="color",
    overlay_value_col="lue_share",
    x_label="% of domain publications",
    share_max=share_max_fields,
)
st.pyplot(fig_fields, use_container_width=True)

st.markdown("---")

# ------------------------ Partners in this domain ------------------------

st.markdown("#### Top partners in this domain")

# FR partners
st.markdown("**Top 20 French partners**")
fr_df = parse_parallel_lists(
    drow.get("Top 20 FR partners (name)", ""),
    drow.get("Top 20 FR partners (type)", ""),
    "",  # no country column
    drow.get("Top 20 FR partners (totals copubs in this domain)", ""),
    drow.get("Top 20 FR partners (% of UL total copubs)", ""),
)
if fr_df.empty:
    st.info("No French partners listed for this domain.")
else:
    fr_df = fr_df.rename(columns={
        "name":"Partner", "type":"Type", "count":"Co-pubs in this domain", "pct_ul":"Share of UL–partner copubs (this domain)"
    })
    fr_df["Share of UL–partner copubs (this domain)"] = pd.to_numeric(fr_df["Share of UL–partner copubs (this domain)"], errors="coerce") * 100.0
    fr_df = pad_table_rows(fr_df, 20, numeric_cols=["Co-pubs in this domain","Share of UL–partner copubs (this domain)"])
    st.dataframe(
        fr_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Partner": st.column_config.TextColumn("Partner"),
            "Co-pubs in this domain": st.column_config.NumberColumn("Co-pubs in this domain"),
            "Share of UL–partner copubs (this domain)": st.column_config.ProgressColumn(
                "% of UL–partner copubs", format="%.1f %%", min_value=0.0, max_value=float(fr_df["Share of UL–partner copubs (this domain)"].max() or 100.0)
            ),
            "Type": st.column_config.TextColumn("Type"),
        },
    )

# International partners
st.markdown("**Top 20 international partners**")
int_df = parse_parallel_lists(
    drow.get("Top 20 int partners (name)", ""),
    drow.get("Top 20 int partners (type)", ""),
    drow.get("Top 20 int partners (country)", ""),
    drow.get("Top 20 int partners (totals copubs in this domain)", ""),
    drow.get("Top 20 int partners (% of UL total copubs)", ""),
)
if int_df.empty:
    st.info("No international partners listed for this domain.")
else:
    int_df = int_df.rename(columns={
        "name":"Partner", "type":"Type", "country":"Country",
        "count":"Co-pubs in this domain", "pct_ul":"Share of UL–partner copubs (this domain)"
    })
    int_df["Share of UL–partner copubs (this domain)"] = pd.to_numeric(int_df["Share of UL–partner copubs (this domain)"], errors="coerce") * 100.0
    int_df = pad_table_rows(int_df, 20, numeric_cols=["Co-pubs in this domain","Share of UL–partner copubs (this domain)"])
    st.dataframe(
        int_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Partner": st.column_config.TextColumn("Partner"),
            "Country": st.column_config.TextColumn("Country"),
            "Co-pubs in this domain": st.column_config.NumberColumn("Co-pubs in this domain"),
            "Share of UL–partner copubs (this domain)": st.column_config.ProgressColumn(
                "% of UL–partner copubs", format="%.1f %%", min_value=0.0, max_value=float(int_df["Share of UL–partner copubs (this domain)"].max() or 100.0)
            ),
            "Type": st.column_config.TextColumn("Type"),
        },
    )

st.markdown("---")

# ------------------------ Top authors in this domain ------------------------

st.markdown("#### Top 20 authors")

def _explode(cell: str) -> List[str]:
    if pd.isna(cell) or cell is None:
        return []
    return [c.strip() for c in str(cell).replace(";", "|").split("|") if c.strip()]

auth_names = _explode(drow.get("Top 20 authors (name)", ""))
if not auth_names:
    st.info("No authors listed for this domain.")
else:
    n = len(auth_names)
    def pad(lst, fill):
        return lst + [fill] * max(0, n - len(lst))

    pubs   = pad([_to_int_safe(x) for x in _explode(drow.get("Top 20 authors (pubs)", ""))], 0)
    fwci   = pad([_to_float_safe(x) for x in _explode(drow.get("Top 20 authors (Average FWCI_FR)", ""))], np.nan)
    top10  = pad([_to_int_safe(x) for x in _explode(drow.get("Top 20 authors (PPtop10% Count)", ""))], 0)
    top1   = pad([_to_int_safe(x) for x in _explode(drow.get("Top 20 authors (PPtop1% Count)", ""))], 0)
    isul   = pad([str(x).strip().lower()=="true" for x in _explode(drow.get("Top 20 authors (Is Lorraine)", ""))], False)
    orcid  = pad([x.split("|")[0].strip() if x else "" for x in _explode(drow.get("Top 20 authors (Orcid)", ""))], "")
    aid    = pad([x.split("|")[0].strip() if x else "" for x in _explode(drow.get("Top 20 authors (ID)", ""))], "")

    top_df = pd.DataFrame({
        "Author": auth_names,
        "Pubs in this domain": pubs,
        "Avg FWCI (FR)": fwci,
        "Top10 count": top10,
        "Top1 count": top1,
        "Is UL": ["Yes" if b else "No" for b in isul],
        "ORCID": orcid,
        "Author ID": aid,
    })

    # Optional enrichment: total pubs at UL via authors parquet (if available)
    total_map_by_id, total_map_by_orcid = {}, {}
    # If you have an authors parquet with "Author ID"/"ORCID"/"Publications (unique)", populate these maps here.

    top_df["Total pubs (UL)"] = pd.to_numeric(top_df["Author ID"].map(total_map_by_id), errors="coerce")
    missing = top_df["Total pubs (UL)"].isna()
    top_df.loc[missing, "Total pubs (UL)"] = pd.to_numeric(top_df.loc[missing, "ORCID"].map(total_map_by_orcid), errors="coerce")
    denom = top_df["Total pubs (UL)"].replace(0, pd.NA)
    top_df["% in this domain"] = ((top_df["Pubs in this domain"] / denom).fillna(0.0) * 100.0).clip(lower=0)

    show_ids = st.toggle("Show ORCID and Author ID", False, key="authors_ids_toggle")
    cols_basic = ["Author","Pubs in this domain","Total pubs (UL)","% in this domain","Avg FWCI (FR)","Top10 count","Top1 count","Is UL"]
    cols_all = cols_basic + ["ORCID","Author ID"]
    st.dataframe(
        top_df[cols_all if show_ids else cols_basic],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Pubs in this domain": st.column_config.NumberColumn("Pubs in this domain"),
            "Total pubs (UL)": st.column_config.NumberColumn("Total pubs (UL)"),
            "% in this domain": st.column_config.ProgressColumn("% in this domain", format="%.1f %%", min_value=0.0, max_value=float(top_df["% in this domain"].max() or 100.0)),
            "Avg FWCI (FR)": st.column_config.NumberColumn("Avg FWCI (FR)", format="%.3f"),
        },
    )

st.divider()

# ============================== UL vs Partner — field shape inside a domain ==============================

st.subheader("Compare field shape inside a domain (UL vs partner)")

# pick domain & partner
col1, col2 = st.columns(2)
with col1:
    cmp_domain = st.selectbox("Domain", options=domains_list, index=max(0, domains_list.index(sel_domain)))
with col2:
    partners_list = sorted(df_fields["Institution name"].dropna().astype(str).unique().tolist())
    default_partner = "Université de Strasbourg"
    p_index = partners_list.index(default_partner) if default_partner in partners_list else 0
    partner_name = st.selectbox("Partner", options=partners_list, index=p_index)

# UL side from ul_domains_indicators
ul_row = df_domains.loc[df_domains["Domain name"] == cmp_domain]
if ul_row.empty:
    st.info("Selected domain not found for UL in domain indicators.")
else:
    ul_row = ul_row.iloc[0]
    ul_fld_cnt = parse_id_count_blob(ul_row.get("By field: count", ""))
    ul_fld_pct = parse_id_value_blob(ul_row.get("By field: % of domain pubs", ""))
    ul_df = pd.merge(ul_fld_cnt, ul_fld_pct, on="id", how="outer").fillna({"count":0, "value":0.0})
    ul_df["Field"] = ul_df["id"].astype(str).apply(lambda t: look["id2name"].get(t, t) if t.isdigit() else t)
    ul_df["Label"] = ul_df["Field"]
    ul_df["pct"] = pd.to_numeric(ul_df["value"], errors="coerce").fillna(0.0)  # 0..1
    ul_df["color"] = ul_df["Field"].apply(get_field_color)

# Partner side from ul_fields_indicators
prow = df_fields.loc[df_fields["Institution name"] == partner_name]
if prow.empty:
    st.info("Selected partner not found in fields indicators.")
else:
    prow = prow.iloc[0]
    # domain total for this partner: parse "By domain: count" and pick this domain id
    dom_id = look["name2id"].get(cmp_domain)  # string id like "3"
    p_dom_counts = parse_id_count_blob(prow.get("By domain: count", ""))
    if p_dom_counts.empty or dom_id is None:
        st.info("Partner domain totals unavailable for comparison.")
    else:
        # keep id as numeric-ish or string; entries in file likely like '1 (1453)'
        # We'll match either id or domain name
        dom_total = 0
        # try id match
        m_id = p_dom_counts.loc[p_dom_counts["id"].astype(str) == str(dom_id)]
        if not m_id.empty:
            dom_total = int(m_id["count"].iloc[0])
        else:
            # try name match
            m_nm = p_dom_counts.loc[p_dom_counts["id"].astype(str) == cmp_domain]
            if not m_nm.empty:
                dom_total = int(m_nm["count"].iloc[0])

        if dom_total <= 0:
            st.info("Partner has zero publications in this domain for 2019–2023.")
        else:
            # partner field counts (all fields) -> keep only fields that belong to cmp_domain
            p_field_counts = parse_id_count_blob(prow.get("By field: count", ""))
            if p_field_counts.empty:
                st.info("Partner field counts unavailable.")
            else:
                def field_belongs_to_domain(fid_tok: str) -> bool:
                    tok = str(fid_tok).strip()
                    if tok.isdigit():
                        return look["field_id_to_domain"].get(tok) == cmp_domain
                    # if name provided, map to id then check
                    fid = look["name2id"].get(tok)
                    if fid:
                        return look["field_id_to_domain"].get(fid) == cmp_domain
                    # last resort: assume it's a field name and consult fields_by_domain
                    return tok in look["fields_by_domain"].get(cmp_domain, [])
                p_field_counts = p_field_counts[p_field_counts["id"].astype(str).apply(field_belongs_to_domain)].copy()
                partner_df = p_field_counts.copy()
                partner_df["Field"] = partner_df["id"].astype(str).apply(lambda t: look["id2name"].get(t, t) if t.isdigit() else t)
                partner_df["Label"] = partner_df["Field"]
                partner_df["pct"] = partner_df["count"].astype(float) / float(dom_total)  # 0..1
                partner_df["color"] = partner_df["Field"].apply(get_field_color)

                # Canonical order for cmp_domain
                canon_domain_fields = [f for f in look["fields_by_domain"].get(cmp_domain, []) if f in set(ul_df["Field"]).union(set(partner_df["Field"]))]
                if not canon_domain_fields:
                    canon_domain_fields = sorted(set(ul_df["Field"]).union(set(partner_df["Field"])))

                # Shared x-scale
                share_max_cmp = float(max(ul_df["pct"].max() if not ul_df.empty else 0.0,
                                          partner_df["pct"].max() if not partner_df.empty else 0.0,
                                          0.05))

                cL, cR = st.columns(2)
                with cL:
                    fig_ul = plot_barh_with_gutter(
                        df=ul_df.rename(columns={"pct":"value", "count":"count"})[["Label","value","count","color"]]
                              .assign(**{"%value": lambda x: x["value"]}),
                        order_labels=canon_domain_fields,
                        value_col="%value",
                        count_col="count",
                        title=f"{cmp_domain} — UL: % of domain publications by field",
                        color_col="color",
                        x_label="% of domain publications",
                        share_max=share_max_cmp,
                    )
                    st.pyplot(fig_ul, use_container_width=True)

                with cR:
                    fig_partner = plot_barh_with_gutter(
                        df=partner_df.rename(columns={"pct":"value"})[["Label","value","count","color"]]
                                  .assign(**{"%value": lambda x: x["value"]}),
                        order_labels=canon_domain_fields,
                        value_col="%value",
                        count_col="count",
                        title=f"{cmp_domain} — {partner_name}: % of domain publications by field",
                        color_col="color",
                        x_label="% of domain publications",
                        share_max=share_max_cmp,
                    )
                    st.pyplot(fig_partner, use_container_width=True)
