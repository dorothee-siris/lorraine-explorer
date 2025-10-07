# pages/1_🏭_Lab_View.py
from __future__ import annotations

import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Make lib/taxonomy.py importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "lib"))

# plotting
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    st.error("matplotlib is not installed. Add it to requirements.txt or `pip install matplotlib`.")
    st.stop()

# taxonomy helpers
from lib.taxonomy import (
    build_taxonomy_lookups,
    canonical_field_order,
    get_field_color,
)

# ---------------------------- paths & cache ----------------------------

DATA_PATH = REPO_ROOT / "data" / "ul_units_indicators.parquet"

@st.cache_data(show_spinner=False)
def load_units() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)

    # Normalize column names (strip only)
    df = df.rename(columns={c: c.strip() for c in df.columns})

    # Ensure expected key columns exist (rename if casing differs)
    canon = {
        "unit name": "Unit Name",
        "type": "Type",
        "pubs": "Pubs",
        "pubs lue": "Pubs LUE",
        "pptop10%": "PPtop10%",
        "pptop1%": "PPtop1%",
        "year distribution (2019-2023)": "Year distribution (2019-2023)",
        "type distribution (articles|chapters|books|reviews, 2019-2023)": "Type distribution (articles|chapters|books|reviews, 2019-2023)",
        "by field: counts": "By field: counts",
        "by field: lue counts": "By field: LUE counts",
        "by field: fwci_fr min": "By field: FWCI_FR min",
        "by field: fwci_fr q1": "By field: FWCI_FR Q1",
        "by field: fwci_fr q2": "By field: FWCI_FR Q2",
        "by field: fwci_fr q3": "By field: FWCI_FR Q3",
        "by field: fwci_fr max": "By field: FWCI_FR max",
        "collab pubs (other labs)": "Collab pubs (other labs)",
        "% collab w/ another internal lab": "% collab w/ another internal lab",
        "collab pubs (other structures)": "Collab pubs (other structures)",
        "% collab w/ another internal structure": "% collab w/ another internal structure",
        "collab labs (by ror)": "Collab labs (by ROR)",
        "collab other structures (by ror)": "Collab other structures (by ROR)",
        "% international": "% international",
        "top 10 int partners (name)": "Top 10 int partners (name)",
        "top 10 int partners (type)": "Top 10 int partners (type)",
        "top 10 int partners (country)": "Top 10 int partners (country)",
        "top 10 int partners (copubs with lab)": "Top 10 int partners (copubs with lab)",
        "top 10 int partners (% of ul copubs)": "Top 10 int partners (% of UL copubs)",
        "top 10 fr partners (name)": "Top 10 FR partners (name)",
        "top 10 fr partners (type)": "Top 10 FR partners (type)",
        "top 10 fr partners (copubs with lab)": "Top 10 FR partners (copubs with lab)",
        "top 10 fr partners (% of ul copubs)": "Top 10 FR partners (% of UL copubs)",
        "ror": "ROR",
    }
    lower2actual = {c.lower(): c for c in df.columns}
    for low, want in canon.items():
        if low in lower2actual and lower2actual[low] != want:
            df = df.rename(columns={lower2actual[low]: want})

    # Types
    for c in ("Pubs", "Pubs LUE", "Collab pubs (other labs)", "Collab pubs (other structures)"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in ("PPtop10%", "PPtop1%", "% collab w/ another internal lab",
              "% collab w/ another internal structure", "% international"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def get_lookups() -> Dict:
    return build_taxonomy_lookups()

# --------------------------- parsing helpers ---------------------------

FIELD_PAIR_RE = re.compile(r"^\s*(.*?)\s*\(([^)]*)\)\s*$")

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

def parse_field_counts_blob(blob: str, look: Dict) -> pd.DataFrame:
    """Return DataFrame[field_name,count] from 'id/name (count) | id/name (count) ...'"""
    if pd.isna(blob) or not str(blob).strip():
        return pd.DataFrame(columns=["field_name", "count"])
    parts = [p.strip() for p in str(blob).split("|")]
    rows = []
    canon_fields = set(canonical_field_order())
    for p in parts:
        m = FIELD_PAIR_RE.match(p)
        if not m: 
            continue
        token = m.group(1).strip()
        count = _to_int_safe(m.group(2))
        field_name = look["id2name"].get(token, token) if token.isdigit() else token
        if field_name in canon_fields:
            rows.append((field_name, count))
    return pd.DataFrame(rows, columns=["field_name", "count"])

def parse_field_value_blob(blob: str, look: Dict) -> pd.DataFrame:
    """Return DataFrame[field_name,value] from 'id/name (value) | ...' where value is float."""
    if pd.isna(blob) or not str(blob).strip():
        return pd.DataFrame(columns=["field_name", "value"])
    parts = [p.strip() for p in str(blob).split("|")]
    rows = []
    canon_fields = set(canonical_field_order())
    for p in parts:
        m = FIELD_PAIR_RE.match(p)
        if not m: 
            continue
        token = m.group(1).strip()
        val = _to_float_safe(m.group(2))
        field_name = look["id2name"].get(token, token) if token.isdigit() else token
        if field_name in canon_fields:
            rows.append((field_name, val))
    return pd.DataFrame(rows, columns=["field_name", "value"])

def parse_pipe_number_list(blob: str) -> List[int]:
    """Parse '9 | 20 | 15 | 17 | 14' -> [9,20,15,17,14]."""
    if pd.isna(blob) or not str(blob).strip():
        return []
    return [_to_int_safe(x) for x in str(blob).split("|")]

def domain_from_field(field_name: str, look: Dict) -> str:
    for dom, fields in look["fields_by_domain"].items():
        if field_name in fields:
            return dom
    return "Other"

def darken_hex(hex_color: str, factor: float = 0.65) -> str:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join([c*2 for c in h])
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    r, g, b = int(r*factor), int(g*factor), int(b*factor)
    r, g, b = max(0,r), max(0,g), max(0,b)
    return f"#{r:02x}{g:02x}{b:02x}"

# --------------------------- shaping tables ---------------------------

def build_fields_table(row: pd.Series, look: Dict) -> pd.DataFrame:
    pubs = int(row.get("Pubs", 0)) or 1
    df_counts = parse_field_counts_blob(row.get("By field: counts", ""), look)
    df_lue    = parse_field_counts_blob(row.get("By field: LUE counts", ""), look)
    df = pd.merge(df_counts, df_lue, on="field_name", how="outer", suffixes=("", "_lue")).fillna(0)
    if "count_lue" not in df.columns: df["count_lue"] = 0
    df["count"] = df["count"].astype(int)
    df["count_lue"] = df["count_lue"].astype(int)
    df["share"] = (df["count"] / pubs).astype(float)
    df["lue_share"] = (df["count_lue"] / pubs).astype(float)
    # ordering & colors
    order = {n:i for i,n in enumerate(canonical_field_order())}
    df["__ord"] = df["field_name"].map(order).fillna(10_000)
    df["domain"] = df["field_name"].apply(lambda n: domain_from_field(n, look))
    df["color"] = df["field_name"].apply(get_field_color)
    df["color_lue"] = df["color"].apply(lambda c: darken_hex(c, 0.65))
    df = df.sort_values("__ord").drop(columns="__ord").reset_index(drop=True)
    return df

def build_fwci_table(row: pd.Series, look: Dict) -> pd.DataFrame:
    """Return per field: min, q1, med, q3, max, color, domain; ordered canonically."""
    blobs = {
        "min": row.get("By field: FWCI_FR min", ""),
        "q1" : row.get("By field: FWCI_FR Q1", ""),
        "med": row.get("By field: FWCI_FR Q2", ""),
        "q3" : row.get("By field: FWCI_FR Q3", ""),
        "max": row.get("By field: FWCI_FR max", ""),
    }
    parts = {k: parse_field_value_blob(v, look) for k,v in blobs.items()}
    # start from union of fields
    fields = set()
    for dfp in parts.values():
        fields.update(dfp["field_name"].tolist())
    order = canonical_field_order()
    ordered = [f for f in order if f in fields]
    base = pd.DataFrame({"field_name": ordered})
    for k,dfp in parts.items():
        base = base.merge(dfp.rename(columns={"value": k}), on="field_name", how="left")
    base = base.fillna(np.nan)
    base["domain"] = base["field_name"].apply(lambda n: domain_from_field(n, look))
    base["color"]  = base["field_name"].apply(get_field_color)
    return base

def union_ordered_fields(df1: pd.DataFrame, df2: pd.DataFrame) -> List[str]:
    present = set(df1["field_name"]).union(set(df2["field_name"]))
    return [f for f in canonical_field_order() if f in present]

# ------------------------------ plotting ------------------------------

def plot_unit_fields_barh(df_fields: pd.DataFrame,
                          all_fields_order: List[str],
                          share_max: float,
                          title: str,
                          show_counts_gutter: bool = True,
                          heights: float = 0.8) -> plt.Figure:
    base = pd.DataFrame({"field_name": all_fields_order})
    df = base.merge(df_fields, on="field_name", how="left").fillna({
        "count": 0, "share": 0.0, "count_lue": 0, "lue_share": 0.0,
        "color": "#7f7f7f", "color_lue": "#5a5a5a", "domain": "Other",
    })
    y = np.arange(len(df))
    fig_h = max(1.0, 0.42 * len(df) + 0.8)
    fig, ax = plt.subplots(figsize=(7.2, fig_h))

    # left gutter
    left_pad_px, offset_px = (72, 6)
    ax.set_xlim(0, share_max)
    fig.canvas.draw()
    bb = ax.get_window_extent(renderer=fig.canvas.get_renderer())
    ax_width_px = bb.width if bb.width > 0 else 600
    data_per_px = (share_max - 0.0) / ax_width_px if ax_width_px else 0.0001
    left_pad_data = left_pad_px * data_per_px
    offset_data   = offset_px * data_per_px
    ax.set_xlim(-left_pad_data, share_max)

    # bars
    for i, row in df.iterrows():
        ax.barh(y[i], width=row["share"], left=0.0, height=heights,
                edgecolor="none", color=row["color"], alpha=0.95, zorder=2)
        if row["lue_share"] > 0:
            ax.barh(y[i], width=row["lue_share"], left=0.0, height=heights*0.7,
                    edgecolor="none", color=row["color_lue"], alpha=1.0, zorder=3)

    # gutter counts
    if show_counts_gutter:
        for yi, cnt in enumerate(df["count"].astype(int).tolist()):
            ax.text(-left_pad_data + offset_data, yi, f"{cnt:,}".replace(",", " "),
                    va="center", ha="left", fontsize=9, color="#444")

    # axes
    ax.set_title(title, fontsize=12, pad=6)
    ax.set_yticks(y)
    ax.set_yticklabels(df["field_name"], fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#eeeeee")
    ax.set_axisbelow(True)

    # ticks every 5% (you used 10% in your snippet; set to 5% here)
    max_tick = np.ceil(share_max * 20) / 20.0
    xticks = np.arange(0.0, max(0.05, max_tick) + 1e-9, 0.05)
    xticks = xticks[xticks <= share_max + 1e-9]
    ax.set_xticks(xticks)
    ax.set_xlabel("% of unit publications", fontsize=11)
    ax.set_xlim(-left_pad_data, share_max)
    ax.set_xticklabels([f"{int(x*100)}%" for x in xticks], fontsize=10)

    for spine in ("top","right","left"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    return fig

def plot_fwci_whiskers(df_fwci: pd.DataFrame,
                       all_fields_order: List[str],
                       xmax: float,
                       title: str) -> plt.Figure:
    """Draw min—Q1—Median—Q3—Max whiskers per field (horizontal)."""
    base = pd.DataFrame({"field_name": all_fields_order})
    df = base.merge(df_fwci, on="field_name", how="left")
    df = df.fillna(np.nan)

    y = np.arange(len(df))
    fig_h = max(1.0, 0.40 * len(df) + 0.8)
    fig, ax = plt.subplots(figsize=(7.2, fig_h))

    # draw per row
    for i, r in df.iterrows():
        if pd.isna(r["min"]) and pd.isna(r["q1"]) and pd.isna(r["med"]) and pd.isna(r["q3"]) and pd.isna(r["max"]):
            continue
        # whiskers
        if pd.notna(r["min"]) and pd.notna(r["max"]):
            ax.hlines(y[i], xmin=r["min"], xmax=r["max"], color=r.get("color","#7f7f7f"), linewidth=1.2, zorder=2)
        # box (Q1-Q3)
        if pd.notna(r["q1"]) and pd.notna(r["q3"]):
            ax.barh(y[i], width=r["q3"]-r["q1"], left=r["q1"], height=0.5,
                    color=r.get("color","#7f7f7f"), alpha=0.25, edgecolor="none", zorder=3)
        # median
        if pd.notna(r["med"]):
            ax.vlines(r["med"], ymin=y[i]-0.25, ymax=y[i]+0.25, color=r.get("color","#7f7f7f"), linewidth=2.0, zorder=4)

    # axes & cosmetics
    ax.set_title(title, fontsize=12, pad=6)
    ax.set_yticks(y)
    ax.set_yticklabels(df["field_name"], fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#eeeeee")
    ax.set_axisbelow(True)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("FWCI (France)", fontsize=11)

    for spine in ("top","right","left"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    return fig

def plot_mini_stacked_bar(values: List[int], labels: List[str], title: str, colors: List[str] | None = None) -> plt.Figure:
    """Single stacked vertical bar with segments = values."""
    v = np.array(values, dtype=float)
    total = v.sum() if v.sum() > 0 else 1.0
    shares = v / total
    if colors is None:
        colors = ["#888"] * len(values)
    fig, ax = plt.subplots(figsize=(2.2, 2.6))
    bottom = 0.0
    for i, s in enumerate(shares):
        ax.bar(0, s, bottom=bottom, color=colors[i], edgecolor="white", linewidth=0.5)
        bottom += s
    ax.set_title(title, fontsize=11, pad=4)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([0,0.5,1.0])
    ax.set_yticklabels(["0%","50%","100%"], fontsize=8)
    ax.grid(False)
    for spine in ("top","right","left","bottom"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return fig

# ------------------------------ UI ------------------------------

st.set_page_config(page_title="Lab view · Fields", layout="wide")
st.title("🏭 Lab view — Field distribution")

look = get_lookups()
df_units = load_units()

# Filter labs and let user pick 2
labs = df_units.loc[df_units["Type"].astype(str).str.lower() == "lab"].copy()
lab_names = labs["Unit Name"].dropna().astype(str).sort_values().tolist()
if not lab_names:
    st.error("No units of Type = 'lab' found.")
    st.stop()

csel1, csel2 = st.columns(2)
with csel1:
    unit1 = st.selectbox("Select unit A (lab)", lab_names, index=0, key="unit_a")
with csel2:
    default_idx = 1 if len(lab_names) > 1 else 0
    unit2 = st.selectbox("Select unit B (lab)", lab_names, index=default_idx, key="unit_b")

row1 = labs.loc[labs["Unit Name"] == unit1].iloc[0]
row2 = labs.loc[labs["Unit Name"] == unit2].iloc[0]

# ------------------------- KPIs + mini bars -------------------------

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Pubs (2019–2023)", f"{int(row1.get('Pubs',0)):,}".replace(",", " "), help=unit1)
kpi2.metric("… incl. LUE", f"{int(row1.get('Pubs LUE',0)):,}".replace(",", " "))
kpi3.metric("… incl. Top 10%", f"{int(row1.get('PPtop10%',0)):,}".replace(",", " "))
kpi4.metric("… incl. Top 1%", f"{int(row1.get('PPtop1%',0)):,}".replace(",", " "))

# small stacked bars
yb = parse_pipe_number_list(row1.get("Year distribution (2019-2023)", ""))
tb = parse_pipe_number_list(row1.get("Type distribution (articles|chapters|books|reviews, 2019-2023)", ""))

mini1, mini2 = st.columns(2)
with mini1:
    years = [2019,2020,2021,2022,2023]
    year_colors = ["#ddd", "#cfcfcf", "#bfbfbf", "#a9a9a9", "#8f8f8f"]
    fig_y = plot_mini_stacked_bar(yb, [str(y) for y in years[:len(yb)]], "Year distribution", year_colors[:len(yb)])
    st.pyplot(fig_y, use_container_width=False)
with mini2:
    types = ["Articles","Chapters","Books","Reviews"]
    # map types to domain-like palette for visual consistency
    type_colors = ["#8190FF","#FFCB3A","#0CA750","#F85C32"]
    fig_t = plot_mini_stacked_bar(tb, types[:len(tb)], "Types", type_colors[:len(tb)])
    st.pyplot(fig_t, use_container_width=False)

st.markdown("---")

# ------------------- Field bars (share % + LUE overlay) -------------------

df_f1 = build_fields_table(row1, look)
df_f2 = build_fields_table(row2, look)
fields_union = union_ordered_fields(df_f1, df_f2)
share_max = float(max(df_f1["share"].max() if not df_f1.empty else 0.0,
                      df_f2["share"].max() if not df_f2.empty else 0.0, 0.05))

c1, c2 = st.columns(2)
with c1:
    fig1 = plot_unit_fields_barh(df_f1, fields_union, share_max, f"{unit1} — field distribution")
    st.pyplot(fig1, use_container_width=True)
with c2:
    fig2 = plot_unit_fields_barh(df_f2, fields_union, share_max, f"{unit2} — field distribution")
    st.pyplot(fig2, use_container_width=True)

# --------------------------- FWCI whiskers ---------------------------

df_w1 = build_fwci_table(row1, look)
df_w2 = build_fwci_table(row2, look)
fields_union_fwci = union_ordered_fields(df_w1.rename(columns={"field_name":"field_name"}),
                                         df_w2.rename(columns={"field_name":"field_name"}))

xmax_fwci = np.nanmax([
    df_w1["max"].max() if "max" in df_w1.columns and not df_w1["max"].isna().all() else 0.0,
    df_w2["max"].max() if "max" in df_w2.columns and not df_w2["max"].isna().all() else 0.0,
    1.0
])

st.subheader("FWCI (France) by field")
c3, c4 = st.columns(2)
with c3:
    figw1 = plot_fwci_whiskers(df_w1, fields_union_fwci, xmax_fwci, f"{unit1} — FWCI by field")
    st.pyplot(figw1, use_container_width=True)
with c4:
    figw2 = plot_fwci_whiskers(df_w2, fields_union_fwci, xmax_fwci, f"{unit2} — FWCI by field")
    st.pyplot(figw2, use_container_width=True)

st.markdown("---")

# ---------------------- Internal collaborations ----------------------

st.subheader("Internal collaborations")

# KPIs
ck1, ck2, ck3, ck4 = st.columns(4)
ck1.metric("Co-pubs with another lab", f"{int(row1.get('Collab pubs (other labs)',0)):,}".replace(",", " "))
ck2.metric("% with another lab", f"{(row1.get('% collab w/ another internal lab') or 0):.1f}%")
ck3.metric("Co-pubs with other structures", f"{int(row1.get('Collab pubs (other structures)',0)):,}".replace(",", " "))
ck4.metric("% with other structures", f"{(row1.get('% collab w/ another internal structure') or 0):.1f}%")

# Top 5 internal partners (labs + other structures by ROR)
def parse_partner_ror_counts(blob: str) -> pd.DataFrame:
    if pd.isna(blob) or not str(blob).strip():
        return pd.DataFrame(columns=["ROR","count"])
    parts = [p.strip() for p in str(blob).split("|")]
    rows = []
    for p in parts:
        m = FIELD_PAIR_RE.match(p)
        if not m: 
            continue
        ror = m.group(1).strip()
        cnt = _to_int_safe(m.group(2))
        rows.append((ror, cnt))
    df = pd.DataFrame(rows, columns=["ROR","count"])
    return df.groupby("ROR", as_index=False)["count"].sum().sort_values("count", ascending=False)

partners_labs  = parse_partner_ror_counts(row1.get("Collab labs (by ROR)", ""))
partners_other = parse_partner_ror_counts(row1.get("Collab other structures (by ROR)", ""))
partners = pd.concat([partners_labs, partners_other], ignore_index=True)
if not partners.empty:
    partners = partners.groupby("ROR", as_index=False)["count"].sum().sort_values("count", ascending=False)

    # Map ROR -> Unit Name when possible (using the same file)
    ror2name = df_units.set_index("ROR")["Unit Name"].to_dict() if "ROR" in df_units.columns else {}
    partners["Name"] = partners["ROR"].map(ror2name).fillna(partners["ROR"])
    partners = partners[["Name","ROR","count"]].head(5)

    st.markdown("**Top 5 internal partners**")
    st.dataframe(
        partners,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Name": st.column_config.TextColumn("Name"),
            "ROR":  st.column_config.TextColumn("ROR", help="Registry identifier"),
            "count": st.column_config.NumberColumn("Co-pubs"),
        }
    )
else:
    st.info("No internal partner data.")

st.markdown("---")

# --------------------- International partners ---------------------

st.subheader("International partners")
st.metric("% international publications", f"{(row1.get('% international') or 0):.1f}%")

def parse_parallel_lists(names_blob: str, *more_blobs: str) -> pd.DataFrame:
    """
    Parse pipe-separated parallel columns into a DataFrame.
    e.g., "A | B" + "type1 | type2" + "FR | DE" + "3 | 5" + "0.2 | 0.8"
    """
    def split_clean(s: str) -> List[str]:
        if pd.isna(s) or not str(s).strip():
            return []
        return [x.strip() for x in str(s).split("|")]
    cols = [split_clean(names_blob)]
    for b in more_blobs:
        cols.append(split_clean(b))
    n = max(len(c) for c in cols) if cols else 0
    cols = [c + [""]*(n-len(c)) for c in cols]
    df = pd.DataFrame({"name": cols[0]})
    if len(cols) > 1: df["type"] = cols[1]
    if len(cols) > 2: df["country"] = cols[2]
    if len(cols) > 3: df["copubs"] = [ _to_int_safe(x) for x in cols[3] ]
    if len(cols) > 4: df["% UL copubs"] = [ _to_float_safe(x) for x in cols[4] ]
    return df

intl_df = parse_parallel_lists(
    row1.get("Top 10 int partners (name)", ""),
    row1.get("Top 10 int partners (type)", ""),
    row1.get("Top 10 int partners (country)", ""),
    row1.get("Top 10 int partners (copubs with lab)", ""),
    row1.get("Top 10 int partners (% of UL copubs)", ""),
)
if not intl_df.empty:
    st.dataframe(
        intl_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "name": st.column_config.TextColumn("Partner"),
            "country": st.column_config.TextColumn("Country"),
            "copubs": st.column_config.NumberColumn("Co-pubs"),
            "type": st.column_config.TextColumn("Type", help="Hidden by default"),
            "% UL copubs": st.column_config.NumberColumn("% of UL co-pubs", help="Hidden by default", format="%.2f"),
        },
        column_order=["name","country","copubs","type","% UL copubs"],
        hide_columns=["type","% UL copubs"],   # hidden by default (you can toggle in UI)
    )
else:
    st.info("No international partner data.")

st.markdown("---")

# ------------------------ French partners ------------------------

st.subheader("French partners")

fr_df = parse_parallel_lists(
    row1.get("Top 10 FR partners (name)", ""),
    row1.get("Top 10 FR partners (type)", ""),
    "",  # no country column for FR list
    row1.get("Top 10 FR partners (copubs with lab)", ""),
    row1.get("Top 10 FR partners (% of UL copubs)", ""),
)
if not fr_df.empty:
    # ensure columns exist
    if "country" not in fr_df.columns: fr_df["country"] = "France"
    st.dataframe(
        fr_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "name": st.column_config.TextColumn("Partner"),
            "country": st.column_config.TextColumn("Country"),
            "copubs": st.column_config.NumberColumn("Co-pubs"),
            "% UL copubs": st.column_config.NumberColumn("% of UL co-pubs", format="%.2f"),
            "type": st.column_config.TextColumn("Type", help="Hidden by default"),
        },
        column_order=["name","country","copubs","% UL copubs","type"],
        hide_columns=["type"],   # hidden by default
    )
else:
    st.info("No French partner data.")
