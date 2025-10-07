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

def build_fwci_table(row: pd.Series, look: Dict, counts_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return per field: min, q1, med, q3, max, color, domain, count; ordered canonically."""
    blobs = {
        "min": row.get("By field: FWCI_FR min", ""),
        "q1" : row.get("By field: FWCI_FR Q1", ""),
        "med": row.get("By field: FWCI_FR Q2", ""),
        "q3" : row.get("By field: FWCI_FR Q3", ""),
        "max": row.get("By field: FWCI_FR max", ""),
    }
    parts = {k: parse_field_value_blob(v, look) for k,v in blobs.items()}
    # union fields
    fields = set()
    for dfp in parts.values():
        fields.update(dfp["field_name"].tolist())
    order = canonical_field_order()
    ordered = [f for f in order if f in fields]
    base = pd.DataFrame({"field_name": ordered})
    for k,dfp in parts.items():
        base = base.merge(dfp.rename(columns={"value": k}), on="field_name", how="left")
    if counts_df is not None:
        base = base.merge(counts_df[["field_name","count"]], on="field_name", how="left")
    base = base.fillna({"count": 0})
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

    # ticks every 5%
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
                       title: str,
                       show_counts_gutter: bool = True) -> plt.Figure:
    """Draw min—Q1—Median—Q3—Max whiskers per field (horizontal) with left count gutter."""
    base = pd.DataFrame({"field_name": all_fields_order})
    df = base.merge(df_fwci, on="field_name", how="left").fillna(np.nan)

    y = np.arange(len(df))
    fig_h = max(1.0, 0.40 * len(df) + 0.8)
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

    # draw per row (skip if count=0 OR all stats are 0/NaN)
    for i, r in df.iterrows():
        c = r.get("color", "#7f7f7f")
        cnt = int(r.get("count") or 0)
        stats = [r.get("min"), r.get("q1"), r.get("med"), r.get("q3"), r.get("max")]
        non_na = [x for x in stats if pd.notna(x)]
        all_zero = (len(non_na) > 0 and all(float(x) == 0.0 for x in non_na))
        if cnt <= 0 or len(non_na) == 0 or all_zero:
            continue

        # whiskers
        if pd.notna(r.get("min")) and pd.notna(r.get("max")):
            ax.hlines(y[i], xmin=r["min"], xmax=r["max"], color=c, linewidth=1.2, zorder=2)
        # box (Q1-Q3)
        if pd.notna(r.get("q1")) and pd.notna(r.get("q3")) and r["q3"] >= r["q1"]:
            ax.barh(y[i], width=r["q3"]-r["q1"], left=r["q1"], height=0.5,
                    color=c, alpha=0.25, edgecolor="none", zorder=3)
        # median
        if pd.notna(r.get("med")):
            ax.vlines(r["med"], ymin=y[i]-0.25, ymax=y[i]+0.25, color=c, linewidth=2.0, zorder=4)

    # gutter counts
    if show_counts_gutter:
        counts = df.get("count")
        if counts is not None:
            for yi, cnt in enumerate(pd.Series(counts).fillna(0).astype(int).tolist()):
                if cnt > 0:
                    ax.text(-left_pad_data + offset_data, yi, f"{cnt:,}".replace(",", " "),
                            va="center", ha="left", fontsize=9, color="#444")

    # axes & cosmetics
    ax.set_title(title, fontsize=12, pad=6)
    ax.set_yticks(y)
    ax.set_yticklabels(df["field_name"], fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#eeeeee")
    ax.set_axisbelow(True)
    ax.set_xlim(-left_pad_data, xmax)
    ax.set_xlabel("FWCI (France)", fontsize=11)

    for spine in ("top","right","left"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    return fig

def plot_year_counts_bar(values: List[int], title: str, ymax: int | None = None) -> plt.Figure:
    """Simple vertical bar chart of totals per year (not stacked)."""
    years = [2019 + i for i in range(len(values))]
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
    return fig

# ------------------------------ UI ------------------------------

st.set_page_config(page_title="Lab view · Fields", layout="wide")
st.title("🏭 Lab view — Field distribution (per lab)")

look = get_lookups()
df_units = load_units()

# Filter labs and let user pick 2
labs = df_units.loc[df_units["Type"].astype(str).str.lower() == "lab"].copy()
lab_names = labs["Unit Name"].dropna().astype(str).sort_values().tolist()
if not lab_names:
    st.error("No units of Type = 'lab' found.")
    st.stop()

sel_a, sel_b = st.columns(2)
with sel_a:
    unit1 = st.selectbox("Select unit A (lab)", lab_names, index=0, key="unit_a")
with sel_b:
    default_idx = 1 if len(lab_names) > 1 else 0
    unit2 = st.selectbox("Select unit B (lab)", lab_names, index=default_idx, key="unit_b")

row1 = labs.loc[labs["Unit Name"] == unit1].iloc[0]
row2 = labs.loc[labs["Unit Name"] == unit2].iloc[0]

# Precompute field tables & shared scales/orders --------------------------------
df_f1 = build_fields_table(row1, look)
df_f2 = build_fields_table(row2, look)
fields_union = union_ordered_fields(df_f1, df_f2)
share_max = float(max(df_f1["share"].max() if not df_f1.empty else 0.0,
                      df_f2["share"].max() if not df_f2.empty else 0.0, 0.05))

df_w1 = build_fwci_table(row1, look, counts_df=df_f1)
df_w2 = build_fwci_table(row2, look, counts_df=df_f2)
fields_union_fwci = union_ordered_fields(df_w1, df_w2)
xmax_fwci = float(np.nanmax([
    df_w1["max"].max() if "max" in df_w1.columns and not df_w1["max"].isna().all() else 0.0,
    df_w2["max"].max() if "max" in df_w2.columns and not df_w2["max"].isna().all() else 0.0,
    1.0
]))

# Yearly y-axis shared max
ycounts1 = parse_pipe_number_list(row1.get("Year distribution (2019-2023)", ""))
ycounts2 = parse_pipe_number_list(row2.get("Year distribution (2019-2023)", ""))
ymax_year = max(max(ycounts1 or [0]), max(ycounts2 or [0]))

# -------------------------- helpers for table padding --------------------------

def pad_table_rows(df: pd.DataFrame, n_rows: int, numeric_cols: List[str] | None = None) -> pd.DataFrame:
    """Ensure df has exactly n_rows; truncate or pad with blanks (numeric cols -> NaN)."""
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

# -------------------------- render one lab panel --------------------------

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
    if len(cols) > 3: df["copubs"] = [ _to_int_safe(x) for x in cols[3] ]
    if len(cols) > 4: df["% UL copubs"] = [ _to_float_safe(x) for x in cols[4] ]
    return df

def render_lab_panel(container, row: pd.Series, unit_name: str,
                     df_fields: pd.DataFrame, df_fwci: pd.DataFrame,
                     yearly_values: List[int], yearly_ymax: int):
    with container:
        st.markdown(f"### {unit_name}")

        # --- KPIs ---
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Pubs (2019–2023)", f"{int(row.get('Pubs',0)):,}".replace(",", " "))
        k2.metric("… incl. LUE", f"{int(row.get('Pubs LUE',0)):,}".replace(",", " "))
        k3.metric("… incl. Top 10%", f"{int(row.get('PPtop10%',0)):,}".replace(",", " "))
        k4.metric("… incl. Top 1%", f"{int(row.get('PPtop1%',0)):,}".replace(",", " "))

        # --- Yearly totals bar (shared y-scale across labs) ---
        if yearly_values:
            fig_years = plot_year_counts_bar(yearly_values, "Yearly publications (totals)", ymax=yearly_ymax)
            st.pyplot(fig_years, use_container_width=True)
        else:
            st.info("No yearly distribution data.")

        # --- Thematic distribution ---
        fig_fields = plot_unit_fields_barh(df_fields, fields_union, share_max,
                                           "Field distribution (% of unit) — total & LUE")
        st.pyplot(fig_fields, use_container_width=True)

        # --- FWCI whiskers (with counts gutter) ---
        fig_fwci = plot_fwci_whiskers(df_fwci, fields_union_fwci, xmax_fwci,
                                      "FWCI (France) by field", show_counts_gutter=True)
        st.pyplot(fig_fwci, use_container_width=True)

        st.markdown("---")

        # --- Internal collaborations ---
        st.markdown("#### Internal collaborations")

        ck1, ck2, ck3, ck4 = st.columns(4)
        ck1.metric("Co-pubs with another lab", f"{int(row.get('Collab pubs (other labs)',0)):,}".replace(",", " "))
        ck2.metric("% with another lab", f"{((row.get('% collab w/ another internal lab') or 0)*100):.1f}%")
        ck3.metric("Co-pubs with other structures", f"{int(row.get('Collab pubs (other structures)',0)):,}".replace(",", " "))
        ck4.metric("% with other structures", f"{((row.get('% collab w/ another internal structure') or 0)*100):.1f}%")

        partners_labs  = parse_partner_ror_counts(row.get("Collab labs (by ROR)", ""))
        partners_other = parse_partner_ror_counts(row.get("Collab other structures (by ROR)", ""))
        partners = pd.concat([partners_labs, partners_other], ignore_index=True)
        if not partners.empty:
            partners = partners.groupby("ROR", as_index=False)["count"].sum().sort_values("count", ascending=False)
            # Map ROR -> Unit Name when possible (hide ROR in table)
            ror2name = df_units.set_index("ROR")["Unit Name"].to_dict() if "ROR" in df_units.columns else {}
            partners["Name"] = partners["ROR"].map(ror2name).fillna(partners["ROR"])
            top5 = partners[["Name","count"]].rename(columns={"count":"Co-pubs"})
        else:
            top5 = pd.DataFrame(columns=["Name","Co-pubs"])

        # enforce exactly 5 rows (pad with blanks)
        top5 = pad_table_rows(top5, 5, numeric_cols=["Co-pubs"])
        st.dataframe(top5, use_container_width=True, hide_index=True)

        st.markdown("---")

        # --- International partners ---
        st.markdown("#### International partners")
        st.metric("% international publications", f"{((row.get('% international') or 0)*100):.1f}%")

        intl_df = parse_parallel_lists(
            row.get("Top 10 int partners (name)", ""),
            row.get("Top 10 int partners (type)", ""),
            row.get("Top 10 int partners (country)", ""),
            row.get("Top 10 int partners (copubs with lab)", ""),
            row.get("Top 10 int partners (% of UL copubs)", ""),
        )
        # visible table (exactly 10 rows)
        vis_cols = ["name","country","copubs"]
        intl_vis = intl_df[vis_cols] if not intl_df.empty else pd.DataFrame(columns=vis_cols)
        intl_vis = intl_vis.rename(columns={"name":"Partner","country":"Country","copubs":"Co-pubs"})
        intl_vis = pad_table_rows(intl_vis, 10, numeric_cols=["Co-pubs"])
        st.dataframe(intl_vis, use_container_width=True, hide_index=True)

        with st.expander("Show additional columns"):
            extra_cols = ["name","type","% UL copubs"]
            intl_extra = intl_df[extra_cols] if not intl_df.empty else pd.DataFrame(columns=extra_cols)
            intl_extra = intl_extra.rename(columns={"name":"Partner","type":"Type","% UL copubs":"% of UL co-pops"})
            # keep same 10 rows (pad)
            intl_extra = pad_table_rows(intl_extra, 10, numeric_cols=["% of UL co-pops"])
            st.dataframe(intl_extra, use_container_width=True, hide_index=True)

        st.markdown("---")

        # --- French partners ---
        st.markdown("#### French partners")
        fr_df = parse_parallel_lists(
            row.get("Top 10 FR partners (name)", ""),
            row.get("Top 10 FR partners (type)", ""),
            "",  # no country column
            row.get("Top 10 FR partners (copubs with lab)", ""),
            row.get("Top 10 FR partners (% of UL copubs)", ""),
        )
        fr_vis_cols = ["name","copubs","% UL copubs"]
        fr_vis = fr_df[fr_vis_cols] if not fr_df.empty else pd.DataFrame(columns=fr_vis_cols)
        fr_vis["Country"] = "France"
        fr_vis = fr_vis.rename(columns={"name":"Partner","copubs":"Co-pubs","% UL copubs":"% of UL co-pops"})
        fr_vis = fr_vis[["Partner","Country","Co-pubs","% of UL co-pops"]]
        fr_vis = pad_table_rows(fr_vis, 10, numeric_cols=["Co-pubs","% of UL co-pops"])
        st.dataframe(fr_vis, use_container_width=True, hide_index=True)

        with st.expander("Show additional columns"):
            fr_extra = (fr_df[["name","type"]].rename(columns={"name":"Partner","type":"Type"})
                        if not fr_df.empty else pd.DataFrame(columns=["Partner","Type"]))
            fr_extra = pad_table_rows(fr_extra, 10, numeric_cols=None)
            st.dataframe(fr_extra, use_container_width=True, hide_index=True)

# -------------------------- Two panels side by side --------------------------
colA, colB = st.columns(2)

render_lab_panel(colA, row1, unit1, df_f1, df_w1, ycounts1, ymax_year)
render_lab_panel(colB, row2, unit2, df_f2, df_w2, ycounts2, ymax_year)
