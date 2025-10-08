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
    get_subfield_color,
    build_taxonomy_lookups,
    get_domain_color,
)

# ---------------------------- paths & cache ----------------------------

UNITS_PATH = REPO_ROOT / "data" / "ul_units_indicators.parquet"
PUBS_PATH  = REPO_ROOT / "data" / "pubs_final.parquet"  # used only for the inter-lab co-publications section

YEAR_START, YEAR_END = 2019, 2023

# Domain colors (legend for collab section)
DOMAIN_COLORS = {
    "Health Sciences": "#F85C32",
    "Life Sciences": "#0CA750",
    "Physical Sciences": "#8190FF",
    "Social Sciences": "#FFCB3A",
    "Other": "#7f7f7f",
}

# ------------------------------- loaders -------------------------------

@st.cache_data(show_spinner=False)
def load_units() -> pd.DataFrame:
    df = pd.read_parquet(UNITS_PATH)

    # Normalize column names (strip only)
    df = df.rename(columns={c: c.strip() for c in df.columns})

    # Ensure expected key columns exist (rename if casing differs)
    canon = {
        "unit name": "Unit Name",
        "department": "Department",
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
        "by domain and by year: counts": "By domain and by year: counts",
        "by subfield: counts": "By subfield: counts",
        "collab pubs (other labs)": "Collab pubs (other labs)",
        "% collab w/ another internal lab": "% collab w/ another internal lab",
        "collab pubs (other structures)": "Collab pubs (other structures)",
        "% collab w/ another internal structure": "% collab w/ another internal structure",
        "collab labs (by ror)": "Collab labs (by ROR)",
        "collab other structures (by ror)": "Collab other structures (by ROR)",
        "% international": "% international",
        "% industrial": "% industrial",
        "avg fwci (france)": "Avg FWCI (France)",
        "% pubs (uni level)": "% Pubs (uni level)",
        "% pubs lue (lab level)": "% Pubs LUE (lab level)",
        "see in openalex": "See in OpenAlex",
        "top 10 int partners (name)": "Top 10 int partners (name)",
        "top 10 int partners (type)": "Top 10 int partners (type)",
        "top 10 int partners (country)": "Top 10 int partners (country)",
        "top 10 int partners (copubs with lab)": "Top 10 int partners (copubs with lab)",
        "top 10 int partners (% of ul copubs)": "Top 10 int partners (% of UL copubs)",
        "top 10 fr partners (name)": "Top 10 FR partners (name)",
        "top 10 fr partners (type)": "Top 10 FR partners (type)",
        "top 10 fr partners (copubs with lab)": "Top 10 FR partners (copubs with lab)",
        "top 10 fr partners (% of ul copubs)": "Top 10 FR partners (% of UL copubs)",
        "top 10 authors (name)": "Top 10 authors (name)",
        "top 10 authors (pubs)": "Top 10 authors (pubs)",
        "top 10 authors (average fwci_fr)": "Top 10 authors (Average FWCI_FR)",
        "top 10 authors (is lorraine)": "Top 10 authors (Is Lorraine)",
        "top 10 authors (other lab(s))": "Top 10 authors (Other lab(s))",
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
              "% collab w/ another internal structure", "% international",
              "% industrial", "% Pubs (uni level)", "% Pubs LUE (lab level)"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Avg FWCI (France)" in df.columns:
        df["Avg FWCI (France)"] = pd.to_numeric(df["Avg FWCI (France)"], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def load_pubs() -> pd.DataFrame | None:
    try:
        df = pd.read_parquet(PUBS_PATH)
        return df
    except Exception as e:
        st.warning(f"Could not load pubs_final.parquet at {PUBS_PATH}. The inter-lab collaboration section will be skipped.\n\n{e}")
        return None

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

# --- NEW: parse "By subfield: counts"  -> subfield_id,count,name,color
def parse_subfield_counts_blob(blob: str, look: Dict) -> pd.DataFrame:
    """
    '2505 (7) | 1211 (3) | ...' -> DataFrame[subfield_id,str; count,int; name,str; color,str]
    """
    # Always return a DF
    empty = pd.DataFrame(columns=["subfield_id", "count", "name", "color"])

    if pd.isna(blob) or not str(blob).strip():
        return empty

    parts = [p.strip() for p in str(blob).split("|")]
    rows = []
    for p in parts:
        m = re.match(r"^\s*([^\s()]+)\s*\((\d+)\)\s*$", p)
        if not m:
            continue
        sid = str(m.group(1)).strip()
        cnt = _to_int_safe(m.group(2))
        rows.append((sid, cnt))

    if not rows:
        return empty

    df = pd.DataFrame(rows, columns=["subfield_id", "count"])
    id2name = look.get("id2name", {})
    df["name"]  = df["subfield_id"].map(lambda x: id2name.get(str(x), str(x)))
    df["color"] = df["subfield_id"].map(lambda x: get_subfield_color(str(x)))
    df = df.groupby(["subfield_id", "name", "color"], as_index=False)["count"].sum()
    return df


# --- NEW: parse "By domain and by year: counts" -> (year, domain_name, count)
def parse_domain_year_counts_blob(blob: str, look: Dict) -> pd.DataFrame:
    """
    Robustly parse blobs like:
      '2019 (1 : 7 | 2 : 0 | 3 : 0 | 4 : 2) ; 2020 (1 : 3 | 2 : 1 | ...)'
    Also tolerates noisy pairs like '3 : domain id : 7' by extracting ints and
    taking FIRST int as domain id, LAST int as count.
    """
    if pd.isna(blob) or not str(blob).strip():
        return pd.DataFrame(columns=["year","domain","count"])

    rows = []
    # split by semicolons into per-year chunks
    for part in re.split(r"[;]+", str(blob)):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^\s*(\d{4})\s*\((.*?)\)\s*$", part)
        if not m:
            continue
        year = int(m.group(1))
        inside = m.group(2)

        # split the inside by '|' into domain:count-ish pairs
        for pair in re.split(r"[|]", inside):
            pair = pair.strip()
            if not pair:
                continue

            # try robust integer extraction
            ints = re.findall(r"(-?\d+)", pair)
            if len(ints) >= 2:
                dom_id = ints[0]          # FIRST int -> domain id
                cnt    = _to_int_safe(ints[-1])  # LAST int -> count
            else:
                # last fallback (rare): exact 'X : Y'
                dm = re.match(r"^\s*([^\s:]+)\s*:\s*([0-9]+)\s*$", pair)
                if not dm:
                    continue
                dom_id = dm.group(1).strip()
                cnt    = _to_int_safe(dm.group(2))

            dom_name = look["id2name"].get(str(dom_id), str(dom_id))
            rows.append((year, dom_name, cnt))

    df = pd.DataFrame(rows, columns=["year","domain","count"])
    if df.empty:
        return df

    # keep only known domains; unknown -> "Other"
    known = set(look.get("domain_order", []))
    df["domain"] = df["domain"].apply(lambda d: d if d in known else "Other")

    # aggregate duplicates (same year/domain repeated in the blob)
    df = df.groupby(["year","domain"], as_index=False)["count"].sum()
    return df


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
    """Draw min—Q1—Median—Q3—Max whiskers per field (horizontal) with left count gutter.
       Skip fields with count==0 or all stats are 0/NaN."""
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

    # draw per row
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

from matplotlib.ticker import MaxNLocator

# --- NEW: yearly stacked bar by domain ---
def plot_yearly_stacked_by_domain(df_yr_dom: pd.DataFrame,
                                  look: Dict,
                                  title: str,
                                  ymax: int | None = None) -> plt.Figure:
    """
    df_yr_dom: DataFrame[year, domain, count]
    """
    if df_yr_dom.empty:
        fig, ax = plt.subplots(figsize=(7.6, 2.8))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return fig

    years = sorted(df_yr_dom["year"].unique().tolist())
    dom_order = look["domain_order"][:]  # keep canonical order
    pivot = (df_yr_dom.pivot_table(index="year", columns="domain", values="count",
                                   aggfunc="sum", fill_value=0)
                       .reindex(index=years, fill_value=0))

    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    bottoms = np.zeros(len(years), dtype=int)
    for dom in dom_order:
        if dom not in pivot.columns:
            continue
        vals = pivot[dom].astype(int).values
        ax.bar(years, vals, bottom=bottoms,
               color=get_domain_color(dom), edgecolor="none", linewidth=0,
               antialiased=False, label=dom)
        bottoms += vals

    ax.set_title(title, fontsize=12, pad=6)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Publications (count)", fontsize=11)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], fontsize=10)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if ymax is not None and ymax > 0:
        ax.set_ylim(0, ymax)
    ax.grid(False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return fig


# ------------------------------ UI ------------------------------

st.set_page_config(page_title="Lab view · Fields", layout="wide")
st.title("🏭 Lab view — Field distribution (per lab)")

look = get_lookups()
df_units = load_units()
df_pubs  = load_pubs()  # may be None

# ----------------------------- topline before comparison -----------------------------

st.subheader("Topline metrics (2019–2023)")
k1, k2, k3 = st.columns(3)
k1.metric("Number of labs", "61")
k2.metric("Total publications", "26 541")
k3.metric("% covered by the labs", "73,03 %")
st.caption("Values provided by user; no computation performed.")

st.divider()

# -------------------------- per-lab overview table (from ul_units_indicators only) --------------------------

st.subheader("Per-lab overview (2019–2023)")

# Only labs
labs_overview = df_units[df_units["Type"].astype(str).str.lower().eq("lab")].copy()

# Columns we need from ul_units_indicators (use exact names from the parquet)
needed_cols = [
    "ROR",
    "Unit Name",
    "Pubs",
    "% Pubs (uni level)",
    "Pubs LUE",
    "% Pubs LUE (lab level)",
    "% collab w/ another internal lab",
    "% collab w/ another internal structure",
    "% international",
    "% industrial",
    "Avg FWCI (France)",
    "See in OpenAlex",
]

# If any column is missing, create a safe default so we never KeyError
for c in needed_cols:
    if c not in labs_overview.columns:
        # Percent-like columns get 0.0, ints get 0, text gets ""
        if c in {"% Pubs (uni level)", "% Pubs LUE (lab level)",
                 "% collab w/ another internal lab", "% collab w/ another internal structure",
                 "% international", "% industrial"}:
            labs_overview[c] = 0.0
        elif c in {"Pubs"}:
            labs_overview[c] = 0
        else:
            labs_overview[c] = ""

summary = labs_overview[needed_cols].copy()

# Friendly column names (also place new internal-collab columns)
summary = summary.rename(columns={
    "Unit Name": "Lab",
    "Pubs": "Publications",
    "% Pubs (uni level)": "% UL pubs",
    "Pubs LUE": "Pubs LUE",
    "% Pubs LUE (lab level)": "% LUE (lab)",
    "% collab w/ another internal lab": "% internal collabs (lab)",
    "% collab w/ another internal structure": "% internal collabs (other)",
    "% international": "% international",
    "% industrial": "% industrial",
    "Avg FWCI (France)": "Avg FWCI (FR)",
    "See in OpenAlex": "OpenAlex",
})

# Convert precomputed ratios (0–1) to percentages
pct_cols = [
    "% UL pubs",
    "% LUE (lab)",
    "% internal collabs (lab)",
    "% internal collabs (other)",
    "% international",
    "% industrial",
]
for c in pct_cols:
    summary[c] = pd.to_numeric(summary[c], errors="coerce") * 100.0

# Sort by publications
summary = summary.sort_values("Publications", ascending=False)

# Maxima for progress bars
max_share         = float(summary["% UL pubs"].max() or 1.0)
max_lue           = float(summary["% LUE (lab)"].max() or 1.0)
max_collab_lab    = float(summary["% internal collabs (lab)"].max() or 1.0)
max_collab_other  = float(summary["% internal collabs (other)"].max() or 1.0)
max_intl          = float(summary["% international"].max() or 1.0)
max_comp          = float(summary["% industrial"].max() or 1.0)

# Display (ROR last; new collab columns between % LUE and % international)
st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True,
    column_order=[
        "Lab", "Publications", "% UL pubs", "Pubs LUE", "% LUE (lab)",
        "% internal collabs (lab)", "% internal collabs (other)",
        "% international", "% industrial",
        "Avg FWCI (FR)", "OpenAlex", "ROR",
    ],
    column_config={
        "Lab": st.column_config.TextColumn("Lab"),
        "Publications": st.column_config.NumberColumn("Publications", format="%.0f"),
        "% UL pubs": st.column_config.ProgressColumn(
            "% Université de Lorraine", format="%.1f %%", min_value=0.0, max_value=max_share
        ),
        "Pubs LUE": st.column_config.NumberColumn("Pubs LUE", format="%.0f"),
        "% LUE (lab)": st.column_config.ProgressColumn(
            "% of pubs LUE", format="%.1f %%", min_value=0.0, max_value=max_lue
        ),
        "% internal collabs (lab)": st.column_config.ProgressColumn(
            "% internal collabs (lab)", format="%.1f %%", min_value=0.0, max_value=max_collab_lab
        ),
        "% internal collabs (other)": st.column_config.ProgressColumn(
            "% internal collabs (other)", format="%.1f %%", min_value=0.0, max_value=max_collab_other
        ),
        "% international": st.column_config.ProgressColumn(
            "% international", format="%.1f %%", min_value=0.0, max_value=max_intl
        ),
        "% industrial": st.column_config.ProgressColumn(
            "% with company", format="%.1f %%", min_value=0.0, max_value=max_comp
        ),
        "Avg FWCI (FR)": st.column_config.NumberColumn("Avg. FWCI (FR)", format="%.3f"),
        "OpenAlex": st.column_config.LinkColumn("See in OpenAlex"),
        "ROR": st.column_config.TextColumn("ROR"),
    },
)

st.divider()

# -------------------- Lab selection & prep for per-lab panels --------------------

# Filter labs and let user pick 2
labs_only = df_units.loc[df_units["Type"].astype(str).str.lower() == "lab"].copy()
lab_names = labs_only["Unit Name"].dropna().astype(str).sort_values().tolist()
if not lab_names:
    st.error("No units of Type = 'lab' found.")
    st.stop()

sel_a, sel_b = st.columns(2)
with sel_a:
    unit1 = st.selectbox("Select unit A (lab)", lab_names, index=0, key="unit_a")
with sel_b:
    default_idx = 1 if len(lab_names) > 1 else 0
    unit2 = st.selectbox("Select unit B (lab)", lab_names, index=default_idx, key="unit_b")

row1 = labs_only.loc[labs_only["Unit Name"] == unit1].iloc[0]
row2 = labs_only.loc[labs_only["Unit Name"] == unit2].iloc[0]

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

# Domain-by-year blobs (per lab) + shared y max for stacked charts
df_yd1 = parse_domain_year_counts_blob(row1.get("By domain and by year: counts", ""), look)
df_yd2 = parse_domain_year_counts_blob(row2.get("By domain and by year: counts", ""), look)

def _max_year_total(df):
    if df.empty:
        return 0
    return int(df.groupby("year")["count"].sum().max() or 0)

ymax_year = max(_max_year_total(df_yd1), _max_year_total(df_yd2), 1)


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

# -------------------------- parse helpers (partners/authors) --------------------------

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

def parse_authors(row: pd.Series) -> pd.DataFrame:
    names = str(row.get("Top 10 authors (name)", "")).split("|") if row.get("Top 10 authors (name)", "") else []
    pubs  = [ _to_int_safe(x) for x in str(row.get("Top 10 authors (pubs)", "")).split("|") ] if row.get("Top 10 authors (pubs)", "") else []
    fwci  = [ _to_float_safe(x) for x in str(row.get("Top 10 authors (Average FWCI_FR)", "")).split("|") ] if row.get("Top 10 authors (Average FWCI_FR)", "") else []
    isul  = [ s.strip().lower() in ("true","1","yes") for s in str(row.get("Top 10 authors (Is Lorraine)", "")).split("|") ] if row.get("Top 10 authors (Is Lorraine)", "") else []
    other = str(row.get("Top 10 authors (Other lab(s))", "")).split("|") if row.get("Top 10 authors (Other lab(s))", "") else []
    n = max(len(names), len(pubs), len(fwci), len(isul), len(other))
    def pad(lst, fill): return lst + [fill]*(n-len(lst))
    names = pad([x.strip() for x in names], "")
    pubs  = pad(pubs, np.nan)
    fwci  = pad(fwci, np.nan)
    isul  = pad(isul, "")
    other = pad([x.strip() for x in other], "")
    df = pd.DataFrame({
        "Author": names,
        "Pubs": pubs,
        "Avg FWCI (FR)": fwci,
        "Is UL": ["Yes" if x is True else ("No" if x is False else "") for x in isul],
        "Other UL lab(s)": [s.replace(";", ", ") for s in other],
    })
    return df

# --- NEW: subfield wordcloud ---
def render_subfield_wordcloud(df_sub: pd.DataFrame | None, title: str):
    """
    df_sub: DataFrame with columns ['name','count','color'] or None.
    Renders a wordcloud colored by each subfield's domain color.
    """
    if df_sub is None or (isinstance(df_sub, pd.DataFrame) and df_sub.empty):
        st.info("No subfield data for wordcloud.")
        return

    try:
        from wordcloud import WordCloud
    except Exception:
        st.info("Install `wordcloud` to see the subfield wordcloud.")
        return

    # Normalize & validate
    df = df_sub.copy()
    for col in ("name", "count"):
        if col not in df.columns:
            st.info("Subfield data missing required columns.")
            return

    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)
    df = df[df["count"] > 0]
    if df.empty:
        st.info("No subfield counts > 0 to display.")
        return

    # Build frequency and color maps
    freqs = dict(zip(df["name"].astype(str), df["count"].astype(int)))
    if "color" in df.columns:
        name2color = dict(zip(df["name"].astype(str), df["color"].astype(str)))
    else:
        name2color = {}

    def wc_color_func(word, *args, **kwargs):
        hexcol = name2color.get(word, "#7f7f7f")
        h = hexcol.lstrip("#")
        try:
            return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
        except Exception:
            return (127, 127, 127)

    # Render
    wc = WordCloud(width=900, height=350, background_color="white", prefer_horizontal=0.95)
    wc.generate_from_frequencies(freqs)
    wc.recolor(color_func=wc_color_func)

    fig_wc, ax_wc = plt.subplots(figsize=(6.5, 2.6))  # a bit narrower for column
    ax_wc.imshow(wc, interpolation="bilinear")
    ax_wc.axis("off")
    ax_wc.set_title(title, fontsize=12, pad=6)

    st.pyplot(fig_wc, use_container_width=True)


# -------------------------- render one lab panel --------------------------

def render_lab_panel(container, row: pd.Series, unit_name: str,
                     df_fields: pd.DataFrame, df_fwci: pd.DataFrame,
                     df_year_domain: pd.DataFrame, ymax_year: int):

    with container:
        st.markdown(f"### {unit_name}")

        # --- KPIs ---
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Pubs (2019–2023)", f"{int(row.get('Pubs',0)):,}".replace(",", " "))
        k2.metric("… incl. LUE", f"{int(row.get('Pubs LUE',0)):,}".replace(",", " "))
        k3.metric("… incl. Top 10%", f"{int(row.get('PPtop10%',0)):,}".replace(",", " "))
        k4.metric("… incl. Top 1%", f"{int(row.get('PPtop1%',0)):,}".replace(",", " "))

        # ----- Legend (right after KPIs)
        looks = build_taxonomy_lookups()
        legend_items = "".join(
            f'<div class="legend-item"><span class="legend-swatch" style="background:{get_domain_color(d)};"></span>{d}</div>'
            for d in looks["domain_order"]
        )
        st.markdown(
            """
            <style>
            .legend-row { display:flex; gap:12px; align-items:center; margin: 6px 0 10px 2px; }
            .legend-item { display:flex; align-items:center; gap:6px; font-size: 0.95rem; color:#333; }
            .legend-swatch { display:inline-block; width:14px; height:14px; border-radius:3px; }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown(f'<div class="legend-row">{legend_items}</div>', unsafe_allow_html=True)

        # --- NEW: Subfield wordcloud (all publications of the lab) ---
        if "By subfield: counts" in row.index:
            df_sub = parse_subfield_counts_blob(row.get("By subfield: counts", ""), look)
        else:
            df_sub = None
        render_subfield_wordcloud(df_sub, "Subfields in unit publications (size = frequency)")

        # --- NEW: Yearly stacked by domain (from precomputed blob) ---
        if not df_year_domain.empty:
            fig_years = plot_yearly_stacked_by_domain(
                df_year_domain, look, "Yearly publications by domain", ymax=ymax_year
            )
            st.pyplot(fig_years, use_container_width=True)
        else:
            st.info("No domain-by-year distribution data.")

        # --- Thematic distribution (fields) ---
        fig_fields = plot_unit_fields_barh(
            df_fields, fields_union, share_max,
            "Field distribution (% of unit) — total & LUE"
        )
        st.pyplot(fig_fields, use_container_width=True)

        # --- FWCI whiskers (with counts gutter) ---
        fig_fwci = plot_fwci_whiskers(
            df_fwci, fields_union_fwci, xmax_fwci,
            "FWCI (France) by field", show_counts_gutter=True
        )
        st.pyplot(fig_fwci, use_container_width=True)

        st.markdown("---")

        # --- Top 5 internal partners ---
        st.markdown("#### Top 5 internal partners")

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
            # Enrich with Name / Department / Type from ul_units_indicators
            if {"ROR","Unit Name","Department","Type"}.issubset(df_units.columns):
                ref = df_units[["ROR","Unit Name","Department","Type"]].drop_duplicates()
                partners = partners.merge(ref, on="ROR", how="left")
                partners["Name"] = partners["Unit Name"].fillna(partners["ROR"])
            else:
                partners["Name"] = partners["ROR"]
                partners["Department"] = ""
                partners["Type"] = ""
            top5 = partners[["Name","Department","Type","count"]].rename(columns={"count":"Co-pubs"})
        else:
            top5 = pd.DataFrame(columns=["Name","Department","Type","Co-pubs"])

        # enforce exactly 5 rows (pad with blanks)
        top5 = pad_table_rows(top5, 5, numeric_cols=["Co-pubs"])
        st.dataframe(
            top5,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Name": st.column_config.TextColumn("Partner"),
                "Department": st.column_config.TextColumn("Department"),
                "Type": st.column_config.TextColumn("Type"),
                "Co-pubs": st.column_config.NumberColumn("Co-pubs"),
            },
        )

        st.markdown("---")

        # --- Top 10 International Partners ---
        st.markdown("#### Top 10 International Partners")
        st.metric("% international publications", f"{((row.get('% international') or 0)*100):.1f}%")

        intl_df = parse_parallel_lists(
            row.get("Top 10 int partners (name)", ""),
            row.get("Top 10 int partners (type)", ""),
            row.get("Top 10 int partners (country)", ""),
            row.get("Top 10 int partners (copubs with lab)", ""),
            row.get("Top 10 int partners (% of UL copubs)", ""),
        )
        if intl_df.empty:
            intl_df = pd.DataFrame(columns=["name","country","copubs","type","% UL copubs"])

        intl_df["% of UL copubs with this partner"] = pd.to_numeric(
            intl_df.get("% UL copubs"), errors="coerce"
        ) * 100.0
        intl_df = intl_df.rename(columns={
            "name":"Partner","country":"Country","copubs":"Co-pubs","type":"Type"
        })[["Partner","Country","Co-pubs","% of UL copubs with this partner","Type"]]
        intl_df = pad_table_rows(intl_df, 10, numeric_cols=["Co-pubs","% of UL copubs with this partner"])

        st.dataframe(
            intl_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Partner": st.column_config.TextColumn("Partner"),
                "Country": st.column_config.TextColumn("Country"),
                "Co-pubs": st.column_config.NumberColumn("Co-pubs"),
                "% of UL copubs with this partner": st.column_config.ProgressColumn(
                    "% of UL copubs with this partner", format="%.1f %%", min_value=0.0, max_value=100.0
                ),
                "Type": st.column_config.TextColumn("Type"),
            },
        )

        st.markdown("---")

        # --- Top 10 French Partners ---
        st.markdown("#### Top 10 French Partners")
        fr_df = parse_parallel_lists(
            row.get("Top 10 FR partners (name)", ""),
            row.get("Top 10 FR partners (type)", ""),
            "",  # no country column
            row.get("Top 10 FR partners (copubs with lab)", ""),
            row.get("Top 10 FR partners (% of UL copubs)", ""),
        )
        if fr_df.empty:
            fr_df = pd.DataFrame(columns=["name","type","copubs","% UL copubs"])

        fr_df["% of UL copubs with this partner"] = pd.to_numeric(
            fr_df.get("% UL copubs"), errors="coerce"
        ) * 100.0
        fr_df = fr_df.rename(columns={"name":"Partner","type":"Type","copubs":"Co-pubs"})[
            ["Partner","Co-pubs","% of UL copubs with this partner","Type"]
        ]
        fr_df = pad_table_rows(fr_df, 10, numeric_cols=["Co-pubs","% of UL copubs with this partner"])

        st.dataframe(
            fr_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Partner": st.column_config.TextColumn("Partner"),
                "Co-pubs": st.column_config.NumberColumn("Co-pubs"),
                "% of UL copubs with this partner": st.column_config.ProgressColumn(
                    "% of UL copubs with this partner", format="%.1f %%", min_value=0.0, max_value=100.0
                ),
                "Type": st.column_config.TextColumn("Type"),
            },
        )

        st.markdown("---")

        # --- Top 10 Authors ---
        st.markdown("#### Top 10 Authors")
        authors = parse_authors(row)  # expects your helper to return columns: Author, Pubs, Avg FWCI (FR), Is UL?, Other UL lab(s)
        authors = pad_table_rows(authors, 10, numeric_cols=["Pubs","Avg FWCI (FR)"])
        st.dataframe(
            authors,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Author": st.column_config.TextColumn("Author"),
                "Pubs": st.column_config.NumberColumn("Pubs", format="%.0f"),
                "Avg FWCI (FR)": st.column_config.NumberColumn("Avg FWCI (FR)", format="%.3f"),
                "Is UL?": st.column_config.TextColumn("Is UL?"),
                "Other UL lab(s)": st.column_config.TextColumn("Other UL lab(s)"),
            },
        )


# -------------------------- Two panels side by side --------------------------
colA, colB = st.columns(2)

render_lab_panel(colA, row1, unit1, df_f1, df_w1, df_yd1, ymax_year)
render_lab_panel(colB, row2, unit2, df_f2, df_w2, df_yd2, ymax_year)

st.divider()

# =========================== Inter-lab co-publications (from pubs_final.parquet) ===========================

st.subheader(f"Co-publications between **{unit1}** and **{unit2}** (2019–2023)")

if df_pubs is None:
    st.info("Inter-lab collaboration details unavailable because pubs_final.parquet could not be loaded.")
else:
    # exact schema
    col_year        = "Publication Year"
    col_openalex    = "OpenAlex ID"
    col_fwci        = "FWCI_FR"
    col_labs_rors   = "Labs_RORs"
    col_is_lue      = "In_LUE"
    col_top10       = "Is_PPtop10%_(field)"
    col_top1        = "Is_PPtop1%_(field)"
    col_subfield_id = "Primary Subfield ID"
    col_topic       = "Primary Topic"
    col_domain_id   = "Primary Domain ID"

    missing_core = [c for c in [col_year, col_labs_rors] if c not in df_pubs.columns]
    if missing_core:
        st.warning(f"Cannot compute co-publication stats — missing columns in pubs_final.parquet: {missing_core}")
    else:
        # ----- filter to period
        pubs = df_pubs.copy()
        pubs[col_year] = pd.to_numeric(pubs[col_year], errors="coerce")
        pubs = pubs[pubs[col_year].between(YEAR_START, YEAR_END, inclusive="both")]

        # normalize labs ROR list
        def to_ror_list(x):
            if isinstance(x, (list, tuple)):
                return [str(s).strip() for s in x]
            s = str(x)
            if not s or s.lower() in {"none", "nan"}:
                return []
            return [p.strip() for p in re.split(r"[|;,]", s) if p.strip()]

        ror1 = str(row1.get("ROR", "")).strip()
        ror2 = str(row2.get("ROR", "")).strip()

        pubs["_lab_rors"] = pubs[col_labs_rors].apply(to_ror_list)
        mask = pubs["_lab_rors"].apply(lambda lst: (ror1 in lst) and (ror2 in lst))
        copubs = pubs.loc[mask].copy()

        # ----- KPIs (removed LUE % as requested)
        n_copubs    = int(len(copubs))
        lue_count   = int(copubs[col_is_lue].fillna(False).astype(bool).sum()) if col_is_lue in copubs.columns else 0
        top10_count = int(copubs[col_top10].fillna(False).astype(bool).sum())  if col_top10  in copubs.columns else 0
        top1_count  = int(copubs[col_top1 ].fillna(False).astype(bool).sum())  if col_top1   in copubs.columns else 0
        avg_fwci    = float(pd.to_numeric(copubs[col_fwci], errors="coerce").mean()) if col_fwci in copubs.columns else float("nan")

        mk1, mk2, mk3, mk4 = st.columns(4)
        mk1.metric("Co-publications", f"{n_copubs:,}".replace(",", " "))
        mk2.metric("… incl. LUE (count)", f"{lue_count:,}".replace(",", " "))
        mk3.metric("… Top 10% (count)", f"{top10_count:,}".replace(",", " "))
        mk4.metric("… Top 1% (count)", f"{top1_count:,}".replace(",", " "))
        if not np.isnan(avg_fwci):
            st.metric("Average FWCI (FR) of co-pubs", f"{avg_fwci:.3f}")

        # ----- Legend (right after KPIs)
        looks = build_taxonomy_lookups()
        legend_items = "".join(
            f'<div class="legend-item"><span class="legend-swatch" style="background:{get_domain_color(d)};"></span>{d}</div>'
            for d in looks["domain_order"]
        )
        st.markdown(
            """
            <style>
            .legend-row { display:flex; gap:12px; align-items:center; margin: 6px 0 10px 2px; }
            .legend-item { display:flex; align-items:center; gap:6px; font-size: 0.95rem; color:#333; }
            .legend-swatch { display:inline-block; width:14px; height:14px; border-radius:3px; }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown(f'<div class="legend-row">{legend_items}</div>', unsafe_allow_html=True)

        # ----- determine domain per pub (prefer subfield→domain to avoid mislabels)
        def domain_from_row(row):
            if pd.notna(row.get(col_subfield_id, pd.NA)):
                try:
                    from lib.taxonomy import get_domain_for_subfield
                    return get_domain_for_subfield(row[col_subfield_id])
                except Exception:
                    pass
            if col_domain_id in row and pd.notna(row[col_domain_id]):
                # fallback: numeric domain id -> name via taxonomy
                return looks["id2name"].get(str(int(row[col_domain_id])), "Other")
            return "Other"

        copubs["_domain_name"] = copubs.apply(domain_from_row, axis=1)

        # ===== Yearly distribution stacked by domain (int ticks, no white grid lines) =====
        yr_dom = (copubs.dropna(subset=[col_year])
                        .groupby([col_year, "_domain_name"])
                        .size().reset_index(name="count"))
        years = list(range(YEAR_START, YEAR_END + 1))
        dom_order = looks["domain_order"]
        grid = pd.MultiIndex.from_product([years, dom_order], names=[col_year, "_domain_name"]).to_frame(index=False)
        yr_dom = grid.merge(yr_dom, on=[col_year, "_domain_name"], how="left").fillna({"count": 0})
        pivot = yr_dom.pivot(index=col_year, columns="_domain_name", values="count").fillna(0).reindex(years, fill_value=0)

        from matplotlib.ticker import MaxNLocator

        fig_y, ax_y = plt.subplots(figsize=(7.6, 3.2))
        bottoms = np.zeros(len(pivot))
        for dom in dom_order:
            vals = pivot.get(dom, pd.Series([0]*len(years), index=pivot.index)).astype(int).values
            ax_y.bar(pivot.index, vals, bottom=bottoms, color=get_domain_color(dom),
                     edgecolor="none", linewidth=0, antialiased=False, label=dom)
            bottoms += vals

        ax_y.set_title("Yearly co-publications by domain", fontsize=12, pad=6)
        ax_y.set_xlabel("Year", fontsize=11)
        ax_y.set_ylabel("Co-publications (count)", fontsize=11)
        ax_y.set_xticks(years)
        ax_y.set_xticklabels([str(y) for y in years], fontsize=10)
        ax_y.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax_y.grid(False)  # remove grid lines entirely
        for spine in ("top","right"):
            ax_y.spines[spine].set_visible(False)

        # center without stretching
        c1, c2, c3 = st.columns([1, 2.0, 1])
        with c2:
            st.pyplot(fig_y, use_container_width=False)

        # ===== Subfield counts (color by the subfield's domain; x ticks = ints) =====
        if col_subfield_id in copubs.columns:
            sub_counts = (copubs
                          .assign(_sub=pd.to_numeric(copubs[col_subfield_id], errors="coerce").astype("Int64"))
                          .dropna(subset=["_sub"]))
            sub_counts = (sub_counts.groupby("_sub")
                                      .size().reset_index(name="count")
                                      .sort_values("count", ascending=False))

            id2name = looks["id2name"]
            sub_counts["name"]  = sub_counts["_sub"].astype(str).map(id2name).fillna(sub_counts["_sub"].astype(str))
            # color by domain of the subfield
            sub_counts["color"] = sub_counts["_sub"].astype(str).apply(get_subfield_color)

            y = np.arange(len(sub_counts))
            fig_s, ax_s = plt.subplots(figsize=(7.6, max(2.0, 0.35 * len(sub_counts) + 0.6)))
            ax_s.barh(y, sub_counts["count"].astype(int).values,
                      color=sub_counts["color"].tolist(), edgecolor="none", alpha=0.95)
            ax_s.set_yticks(y)
            ax_s.set_yticklabels(sub_counts["name"], fontsize=10)
            ax_s.invert_yaxis()
            ax_s.set_xlabel("Co-publications (count)", fontsize=11)
            ax_s.set_title("Co-publications by primary subfield", fontsize=12, pad=6)
            ax_s.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax_s.grid(False)
            for spine in ("top","right","left"):
                ax_s.spines[spine].set_visible(False)

            c1, c2, c3 = st.columns([1, 2.0, 1])
            with c2:
                st.pyplot(fig_s, use_container_width=False)

        # ===== Word cloud of topics (NAMES, colored by DOMAIN) =====
        if col_topic in copubs.columns:
            topics_raw = copubs[col_topic].dropna().astype(str).tolist()
            from collections import Counter
            try:
                from wordcloud import WordCloud
            except Exception:
                WordCloud = None

            if topics_raw and WordCloud is not None:
                # map ID -> name via taxonomy
                try:
                    from lib.taxonomy import topic_id_to_name, get_topic_color
                    names = [topic_id_to_name(t) for t in topics_raw]
                    names = [n for n in names if n and n.lower() != "nan"]
                    freqs = Counter(names)
                    def wc_color_func(word, *args, **kwargs):
                        hexcol = get_topic_color(word)
                        h = hexcol.lstrip("#")
                        return (int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))
                except Exception:
                    # fallback: leave names as IDs, gray color
                    freqs = Counter(topics_raw)
                    def wc_color_func(word, *args, **kwargs):
                        return (127, 127, 127)

                if freqs:
                    wc = WordCloud(width=900, height=400, background_color="white", prefer_horizontal=0.95)
                    wc.generate_from_frequencies(freqs)
                    wc.recolor(color_func=wc_color_func)

                    fig_wc, ax_wc = plt.subplots(figsize=(8.0, 3.6))
                    ax_wc.imshow(wc, interpolation="bilinear")
                    ax_wc.axis("off")
                    ax_wc.set_title("Topics (size = frequency in co-pubs)", fontsize=12, pad=6)

                    c1, c2, c3 = st.columns([1, 2.0, 1])
                    with c2:
                        st.pyplot(fig_wc, use_container_width=False)
            else:
                st.info("No topics to display for wordcloud.")

        # ===== Download CSV of co-pubs =====
        st.download_button(
            "Download co-publications (CSV)",
            data=copubs.to_csv(index=False, encoding="utf-8-sig"),
            file_name=f"copubs_{unit1}_AND_{unit2}_{YEAR_START}-{YEAR_END}.csv",
            mime="text/csv",
        )
