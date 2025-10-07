# pages/1_🏭_Lab_View.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Your taxonomy helpers
from lib.taxonomy import (
    build_taxonomy_lookups,
    canonical_field_order,
    get_field_color,
    field_id_to_name,          # <-- optional helper added above; if not added, replace calls below
    get_domain_for_field,      # <-- optional helper added above; if not added, derive from lookups
)

# ------------------------ paths & caching ------------------------

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "ul_units_indicators.parquet"

@st.cache_data(show_spinner=False)
def load_units() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)

    # Normalize column names we rely on (be lenient with casing/spaces)
    rename_map = {c: c.strip() for c in df.columns}
    df = df.rename(columns=rename_map)

    # Expected columns (exact preferred names used below):
    # "Unit Name", "Type", "Pubs", "By field: counts", "By field: LUE counts"
    # If your file uses slightly different casing, fix here:
    canonical = {
        "unit name": "Unit Name",
        "type": "Type",
        "pubs": "Pubs",
        "by field: counts": "By field: counts",
        "by field: lue counts": "By field: LUE counts",
    }
    lower2actual = {c.lower(): c for c in df.columns}
    for low, want in canonical.items():
        if low in lower2actual and lower2actual[low] != want:
            df = df.rename(columns={lower2actual[low]: want})

    # Ensure numeric Pubs
    df["Pubs"] = pd.to_numeric(df["Pubs"], errors="coerce").fillna(0).astype(int)

    return df

@st.cache_data(show_spinner=False)
def get_lookups() -> Dict:
    return build_taxonomy_lookups()

# ------------------------ parsing utilities ------------------------

FIELD_PAIR_RE = re.compile(r"^\s*(.*?)\s*\(([^)]*)\)\s*$")

def _to_int_safe(s) -> int:
    try:
        # keep only leading number (handles "5" or "5 ..." or "5,0")
        num = re.match(r"^\s*([0-9]+)", str(s))
        return int(num.group(1)) if num else 0
    except Exception:
        return 0

def parse_field_counts_blob(blob: str, look: Dict) -> pd.DataFrame:
    """
    Parse strings like: "11 (5) | 13 (27) | Field Name (4)"
    Returns DataFrame with columns: ["field_name","count"]
    Unknown fields (not in taxonomy) are dropped.
    """
    if pd.isna(blob) or not str(blob).strip():
        return pd.DataFrame(columns=["field_name", "count"])

    parts = [p.strip() for p in str(blob).split("|")]
    rows: List[Tuple[str, int]] = []
    canon_fields = set(canonical_field_order())

    for p in parts:
        if not p:
            continue
        m = FIELD_PAIR_RE.match(p)
        if not m:
            continue
        token = m.group(1).strip()              # could be field id or field name
        count = _to_int_safe(m.group(2))

        # normalize token -> field_name
        if token.isdigit():
            field_name = look["id2name"].get(token, token)
        else:
            field_name = token

        # keep only real fields from taxonomy (avoid stray tokens)
        if field_name in canon_fields:
            rows.append((field_name, count))

    return pd.DataFrame(rows, columns=["field_name", "count"])

# ------------------------ color helpers ------------------------

def darken_hex(hex_color: str, factor: float = 0.75) -> str:
    """
    Darken a hex color by multiplying each channel by factor (0..1).
    factor=0.75 is a nice subtle darkening.
    """
    h = hex_color.lstrip("#")
    if len(h) == 3:  # expand short form
        h = "".join([c*2 for c in h])
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = max(0, min(255, int(r * factor)))
    g = max(0, min(255, int(g * factor)))
    b = max(0, min(255, int(b * factor)))
    return f"#{r:02x}{g:02x}{b:02x}"

# ------------------------ shaping for plotting ------------------------

def build_fields_table(row: pd.Series, look: Dict) -> pd.DataFrame:
    """
    For a given unit (row from df_units), build a table with:
      field_name, count, share, lue_count, lue_share, color, color_lue, domain
    """
    pubs = int(row.get("Pubs", 0)) or 1  # avoid /0; if truly 0 pubs, %s will be 0 anyway

    df_counts = parse_field_counts_blob(row.get("By field: counts", ""), look)
    df_lue    = parse_field_counts_blob(row.get("By field: LUE counts", ""), look)

    # merge and compute shares
    df = pd.merge(df_counts, df_lue, on="field_name", how="outer", suffixes=("", "_lue")).fillna(0)
    df["count"] = df["count"].astype(int)
    df["count_lue"] = df["count_lue"].astype(int)
    if "count_lue" not in df.columns:
        df["count_lue"] = 0

    df["share"]     = (df["count"] / pubs).astype(float)
    df["lue_share"] = (df["count_lue"] / pubs).astype(float)

    # add ordering + colors
    order = {name: i for i, name in enumerate(canonical_field_order())}
    df["__ord"] = df["field_name"].map(order).fillna(10_000).astype(int)

    # domain & colors
    df["domain"]    = df["field_name"].apply(get_domain_for_field)
    df["color"]     = df["field_name"].apply(get_field_color)
    df["color_lue"] = df["color"].apply(lambda c: darken_hex(c, 0.65))

    # sort by canonical order (domain-grouped alphabetical)
    df = df.sort_values("__ord").drop(columns="__ord").reset_index(drop=True)
    return df

def union_ordered_fields(df1: pd.DataFrame, df2: pd.DataFrame) -> List[str]:
    """
    Union of field names present in either df, ordered by canonical_field_order.
    """
    present = set(df1["field_name"]).union(set(df2["field_name"]))
    order   = canonical_field_order()
    return [f for f in order if f in present]

# ------------------------ plotting ------------------------

def plot_unit_fields_barh(df_fields: pd.DataFrame,
                          all_fields_order: List[str],
                          share_max: float,
                          title: str,
                          show_counts_gutter: bool = True) -> plt.Figure:
    """
    Build a horizontal bar + LUE overlay chart for one unit, aligned to a shared y order.
    """
    # Reindex to the union order, introducing missing fields as zeros
    base = pd.DataFrame({"field_name": all_fields_order})
    df = base.merge(df_fields, on="field_name", how="left").fillna({
        "count": 0, "share": 0.0, "count_lue": 0, "lue_share": 0.0,
        "color": "#7f7f7f", "color_lue": "#5a5a5a", "domain": "Other",
    })


    y = np.arange(len(df))
    heights = 0.8  # bar thickness

    fig_h = max(1.0, 0.3 * len(df) + 0.8)
    fig, ax = plt.subplots(figsize=(7.2, fig_h))

    # Left gutter for counts (fixed pixels translated to data coords)
    left_pad_px = 72 if show_counts_gutter else 0
    offset_px   = 6 
    ax.set_xlim(0, share_max)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb = ax.get_window_extent(renderer=renderer)
    ax_width_px = bb.width if bb.width > 0 else 600
    data_per_px = (share_max - 0.0) / ax_width_px if ax_width_px else 0.0001
    left_pad_data = left_pad_px * data_per_px
    offset_data   = offset_px * data_per_px
    ax.set_xlim(-left_pad_data, share_max)

    # Base bars (total share)
    for i, row in df.iterrows():
        ax.barh(y[i], width=row["share"], left=0.0, height=heights,
                edgecolor="none", color=row["color"], alpha=0.95, zorder=2)

        # LUE overlay
        if row["lue_share"] > 0:
            ax.barh(y[i], width=row["lue_share"], left=0.0, height=heights*0.7,
                    edgecolor="none", color=row["color_lue"], alpha=1.0, zorder=3)

    # Counts gutter text
    if show_counts_gutter:
        for yi, cnt in enumerate(df["count"].astype(int).tolist()):
            ax.text(-left_pad_data + offset_data, yi, f"{cnt:,}".replace(",", " "),
                    va="center", ha="left", fontsize=9, color="#444")

    # Axis cosmetics
    ax.set_title(title, fontsize=12, pad=6)
    ax.set_yticks(y)
    ax.set_yticklabels(df["field_name"], fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#eeeeee")
    ax.set_axisbelow(True)

    # x ticks every 5%
    max_pct = share_max
    # ceiling to nearest 5%
    max_tick = np.ceil(max_pct * 20) / 20.0  # 1/20 = 5%
    xticks = np.arange(0.0, max(0.05, max_tick) + 1e-9, 0.05)
    xticks = xticks[xticks <= share_max + 1e-9]  # keep within axis
    ax.set_xticks(xticks)
    ax.set_xlabel("% of unit publications", fontsize=10)
    ax.set_xlim(-left_pad_data, share_max)

    # Format x tick labels as percentages (0%, 5%, 10%, ...)
    ax.set_xticklabels([f"{int(x*100)}%" for x in xticks], fontsize=9)

    # Clean spines
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    return fig

# ------------------------ Streamlit UI ------------------------

st.set_page_config(page_title="Lab view · Fields", layout="wide")
st.title("🏭 Lab view — Field distribution")

look = get_lookups()
df_units = load_units()

# Filter labs
labs = df_units.loc[df_units["Type"].astype(str).str.lower() == "lab"].copy()
lab_names = labs["Unit Name"].dropna().astype(str).sort_values().tolist()
if len(lab_names) == 0:
    st.error("No units of Type = 'lab' found in ul_units_indicators.parquet.")
    st.stop()

csel1, csel2 = st.columns(2)
with csel1:
    unit1 = st.selectbox("Select unit A (lab)", lab_names, index=0, key="unit_a")
with csel2:
    # pick a different default if available
    default_idx = 1 if len(lab_names) > 1 else 0
    unit2 = st.selectbox("Select unit B (lab)", lab_names, index=default_idx, key="unit_b")

row1 = labs.loc[labs["Unit Name"] == unit1].iloc[0]
row2 = labs.loc[labs["Unit Name"] == unit2].iloc[0]

df_f1 = build_fields_table(row1, look)
df_f2 = build_fields_table(row2, look)

# Union of fields present across both, ordered canonically
fields_union = union_ordered_fields(df_f1, df_f2)

# Shared x scale: highest % in any field across both labs
share_max = float(
    max(
        (df_f1["share"].max() if not df_f1.empty else 0.0),
        (df_f2["share"].max() if not df_f2.empty else 0.0),
        0.0,
    )
)
# If no pubs at all, avoid a 0-width axis
if share_max <= 0:
    share_max = 0.05  # show a tiny scale so the plot has axes

c1, c2 = st.columns(2)
with c1:
    fig1 = plot_unit_fields_barh(
        df_fields=df_f1,
        all_fields_order=fields_union,
        share_max=share_max,
        title=f"{unit1} — field distribution",
    )
    st.pyplot(fig1, use_container_width=True)

with c2:
    fig2 = plot_unit_fields_barh(
        df_fields=df_f2,
        all_fields_order=fields_union,
        share_max=share_max,
        title=f"{unit2} — field distribution",
    )
    st.pyplot(fig2, use_container_width=True)

# Optional downloads for the two tables
exp = st.expander("Download data")
with exp:
    dl1 = df_f1[["field_name","domain","count","share","count_lue","lue_share"]].copy()
    dl2 = df_f2[["field_name","domain","count","share","count_lue","lue_share"]].copy()
    st.download_button(
        "Download Unit A fields (CSV)",
        data=dl1.to_csv(index=False, encoding="utf-8-sig"),
        file_name=f"{unit1}_fields.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download Unit B fields (CSV)",
        data=dl2.to_csv(index=False, encoding="utf-8-sig"),
        file_name=f"{unit2}_fields.csv",
        mime="text/csv",
    )
