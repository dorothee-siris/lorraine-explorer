from __future__ import annotations
import pandas as pd
import streamlit as st
from lib.taxonomy import build_taxonomy_lookups, get_domain_color, canonical_field_order, canonical_subfield_order

st.set_page_config(page_title="🔬 Topic View", layout="wide")
st.title("🔬 Topic View")

look = build_taxonomy_lookups()  # auto-reads data/all_topics.parquet
st.write("Domain order:", look["domain_order"])
st.write("First 10 canonical fields:", canonical_field_order()[:10])

# quick demo: show domain colors
cols = st.columns(len(look["domain_order"]))
for c, d in zip(cols, look["domain_order"]):
    with c:
        st.markdown(f"<div style='padding:10px;background:{get_domain_color(d)};color:#000'>{d}</div>", unsafe_allow_html=True)
