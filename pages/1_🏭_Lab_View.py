# pages/1_🏭_Lab_View.py

import streamlit as st
st.set_page_config(page_title="Lab view (placeholder)", page_icon="🏭", layout="wide")
st.title("🏭 Lab_View (placeholder)")
st.info("This view will surface lab collaborations.")

from lib.taxonomy import build_taxonomy_lookups, get_domain_color, canonical_field_order, canonical_subfield_order
