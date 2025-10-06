from __future__ import annotations
import streamlit as st

st.set_page_config(page_title="Lorraine Explorer", layout="wide")
st.title("Lorraine Explorer")

st.write("Use the sidebar to open a view:")
st.page_link("pages/1_🏭_Lab_View.py", label="🏭 Lab View")
st.page_link("pages/2_🔬_Topic_View.py", label="🔬 Topic View")
st.page_link("pages/3_🤝_Partners_View.py", label="🤝 Partners View")