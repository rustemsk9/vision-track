import streamlit as st
import os

os.makedirs("static", exist_ok=True)
with open("static/test2.html", "w") as f:
    f.write("<html><body>TEST HTML</body></html>")

st.markdown('<iframe src="/app/static/test2.html" width="100%" height="550px" allow="camera; microphone" style="border:none; border-radius:8px;"></iframe>', unsafe_allow_html=True)
