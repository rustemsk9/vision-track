import streamlit as st
import os
os.makedirs("static", exist_ok=True)
with open("static/test.html", "w") as f:
    f.write("<html><body>TEST HTML</body></html>")
st.components.v1.iframe("/app/static/test.html")
