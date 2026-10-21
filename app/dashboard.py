"""FloodCam Streamlit dashboard — reads the SQLite event log.

Milestones 6-7 will fill this in. This stub confirms the Streamlit entry
point is wired up correctly.
"""
import streamlit as st

st.set_page_config(
    page_title="FloodCam Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("FloodCam Monitor")
st.info(
    "Dashboard not yet implemented. "
    "Complete milestones 4-6 first, then run `make synth-demo` to generate "
    "a synthetic event log before returning here."
)

with st.sidebar:
    st.header("Camera")
    st.selectbox("Camera ID", ["(none)"])
    st.header("Run")
    st.selectbox("Select run", ["(none)"])
    st.slider("Replay speed", min_value=1, max_value=10, value=1, disabled=True)
