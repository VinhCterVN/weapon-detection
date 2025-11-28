import streamlit as st
from ultralytics import YOLO


@st.cache_resource
def load_model(model_path: str):
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Lỗi khi load model: {e}")
        return None
