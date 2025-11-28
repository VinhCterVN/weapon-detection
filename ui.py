from collections import defaultdict

import streamlit as st


def init_session_state():
    if 'detection_count' not in st.session_state:
        st.session_state.detection_count = defaultdict(int)
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
    if 'total_detections' not in st.session_state:
        st.session_state.total_detections = 0
    if 'alert_log' not in st.session_state:
        st.session_state.alert_log = []
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'tracked_objects' not in st.session_state:
        st.session_state.tracked_objects = {}
    if 'unique_objects' not in st.session_state:
        st.session_state.unique_objects = set()


def load_css(path="asset/style.css"):
    try:
        with open(path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception:
        # silently continue if missing
        pass


def sidebar_config():
    import torch
    st.set_page_config(
        page_title="Weapon Detection System",
        page_icon="🔫",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with st.sidebar:
        st.header("⚙️ Cấu hình")
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            st.success(f"✅ GPU khả dụng: {torch.cuda.get_device_name(0)}")
        else:
            st.warning("⚠️ Đang sử dụng CPU (không phát hiện GPU)")

        model_path = st.text_input("Đường dẫn model", value="best.pt")
        confidence_threshold = st.slider("Ngưỡng tin cậy", 0.0, 1.0, 0.5, 0.05)
        camera_index = st.number_input("Camera Index", value=0, min_value=0)

        st.subheader("📹 Cấu hình Video")
        resolution = st.selectbox("Độ phân giải",
                                  ["640x480 (Nhanh)", "1280x720 (Trung bình)", "1920x1080 (Chậm)"],
                                  index=0)
        fps_limit = st.slider("Giới hạn FPS", 5, 30, 15, 5)

        st.subheader("🎯 Cấu hình Tracking")
        use_tracking = st.checkbox("Bật tracking (đếm vật thể duy nhất)", value=True)
        tracking_persist = st.slider("Thời gian nhớ vật thể (giây)", 1, 10, 3, 1)

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Bắt đầu", width="stretch"):
                st.session_state.running = True
        with col2:
            if st.button("⏸️ Dừng", width="stretch"):
                st.session_state.running = False

        if st.button("🔄 Reset dữ liệu", width="stretch"):
            st.session_state.detection_count = defaultdict(int)
            st.session_state.detection_history = []
            st.session_state.total_detections = 0
            st.session_state.alert_log = []
            st.session_state.tracked_objects = {}
            st.session_state.unique_objects = set()
            st.rerun()

        st.divider()
        st.info(
            "📌 **Hướng dẫn:**\n\n1. Cấu hình đường dẫn model\n2. Điều chỉnh ngưỡng tin cậy\n3. Nhấn 'Bắt đầu' để phát hiện"
        )

    # return config
    return {
        "model_path": model_path,
        "confidence_threshold": confidence_threshold,
        "camera_index": camera_index,
        "resolution": resolution,
        "fps_limit": fps_limit,
        "use_tracking": use_tracking,
        "tracking_persist": tracking_persist,
        "cuda_available": cuda_available
    }


def create_placeholders():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📹 Camera Trực Tiếp")
        video_placeholder = st.empty()

    with col2:
        st.subheader("📊 Thống Kê Theo Thời Gian Thực")
        metric_col1, metric_col2 = st.columns(2)
        total_metric = metric_col1.empty()
        unique_metric = metric_col2.empty()
        count_table = st.empty()
        st.subheader("⚠️ Cảnh Báo")
        alert_placeholder = st.empty()

    st.subheader("📋 Lịch Sử Phát Hiện")
    history_placeholder = st.empty()

    return {
        "video": video_placeholder,
        "total_metric": total_metric,
        "unique_metric": unique_metric,
        "count_table": count_table,
        "alert": alert_placeholder,
        "history": history_placeholder
    }
