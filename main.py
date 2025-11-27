import time
from collections import defaultdict
from datetime import datetime

import cv2
import pandas as pd
import streamlit as st
import torch
from ultralytics import YOLO

st.set_page_config(
    page_title="Weapon Detection System",
    page_icon="🔫",
    layout="wide",
    initial_sidebar_state="expanded"
)

with open("asset/style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Khởi tạo session state
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
    st.session_state.tracked_objects = {}  # {track_id: {'class': name, 'first_seen': time, 'counted': bool}}
if 'unique_objects' not in st.session_state:
    st.session_state.unique_objects = set()  # Set of counted track_ids


# Load model
@st.cache_resource
def load_model(model_path):
    try:
        yolo_model = YOLO(model_path)
        return yolo_model
    except Exception as e:
        st.error(f"Lỗi khi load model: {e}")
        return None


# Header
st.markdown('<h1 class="main-header">🔫 HỆ THỐNG PHÁT HIỆN VŨ KHÍ</h1>', unsafe_allow_html=True)

# Kiểm tra CUDA
cuda_available = torch.cuda.is_available()
device = 0 if cuda_available else 'cpu'

# Sidebar
with st.sidebar:
    st.header("⚙️ Cấu hình")

    # Hiển thị trạng thái GPU
    if cuda_available:
        st.success(f"✅ GPU khả dụng: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("⚠️ Đang sử dụng CPU (không phát hiện GPU)")

    model_path = st.text_input("Đường dẫn model", value="best.pt")
    confidence_threshold = st.slider("Ngưỡng tin cậy", 0.0, 1.0, 0.5, 0.05)
    camera_index = st.number_input("Camera Index", value=0, min_value=0)

    # Cấu hình video
    st.subheader("📹 Cấu hình Video")
    resolution = st.selectbox("Độ phân giải",
                              ["640x480 (Nhanh)", "1280x720 (Trung bình)", "1920x1080 (Chậm)"],
                              index=0)
    fps_limit = st.slider("Giới hạn FPS", 5, 30, 15, 5)

    # Cấu hình tracking
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
        "📌 **Hướng dẫn:**\n\n1. Cấu hình đường dẫn model\n2. Điều chỉnh ngưỡng tin cậy\n3. Nhấn 'Bắt đầu' để phát hiện")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Camera Trực Tiếp")
    video_placeholder = st.empty()

with col2:
    st.subheader("📊 Thống Kê Theo Thời Gian Thực")

    # Metrics
    metric_col1, metric_col2 = st.columns(2)
    total_metric = metric_col1.empty()
    unique_metric = metric_col2.empty()

    # Bảng đếm theo loại
    count_table = st.empty()

    # Cảnh báo
    st.subheader("⚠️ Cảnh Báo")
    alert_placeholder = st.empty()

# Bảng lịch sử
st.subheader("📋 Lịch Sử Phát Hiện")
history_placeholder = st.empty()

# Chạy detection
if st.session_state.running:
    model = load_model(model_path)

    if model is not None:
        # Parse resolution
        width, height = map(int, resolution.split('x')[0].split()[0]), \
            map(int, resolution.split('x')[1].split()[0])
        width = int(resolution.split('x')[0])
        height = int(resolution.split('x')[1].split()[0])

        # Sử dụng DirectShow backend trên Windows để tránh lỗi MSMF
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

        # Set cấu hình
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps_limit)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            st.error("❌ Không thể mở camera")
            st.session_state.running = False
        else:
            frame_count = 0
            last_process_time = time.time()
            frame_interval = 1.0 / fps_limit

            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Không đọc được frame từ camera. Đang thử kết nối lại...")
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    cap.set(cv2.CAP_PROP_FPS, fps_limit)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    continue

                # Kiểm tra thời gian để giới hạn FPS
                current_time = time.time()
                if current_time - last_process_time < frame_interval:
                    continue
                last_process_time = current_time

                # Chạy detection hoặc tracking
                if use_tracking:
                    # Sử dụng track() thay vì __call__() để có track_id
                    results = model.track(frame, conf=confidence_threshold, device=device,
                                          persist=True, tracker="bytetrack.yaml")
                else:
                    results = model(frame, conf=confidence_threshold, device=device, stream=True)

                detected_objects = []
                current_frame_tracks = set()  # Track IDs trong frame hiện tại

                for result in results:
                    annotated_frame = result.plot()

                    # Lấy thông tin detection
                    if result.boxes is not None and len(result.boxes) > 0:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]
                            confidence = float(box.conf[0])

                            # Lấy track_id nếu đang tracking
                            if use_tracking and hasattr(box, 'id') and box.id is not None:
                                track_id = int(box.id[0])
                                current_frame_tracks.add(track_id)

                                # Chỉ đếm nếu chưa từng đếm track_id này
                                if track_id not in st.session_state.unique_objects:
                                    st.session_state.unique_objects.add(track_id)
                                    st.session_state.detection_count[class_name] += 1
                                    st.session_state.total_detections += 1

                                    # Lưu thông tin tracked object
                                    st.session_state.tracked_objects[track_id] = {
                                        'class': class_name,
                                        'first_seen': datetime.now(),
                                        'last_seen': datetime.now(),
                                        'counted': True
                                    }

                                    # Thêm vào lịch sử
                                    st.session_state.detection_history.append({
                                        'Thời gian': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'Loại vũ khí': class_name,
                                        'Độ tin cậy': f"{confidence:.2%}",
                                        'Track ID': track_id
                                    })

                                    # Thêm cảnh báo
                                    st.session_state.alert_log.append({
                                        'time': datetime.now().strftime("%H:%M:%S"),
                                        'weapon': class_name,
                                        'confidence': confidence,
                                        'track_id': track_id
                                    })
                                else:
                                    # Cập nhật last_seen
                                    if track_id in st.session_state.tracked_objects:
                                        st.session_state.tracked_objects[track_id]['last_seen'] = datetime.now()

                            else:
                                # Không tracking - đếm mỗi detection (như cũ)
                                detected_objects.append({
                                    'class': class_name,
                                    'confidence': confidence,
                                    'time': datetime.now().strftime("%H:%M:%S")
                                })

                                st.session_state.detection_count[class_name] += 1
                                st.session_state.total_detections += 1

                                st.session_state.detection_history.append({
                                    'Thời gian': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'Loại vũ khí': class_name,
                                    'Độ tin cậy': f"{confidence:.2%}"
                                })

                                st.session_state.alert_log.append({
                                    'time': datetime.now().strftime("%H:%M:%S"),
                                    'weapon': class_name,
                                    'confidence': confidence
                                })

                    # Hiển thị video
                    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB", width="stretch")

                # Xóa các tracked objects cũ (quá thời gian persist)
                if use_tracking:
                    current_time_dt = datetime.now()
                    expired_tracks = []
                    for track_id, info in st.session_state.tracked_objects.items():
                        time_diff = (current_time_dt - info['last_seen']).total_seconds()
                        if time_diff > tracking_persist:
                            expired_tracks.append(track_id)

                    for track_id in expired_tracks:
                        del st.session_state.tracked_objects[track_id]
                        st.session_state.unique_objects.discard(track_id)

                # Cập nhật metrics
                total_metric.metric("Tổng phát hiện", st.session_state.total_detections)
                unique_metric.metric("Số loại vũ khí", len(st.session_state.detection_count))

                # Cập nhật bảng đếm
                if st.session_state.detection_count:
                    count_df = pd.DataFrame([
                        {'Loại vũ khí': k, 'Số lần xuất hiện': v}
                        for k, v in st.session_state.detection_count.items()
                    ]).sort_values('Số lần xuất hiện', ascending=False)
                    count_table.dataframe(count_df, width="stretch", hide_index=True)

                # Hiển thị cảnh báo
                if st.session_state.alert_log:
                    recent_alerts = st.session_state.alert_log[-5:]  # 5 cảnh báo gần nhất
                    alert_text = ""
                    for alert in reversed(recent_alerts):
                        alert_text += f"🚨 **{alert['time']}** - Phát hiện **{alert['weapon']}** ({alert['confidence']:.2%})\n\n"
                    alert_placeholder.markdown(f'<div class="alert-box">{alert_text}</div>', unsafe_allow_html=True)

                # Hiển thị lịch sử
                if st.session_state.detection_history:
                    history_df = pd.DataFrame(st.session_state.detection_history[-20:])  # 20 phát hiện gần nhất
                    history_placeholder.dataframe(history_df, width="stretch", hide_index=True)

                frame_count += 1

                # Không cần sleep nữa vì đã có frame_interval
                # time.sleep(0.01)

            cap.release()
    else:
        st.error("❌ Không thể load model. Vui lòng kiểm tra đường dẫn.")
        st.session_state.running = False
else:
    video_placeholder.info("📷 Nhấn 'Bắt đầu' để bật camera và phát hiện vũ khí")

    # Hiển thị dữ liệu đã có (nếu có)
    if st.session_state.total_detections > 0:
        total_metric.metric("Tổng phát hiện", st.session_state.total_detections)
        unique_metric.metric("Số loại vũ khí", len(st.session_state.detection_count))

        if st.session_state.detection_count:
            count_df = pd.DataFrame([
                {'Loại vũ khí': k, 'Số lần xuất hiện': v}
                for k, v in st.session_state.detection_count.items()
            ]).sort_values('Số lần xuất hiện', ascending=False)
            count_table.dataframe(count_df, width="stretch", hide_index=True)

        if st.session_state.detection_history:
            history_df = pd.DataFrame(st.session_state.detection_history[-20:])
            history_placeholder.dataframe(history_df, width="stretch", hide_index=True)
