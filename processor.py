import time
from datetime import datetime

import cv2
import pandas as pd
import streamlit as st


def _parse_resolution(resolution_str):
    token = resolution_str.split()[0]  # e.g. "640x480"
    w, h = map(int, token.split('x'))
    return w, h


def run_detection(model, config, placeholders):
    device = 0 if config["cuda_available"] else 'cpu'
    confidence_threshold = config["confidence_threshold"]
    camera_index = config["camera_index"]
    resolution = config["resolution"]
    fps_limit = config["fps_limit"]
    use_tracking = config["use_tracking"]
    tracking_persist = config["tracking_persist"]

    width, height = _parse_resolution(resolution)

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps_limit)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        st.error("❌ Không thể mở camera")
        st.session_state.running = False
        return

    last_process_time = time.time()
    frame_interval = 1.0 / max(1, fps_limit)

    try:
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

            current_time = time.time()
            if current_time - last_process_time < frame_interval:
                continue
            last_process_time = current_time

            if use_tracking:
                results = model.track(frame, conf=confidence_threshold, device=device,
                                      persist=True, tracker="bytetrack.yaml")
            else:
                results = model(frame, conf=confidence_threshold, device=device, stream=True)

            detected_objects = []
            current_frame_tracks = set()

            for result in results:
                annotated_frame = result.plot()

                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0])

                        if use_tracking and hasattr(box, 'id') and box.id is not None:
                            track_id = int(box.id[0])
                            current_frame_tracks.add(track_id)

                            if track_id not in st.session_state.unique_objects:
                                st.session_state.unique_objects.add(track_id)
                                st.session_state.detection_count[class_name] += 1
                                st.session_state.total_detections += 1

                                st.session_state.tracked_objects[track_id] = {
                                    'class': class_name,
                                    'first_seen': datetime.now(),
                                    'last_seen': datetime.now(),
                                    'counted': True
                                }

                                st.session_state.detection_history.append({
                                    'Thời gian': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'Loại vũ khí': class_name,
                                    'Độ tin cậy': f"{confidence:.2%}",
                                    'Track ID': track_id
                                })

                                st.session_state.alert_log.append({
                                    'time': datetime.now().strftime("%H:%M:%S"),
                                    'weapon': class_name,
                                    'confidence': confidence,
                                    'track_id': track_id
                                })
                            else:
                                if track_id in st.session_state.tracked_objects:
                                    st.session_state.tracked_objects[track_id]['last_seen'] = datetime.now()
                        else:
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

                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                placeholders["video"].image(frame_rgb, channels="RGB", width="stretch")

            # expire old tracks
            if use_tracking:
                now_dt = datetime.now()
                expired = [tid for tid, info in st.session_state.tracked_objects.items()
                           if (now_dt - info['last_seen']).total_seconds() > tracking_persist]
                for tid in expired:
                    del st.session_state.tracked_objects[tid]
                    st.session_state.unique_objects.discard(tid)

            # update metrics / UI
            placeholders["total_metric"].metric("Tổng phát hiện", st.session_state.total_detections)
            placeholders["unique_metric"].metric("Số loại vũ khí", len(st.session_state.detection_count))

            if st.session_state.detection_count:
                count_df = pd.DataFrame([
                    {'Loại vũ khí': k, 'Số lần xuất hiện': v}
                    for k, v in st.session_state.detection_count.items()
                ]).sort_values('Số lần xuất hiện', ascending=False)
                placeholders["count_table"].dataframe(count_df, width="stretch", hide_index=True)

            if st.session_state.alert_log:
                recent_alerts = st.session_state.alert_log[-5:]
                alert_text = ""
                for alert in reversed(recent_alerts):
                    alert_text += f"🚨 **{alert['time']}** - Phát hiện **{alert['weapon']}** ({alert['confidence']:.2%})\n\n"
                placeholders["alert"].markdown(f'<div class="alert-box">{alert_text}</div>', unsafe_allow_html=True)

            if st.session_state.detection_history:
                history_df = pd.DataFrame(st.session_state.detection_history[-20:])
                placeholders["history"].dataframe(history_df, width="stretch", hide_index=True)

    finally:
        cap.release()
