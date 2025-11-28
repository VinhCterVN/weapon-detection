import streamlit as st

from model_loader import load_model
from processor import run_detection
from ui import init_session_state, load_css, sidebar_config, create_placeholders


def main():
    init_session_state()
    load_css()
    st.markdown('<h1 class="main-header">🔫 HỆ THỐNG PHÁT HIỆN VŨ KHÍ</h1>', unsafe_allow_html=True)

    config = sidebar_config()
    placeholders = create_placeholders()

    if st.session_state.running:
        model = load_model(config["model_path"])
        if model is None:
            st.error("❌ Không thể load model. Vui lòng kiểm tra đường dẫn.")
            st.session_state.running = False
            return
        run_detection(model, config, placeholders)
    else:
        placeholders["video"].info("📷 Nhấn 'Bắt đầu' để bật camera và phát hiện vũ khí")
        # show stored stats when idle
        if st.session_state.total_detections > 0:
            placeholders["total_metric"].metric("Tổng phát hiện", st.session_state.total_detections)
            placeholders["unique_metric"].metric("Số loại vũ khí", len(st.session_state.detection_count))
            if st.session_state.detection_count:
                import pandas as pd
                count_df = pd.DataFrame([
                    {'Loại vũ khí': k, 'Số lần xuất hiện': v}
                    for k, v in st.session_state.detection_count.items()
                ]).sort_values('Số lần xuất hiện', ascending=False)
                placeholders["count_table"].dataframe(count_df, width="stretch", hide_index=True)
            if st.session_state.detection_history:
                import pandas as pd
                history_df = pd.DataFrame(st.session_state.detection_history[-20:])
                placeholders["history"].dataframe(history_df, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
