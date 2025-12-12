"""
AI MoodSense - Streamlit Web Application
Student Facial Emotion Classification Web Application
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import sys
import io
import pandas as pd

# Add src to path
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
sys.path.append(os.path.join(BASE_DIR, 'src'))

from model_utils import EmotionPredictor, EMOJI_MAP

# Page configuration with custom favicon if provided
favicon_path = os.path.join(DATA_DIR, "logo_geargen.ico")
page_icon_value = favicon_path if os.path.exists(favicon_path) else "😊"
st.set_page_config(
    page_title="AI MoodSense - Emotion Recognition",
    page_icon=page_icon_value,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved styling
st.markdown("""
    <style>
    /* Main header styling */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    
    /* Sub header styling */
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    
    /* Logo container */
    .logo-container {
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .logo-text {
        font-size: 2rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    /* Prediction box styling */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: white;
    }
    
    .emotion-display {
        font-size: 2.5rem;
        text-align: center;
        padding: 1rem;
        color: white;
        font-weight: bold;
    }
    
    .confidence-text {
        text-align: center;
        font-size: 1.3rem;
        color: white;
        margin-top: 1rem;
    }
    
    /* Fix for subheader text color */
    .stSubheader {
        color: #262730 !important;
    }
    
    h3 {
        color: #262730 !important;
    }
    
    /* Section headers */
    .section-header {
        color: #262730;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Probability labels */
    .prob-label {
        color: #262730;
        font-weight: 500;
    }
    
    /* Face box labels */
    .face-label {
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
        color: #fff;
        display: inline-block;
        margin-bottom: 6px;
    }

    /* Face box labels */
    .face-label {
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
        color: #fff;
        display: inline-block;
        margin-bottom: 6px;
    }
    
    body {
        background: #f7f9fb;
    }
    .stTabs [role="tab"] {
        padding: 0.75rem 1.25rem;
        font-weight: 600;
    }
    .stTabs [role="tablist"] {
        gap: 0.5rem;
    }
    .prediction-box {
        border: 1px solid rgba(255,255,255,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    try:
        st.session_state.predictor = EmotionPredictor()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

if 'camera_image' not in st.session_state:
    st.session_state.camera_image = None

if 'enable_camera' not in st.session_state:
    st.session_state.enable_camera = False

def load_logo():
    preferred = os.path.join(DATA_DIR, "logo_geargen.png")
    fallback = os.path.join(DATA_DIR, "logo.png")
    if os.path.exists(preferred):
        return preferred
    if os.path.exists(fallback):
        return fallback
    return None

def draw_boxes(image_array, detections):
    """Draw bounding boxes with labels on the image."""
    color_map = {
        'Positive': (46, 204, 113),
        'Neutral': (241, 196, 15),
        'Negative': (231, 76, 60)
    }
    img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR) if len(image_array.shape) == 3 and image_array.shape[2] == 3 else image_array.copy()

    for det in detections:
        (x, y, w, h) = det['box']
        label = f"{det['emotion']} ({det['confidence']*100:.1f}%)"
        color = color_map.get(det['emotion'], (52, 152, 219))
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, 2)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_bgr, (x, y - text_h - 10), (x + text_w + 6, y), color, -1)
        cv2.putText(img_bgr, label, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if len(img_bgr.shape) == 3 else img_bgr

def analyze_image(image_array, detect_face, multi_face, debug_mode):
    predictor = st.session_state.predictor
    results = []

    if multi_face:
        detections = predictor.detect_faces(image_array)
        if detections:
            for face_img, box in detections:
                pred = predictor.predict(face_img, detect_face=False, debug=debug_mode)
                if 'error' in pred:
                    continue
                results.append({
                    'box': box,
                    'emotion': pred['emotion'],
                    'emoji': pred['emoji'],
                    'confidence': pred['confidence'],
                    'probabilities': pred['probabilities'],
                    'raw': pred.get('raw_probabilities')
                })
            if results:
                return results

    # Fallback to single prediction using DeepFace detection
    pred = predictor.predict(image_array, detect_face=detect_face, debug=debug_mode)
    if 'error' in pred:
        return []
    h, w = image_array.shape[:2]
    results.append({
        'box': (0, 0, w, h),
        'emotion': pred['emotion'],
        'emoji': pred['emoji'],
        'confidence': pred['confidence'],
        'probabilities': pred['probabilities'],
        'raw': pred.get('raw_probabilities')
    })
    return results

def detections_to_csv_bytes(detections, source_label):
    rows = []
    for idx, det in enumerate(detections, start=1):
        rows.append({
            "source": source_label,
            "face": idx,
            "emotion": det['emotion'],
            "confidence": round(det['confidence'] * 100, 2)
        })
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

# Header with Logo (left-aligned)
logo_path = load_logo()
if logo_path:
    st.image(logo_path, width=220)
else:
    st.markdown("""
        <div class="logo-container">
            <div class="logo-text">⚙️ GearGen</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="main-header" style="margin-top: -0.5rem;">😊 AI MoodSense</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ứng dụng Phân loại cảm xúc dành cho học sinh GearGen</p>', unsafe_allow_html=True)

st.info("Mẹo: Hãy tải lên hình ảnh rõ nét, khuôn mặt nhìn thẳng. Bật “Detect multiple faces” khi chụp ảnh nhóm. Sau khi dự đoán, bạn có thể tải kết quả theo từng khuôn mặt dưới dạng CSV.")

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <div style="font-size: 1.5rem; font-weight: bold; color: #1f77b4;">⚙️ GearGen</div>
        <div style="color: #666; font-size: 0.9rem;">AI Solutions</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.header("📋 Thông tin sản phẩm")
    st.markdown("""
    **AI MoodSense** sử dụng Computer Vision và Deep Learning nhận dạng cảm xúc dựa trên khuôn mặt.
    
    **Phân loại cảm xúc:**
    - 😊 **Tích cực**: Happy, Surprise
    - 😐 **Trung lập**: Neutral
    - 😞 **Tiêu cực**: Sad, Angry, Fear, Disgust
    
    **Hướng dẫn sử dụng:**
    1. Tải 01 hình ảnh từ máy tính hoặc chụp ảnh từ webcam
    2. Mô hình AI sẽ phân tích khuôn mặt
    3. Hiển thị các chỉ số cảm xúc
    """)
    
    st.header("⚙️ Cài đặt")
    detect_face = st.checkbox("Auto-detect face", value=True, help="Automatically detect and crop face from image before analysis")
    multi_face = st.checkbox("Detect multiple faces", value=True, help="Show bounding boxes and predict each detected face")
    debug_mode = st.checkbox("Debug mode", value=False, help="Show raw DeepFace predictions for debugging")
    
    st.header("📊 Thông tin mô hình")
    st.info("Model: DeepFace (Pre-trained)\nBackend: VGG-Face\nEmotions: 7 → 3 groups\nAccuracy: ~97% (emotion)")

# Main content
tab1, tab2 = st.tabs(["📷 Upload Image", "🎥 Webcam Capture"])

with tab1:
    st.header("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a face image for emotion analysis"
    )
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        col1, col2 = st.columns(2)
        try:
            with st.spinner("Đang phân tích khuôn mặt..."):
                detections = analyze_image(image_array, detect_face, multi_face, debug_mode)
            
            if not detections:
                st.error("No face detected. Try disabling auto-detect or use a clearer image.")
            else:
                annotated = draw_boxes(image_array, detections) if multi_face else image_array
                with col1:
                    st.subheader("Đã phát hiện khuôn mặt")
                    st.image(annotated, use_container_width=True)

                with col2:
                    st.subheader("Prediction Results")
                    for idx, det in enumerate(detections, start=1):
                        emotion = det['emotion']
                        emoji = det['emoji']
                        confidence = det['confidence']
                        st.markdown(f"""
                        <div class="prediction-box">
                            <div class="emotion-display">
                                {emotion} {'(Face ' + str(idx) + ')' if multi_face and len(detections) > 1 else ''}
                            </div>
                            <div class="confidence-text">
                                Confidence: <strong>{confidence*100:.1f}%</strong>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(float(confidence))

                        st.markdown("<h3 class='section-header'>3-Class Probability Distribution</h3>", unsafe_allow_html=True)
                        for emotion_label, prob in det['probabilities'].items():
                            emoji_icon = EMOJI_MAP.get(emotion_label, '')
                            st.markdown(f"<div class='prob-label'>{emoji_icon} <strong>{emotion_label}</strong>: {float(prob)*100:.1f}%</div>", unsafe_allow_html=True)
                            st.progress(float(prob))

                        if debug_mode and det['raw']:
                            st.markdown("<h3 class='section-header'>Raw DeepFace Output (Debug)</h3>", unsafe_allow_html=True)
                            for emotion_label, prob in det['raw'].items():
                                st.markdown(f"<div class='prob-label' style='color: #888;'>{emotion_label.capitalize()}: {prob:.1f}%</div>", unsafe_allow_html=True)
                            st.markdown("---")
                    
                    # Download detections as CSV
                    csv_bytes = detections_to_csv_bytes(detections, "upload")
                    st.download_button(
                        label="⬇️ Download results (CSV)",
                        data=csv_bytes,
                        file_name="moodsense_results.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

with tab2:
    st.header("Webcam Capture")
    
    # Button to enable camera
    col_btn1, col_btn2 = st.columns([1, 2])
    with col_btn1:
        if st.button("📷 Enable Camera", type="primary", use_container_width=True):
            st.session_state.enable_camera = True
            st.rerun()
    
    with col_btn2:
        if st.button("❌ Disable Camera", use_container_width=True):
            st.session_state.enable_camera = False
            st.session_state.camera_image = None
            st.rerun()
    
    # Only show camera if enabled
    if st.session_state.enable_camera:
        st.info("📷 Camera is active. Position your face and click the camera button below to capture.")
        camera_image = st.camera_input("Take a picture", key="webcam_capture")
        
        if camera_image is not None:
            # Read image
            image = Image.open(camera_image)
            image_array = np.array(image)
            
            # Store in session state
            st.session_state.camera_image = image_array
            
            col1, col2 = st.columns(2)

            try:
                with st.spinner("Analyzing faces..."):
                    detections = analyze_image(image_array, detect_face, multi_face, debug_mode)

                if not detections:
                    st.error("No face detected. Try disabling auto-detect or adjust lighting.")
                else:
                    annotated = draw_boxes(image_array, detections) if multi_face else image_array
                    with col1:
                        st.subheader("Captured Image")
                        st.image(annotated, use_container_width=True)

                    with col2:
                        st.subheader("Prediction Results")
                        for idx, det in enumerate(detections, start=1):
                            emotion = det['emotion']
                            emoji = det['emoji']
                            confidence = det['confidence']

                            st.markdown(f"""
                            <div class="prediction-box">
                                <div class="emotion-display">
                                    {emotion} {'(Face ' + str(idx) + ')' if multi_face and len(detections) > 1 else ''}
                                </div>
                                <div class="confidence-text">
                                    Confidence: <strong>{confidence*100:.1f}%</strong>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.progress(float(confidence))

                            st.markdown("<h3 class='section-header'>3-Class Probability Distribution</h3>", unsafe_allow_html=True)
                            for emotion_label, prob in det['probabilities'].items():
                                emoji_icon = EMOJI_MAP.get(emotion_label, '')
                                st.markdown(f"<div class='prob-label'>{emoji_icon} <strong>{emotion_label}</strong>: {float(prob)*100:.1f}%</div>", unsafe_allow_html=True)
                                st.progress(float(prob))

                            if debug_mode and det['raw']:
                                st.markdown("<h3 class='section-header'>Raw DeepFace Output (Debug)</h3>", unsafe_allow_html=True)
                                for emotion_label, prob in det['raw'].items():
                                    st.markdown(f"<div class='prob-label' style='color: #888;'>{emotion_label.capitalize()}: {prob:.1f}%</div>", unsafe_allow_html=True)
                                st.markdown("---")

                        # Download detections as CSV
                        csv_bytes = detections_to_csv_bytes(detections, "webcam")
                        st.download_button(
                            label="⬇️ Download results (CSV)",
                            data=csv_bytes,
                            file_name="moodsense_results_webcam.csv",
                            mime="text/csv"
                        )

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    else:
        st.info("👆 Click 'Enable Camera' button above to start using your webcam.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><strong>⚙️ GearGen</strong> - AI MoodSense</p>
    <p>© Thuộc bản quyền của Trung tâm Công nghệ GearGen.</p>
    <p style="font-size: 0.9rem; color: #999;">Built with Streamlit, TensorFlow, and OpenCV</p>
</div>
""", unsafe_allow_html=True)
