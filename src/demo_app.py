import streamlit as st
import cv2
import tempfile
import time
import os
import onnxruntime as ort
import numpy as np
from PIL import Image

# -----------------------------------------------------------------------------
# 1. Page Configuration (Must be first)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Angiogram Segmentation AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# 2. Custom CSS for Modern Look
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main Background & Font */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    h1 {
        font-weight: 700;
        letter-spacing: -0.02em;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF904B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 700;
        color: #00CC66; /* Green for performance */
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    
    /* Input/Video Container */
    .stImage {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        overflow: hidden;
    }
    
    /* Custom divider */
    hr {
        border-color: #30363D;
        margin: 2rem 0;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 99px;
        font-size: 0.8em;
        font-weight: 600;
        background: rgba(0, 204, 102, 0.15);
        color: #00CC66;
        border: 1px solid rgba(0, 204, 102, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Model Loading (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    try:
        session = ort.InferenceSession(path)
        return session
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# -----------------------------------------------------------------------------
# 4. Processing Logic
# -----------------------------------------------------------------------------
def process_frame(frame, session, threshold, alpha, color_option):
    # Preprocess
    img = cv2.resize(frame, (512, 512)) # Model Input Size
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize & Tensor
    img_float = img.astype(np.float32) / 255.0
    img_tensor = np.transpose(img_float, (2, 0, 1)) # (C, H, W)
    img_tensor = np.expand_dims(img_tensor, axis=0) # (1, C, H, W)
    
    input_name = session.get_inputs()[0].name
    
    # Inference
    start = time.time()
    outputs = session.run(None, {input_name: img_tensor})
    end = time.time()
    inference_time = end - start
    
    # Postprocess
    pred = outputs[0][0][0] # (512, 512)
    pred = 1 / (1 + np.exp(-pred)) # Sigmoid
    mask = (pred > threshold).astype(np.uint8)
    
    # Create overlay
    overlay = img_rgb.copy()
    
    # Color logic
    color_map = {
        "Green (Standard)": [0, 255, 0],
        "Red (Alert)": [255, 0, 0],
        "Blue (Cool)": [0, 100, 255],
        "Heatmap (Yellow)": [255, 255, 0]
    }
    color = color_map.get(color_option, [0, 255, 0])
    
    overlay[mask == 1] = color
    
    # Blend
    combined = cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0)
    
    return combined, inference_time

# -----------------------------------------------------------------------------
# 5. UI Layout
# -----------------------------------------------------------------------------

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ System Config")
    
    # Model Status
    model_path = st.text_input("Model Path", "checkpoints/model.onnx")
    if os.path.exists(model_path):
        st.markdown('<div class="status-badge">● Model Ready</div>', unsafe_allow_html=True)
    else:
        st.error("Model not found!")
    
    st.markdown("---")
    st.markdown("### 🎛️ Parameters")
    
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, help="Minimum probability to consider a pixel as a vessel.")
    overlay_alpha = st.slider("Overlay Opacity", 0.0, 1.0, 0.4)
    color_option = st.selectbox("Overlay Color", ["Green (Standard)", "Red (Alert)", "Blue (Cool)", "Heatmap (Yellow)"])
    
    st.markdown("---")
    st.info("💡 **Tip**: Use 'Heatmap' for better contrast on darker angiograms.")

# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    st.title("Angiogram AI")
    st.markdown("##### Real-Time Coronary Artery Segmentation")

with col2:
    # Header Metrics Placeholder
    pass

st.markdown("---")

# File Upload Section
uploaded_file = st.file_uploader("📂 Upload Angiogram Video", type=["mp4", "avi", "mov"], label_visibility="collapsed")

if uploaded_file is not None:
    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Load Model
    session = load_model(model_path)
    
    if session:
        # Dashboard Layout
        col_video_src, col_video_out, col_stats = st.columns([1.5, 1.5, 1])
        
        with col_video_src:
            st.markdown("**Original Feed**")
            src_placeholder = st.empty()
            
        with col_video_out:
            st.markdown("**AI Segmentation**")
            out_placeholder = st.empty()
            
        with col_stats:
            st.markdown("**Performance**")
            fps_metric = st.empty()
            latency_metric = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            start_btn = st.button("▶ Start Processing", type="primary", use_container_width=True)
            stop_btn = st.button("⏹ Stop", type="secondary", use_container_width=True)
        
        if start_btn:
            frame_count = 0
            
            fps_metric.metric("FPS", "0.0")
            latency_metric.metric("Latency", "0 ms")
            status_text.caption("Initializing Engine...")
            
            while cap.isOpened():
                if stop_btn: # Streamlit button logic is tricky inside loops, usually requires session state, simplified here
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process
                processed_frame, inf_time = process_frame(frame, session, confidence_threshold, overlay_alpha, color_option)
                
                # Update UI
                src_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                out_placeholder.image(processed_frame, channels="RGB", use_container_width=True)
                
                # Metrics
                fps = 1.0 / (inf_time + 1e-6)
                latency_ms = inf_time * 1000
                
                fps_metric.metric("FPS", f"{fps:.1f}")
                latency_metric.metric("Latency", f"{latency_ms:.1f} ms")
                
                # Progress
                prog = min(frame_count / total_frames, 1.0)
                progress_bar.progress(prog)
                status_text.caption(f"Processing Frame {frame_count}/{total_frames}")
                
            cap.release()
            st.success("Analysis Complete!")
            
    else:
        st.error(f"Failed to load model from `{model_path}`.")

else:
    # Placeholder / Hero Section when no file is uploaded
    st.markdown("""
    <div style="text-align: center; padding: 50px; background: rgba(255,255,255,0.05); border-radius: 12px;">
        <h3>👋 Welcome to the Demo</h3>
        <p style="color: #888;">Drag and drop an angiogram video file to begin real-time analysis.</p>
    </div>
    """, unsafe_allow_html=True)
