"""
PawCare AI — Dog Skin Disease Classifier
A premium Streamlit web application for canine skin health analysis
powered by MobileNetV2 deep learning.
"""

import streamlit as st
import numpy as np
from PIL import Image
import os

# ─── Page configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="PawCare AI — Dog Skin Disease Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Class labels (fixed order matching training) ────────────────────────────
CLASS_LABELS = [
    "Bacterial Dermatosis",
    "Fungal Infections",
    "Healthy",
    "Hypersensitivity / Allergic Dermatosis",
]

CLASS_ICONS = ["🦠", "🍄", "✅", "⚠️"]

CLASS_DESCRIPTIONS = [
    "Skin conditions caused by bacterial infection, often presenting as pustules, crusts, or redness.",
    "Infections caused by fungi such as ringworm, typically showing circular patches or hair loss.",
    "No significant skin disease detected. The skin appears within normal parameters.",
    "Allergic or hypersensitivity reactions, which may cause itching, redness, or swelling.",
]

CLASS_COLORS = ["#e74c3c", "#e67e22", "#27ae60", "#f39c12"]

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root variables ── */
:root {
    --forest: #2d6a4f;
    --forest-dark: #1b4332;
    --forest-light: #40916c;
    --sage: #95d5b2;
    --sage-light: #b7e4c7;
    --sage-pale: #d8f3dc;
    --ivory: #fefae0;
    --ivory-light: #fffef5;
    --cream: #f4f1de;
    --text-dark: #1a1a2e;
    --text-muted: #6c757d;
    --white: #ffffff;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.08);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.1);
    --shadow-lg: 0 8px 30px rgba(0,0,0,0.12);
    --radius: 16px;
    --radius-sm: 10px;
}

/* ── Global resets ── */
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: linear-gradient(135deg, var(--ivory-light) 0%, var(--sage-pale) 50%, var(--cream) 100%) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--forest-dark) 0%, var(--forest) 100%) !important;
    border-right: none !important;
}

section[data-testid="stSidebar"] * {
    color: var(--sage-light) !important;
}

section[data-testid="stSidebar"] .stRadio label {
    color: var(--white) !important;
    font-weight: 500 !important;
    padding: 8px 12px !important;
    border-radius: var(--radius-sm) !important;
    transition: background 0.2s ease !important;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.08) !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(149, 213, 178, 0.2) !important;
}

/* ── Cards ── */
.premium-card {
    background: var(--white);
    border-radius: var(--radius);
    padding: 2rem;
    box-shadow: var(--shadow-md);
    border: 1px solid rgba(45, 106, 79, 0.06);
    margin-bottom: 1.5rem;
}

/* ── Hero section ── */
.hero-section {
    background: linear-gradient(135deg, var(--forest-dark) 0%, var(--forest) 40%, var(--forest-light) 100%);
    border-radius: var(--radius);
    padding: 3rem 2.5rem;
    margin-bottom: 2rem;
    box-shadow: var(--shadow-lg);
    position: relative;
    overflow: hidden;
}

.hero-section::before {
    content: "";
    position: absolute;
    top: -40%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(149,213,178,0.15) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: var(--white);
    margin-bottom: 0.5rem;
    line-height: 1.2;
    letter-spacing: -0.5px;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: var(--sage-light);
    font-weight: 400;
    line-height: 1.6;
    max-width: 600px;
}

/* ── Upload area ── */
.upload-zone {
    background: var(--white);
    border: 2px dashed var(--sage);
    border-radius: var(--radius);
    padding: 2.5rem 2rem;
    text-align: center;
    transition: all 0.3s ease;
    margin-bottom: 1.5rem;
}

.upload-zone:hover {
    border-color: var(--forest-light);
    background: var(--sage-pale);
}

/* ── Result card ── */
.result-card {
    background: var(--white);
    border-radius: var(--radius);
    padding: 2rem;
    box-shadow: var(--shadow-lg);
    border-left: 5px solid var(--forest);
    margin-bottom: 1.5rem;
}

.predicted-label {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--forest-dark);
    margin-bottom: 0.25rem;
}

.confidence-value {
    font-size: 2.5rem;
    font-weight: 800;
    color: var(--forest);
    letter-spacing: -1px;
}

/* ── Score bars ── */
.score-bar-container {
    margin-bottom: 1rem;
}

.score-bar-label {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-dark);
    margin-bottom: 4px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.score-bar-bg {
    background: var(--cream);
    border-radius: 8px;
    height: 12px;
    overflow: hidden;
    position: relative;
}

.score-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Disclaimer ── */
.disclaimer-box {
    background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
    border: 1px solid #ffc107;
    border-radius: var(--radius-sm);
    padding: 1rem 1.25rem;
    font-size: 0.85rem;
    color: #856404;
    margin-top: 1.5rem;
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--forest) 0%, var(--forest-light) 100%) !important;
    color: var(--white) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 0.3s ease !important;
    letter-spacing: 0.2px !important;
}

.stButton > button:hover {
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-1px) !important;
}

/* ── About page ── */
.about-card {
    background: var(--white);
    border-radius: var(--radius);
    padding: 2rem;
    box-shadow: var(--shadow-md);
    margin-bottom: 1.5rem;
    border-top: 4px solid var(--forest);
}

/* ── File uploader styling ── */
[data-testid="stFileUploader"] {
    border: none !important;
}

[data-testid="stFileUploader"] section {
    border: 2px dashed var(--sage) !important;
    border-radius: var(--radius) !important;
    padding: 1.5rem !important;
    background: rgba(216, 243, 220, 0.2) !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: var(--forest) !important;
}

/* ── Metric styling ── */
[data-testid="stMetric"] {
    background: var(--sage-pale);
    border-radius: var(--radius-sm);
    padding: 1rem;
}

/* ── Divider ── */
.styled-divider {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--sage), transparent);
    margin: 2rem 0;
}

/* ── Section heading ── */
.section-heading {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--forest-dark);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Image preview ── */
.image-preview-container {
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: var(--shadow-md);
    margin-bottom: 1rem;
}

/* Hide default Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Model loading ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the pre-trained MobileNetV2 model for dog skin disease classification."""
    import tensorflow as tf

    model_path = os.path.join(os.path.dirname(__file__), "best_mobilenetv2.keras")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()

    model = tf.keras.models.load_model(model_path)

    # Validate expected architecture
    expected_output = 4
    actual_output = model.output_shape[-1]
    if actual_output != expected_output:
        st.error(
            f"Model output shape mismatch: expected {expected_output} classes, "
            f"got {actual_output}. Please verify the model file."
        )
        st.stop()

    return model


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess a PIL Image for MobileNetV2 inference."""
    import tensorflow as tf

    img = image.convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array


def run_prediction(model, img_array: np.ndarray) -> np.ndarray:
    """Run model prediction and return scores."""
    predictions = model.predict(img_array, verbose=0)

    if predictions.shape[-1] != 4:
        raise ValueError(
            f"Unexpected output shape {predictions.shape}. Expected (1, 4)."
        )

    return predictions[0]


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1.5rem 0 0.5rem 0;">
        <div style="font-size: 3rem; margin-bottom: 0.25rem;">🐾</div>
        <div style="font-size: 1.6rem; font-weight: 800; color: #ffffff !important;
                    letter-spacing: -0.5px;">PawCare AI</div>
        <div style="font-size: 0.8rem; font-weight: 400; color: #b7e4c7 !important;
                    margin-top: 2px;">Canine Skin Health Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔬 Skin Analysis", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown("""
    <div style="padding: 0.75rem; background: rgba(255,255,255,0.06);
                border-radius: 10px; font-size: 0.78rem; line-height: 1.5;">
        <div style="font-weight: 600; color: #ffffff !important; margin-bottom: 4px;">
            Model Information
        </div>
        <div>Architecture: MobileNetV2</div>
        <div>Input: 224 × 224 px</div>
        <div>Classes: 4</div>
        <div>Format: Keras (.keras)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size: 0.72rem; text-align: center; color: #95d5b2 !important;
                padding-top: 0.5rem;">
        Built with ❤️ for pet parents<br>
        © 2026 PawCare AI
    </div>
    """, unsafe_allow_html=True)


# ─── Dashboard Page ──────────────────────────────────────────────────────────
if page == "🏠 Dashboard":
    # Hero
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🐾 Understand Your Dog's Skin Health</div>
        <div class="hero-subtitle">
            Upload a photo of your dog's skin and let our AI-powered classifier
            help you identify potential conditions. Fast, non-invasive, and informative
            — designed to support conversations with your veterinarian.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    cols = st.columns(3, gap="large")
    features = [
        ("🧠", "AI-Powered Analysis",
         "Deep learning model trained on veterinary dermatology images for pattern recognition."),
        ("⚡", "Instant Results",
         "Get analysis scores in seconds. Upload a clear photo and receive class-by-class breakdowns."),
        ("🛡️", "Four Key Categories",
         "Screens for bacterial, fungal, allergic conditions, and healthy skin indicators."),
    ]
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div class="premium-card" style="text-align:center; min-height: 220px;">
                <div style="font-size: 2.5rem; margin-bottom: 0.75rem;">{icon}</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #1b4332;
                            margin-bottom: 0.5rem;">{title}</div>
                <div style="font-size: 0.9rem; color: #6c757d; line-height: 1.6;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

    # How it works
    st.markdown('<div class="section-heading">📋 How It Works</div>', unsafe_allow_html=True)
    steps = st.columns(4, gap="medium")
    step_data = [
        ("1️⃣", "Upload", "Take a clear photo of the affected skin area and upload it."),
        ("2️⃣", "Process", "The image is resized and preprocessed for the AI model."),
        ("3️⃣", "Analyze", "MobileNetV2 evaluates the image across four categories."),
        ("4️⃣", "Review", "Review scores and discuss findings with your veterinarian."),
    ]
    for col, (num, title, desc) in zip(steps, step_data):
        with col:
            st.markdown(f"""
            <div class="premium-card" style="text-align:center; padding: 1.5rem 1rem;">
                <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">{num}</div>
                <div style="font-size: 1rem; font-weight: 700; color: #2d6a4f;
                            margin-bottom: 0.4rem;">{title}</div>
                <div style="font-size: 0.82rem; color: #6c757d; line-height: 1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer-box">
        <span style="font-size: 1.3rem;">⚠️</span>
        <div>
            <strong>Important Disclaimer:</strong> Experimental AI result only.
            This tool cannot diagnose skin conditions.
            Consult a qualified veterinarian for any health concerns regarding your pet.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Skin Analysis Page ─────────────────────────────────────────────────────
elif page == "🔬 Skin Analysis":
    st.markdown("""
    <div class="section-heading" style="font-size: 1.5rem; margin-bottom: 0.25rem;">
        🔬 Skin Analysis
    </div>
    <p style="color: #6c757d; margin-bottom: 1.5rem; font-size: 0.95rem;">
        Upload a clear photo of your dog's skin for AI-powered analysis.
    </p>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False
    if "predictions" not in st.session_state:
        st.session_state.predictions = None
    if "analyzed_image" not in st.session_state:
        st.session_state.analyzed_image = None

    # Layout: upload left, results right
    col_upload, col_results = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("""
        <div class="premium-card">
            <div class="section-heading" style="font-size: 1.1rem;">📸 Upload Image</div>
            <p style="color: #6c757d; font-size: 0.85rem; margin-bottom: 1rem;">
                Drag and drop or browse for a JPG, JPEG, or PNG image.
                Use a well-lit, focused photo of the affected area for best results.
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload a dog skin image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            help="Supported formats: JPG, JPEG, PNG. Max 200 MB.",
        )

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.markdown("""
                <div class="section-heading" style="font-size: 1rem; margin-top: 0.5rem;">
                    🖼️ Image Preview
                </div>
                """, unsafe_allow_html=True)

                st.image(image, use_container_width=True)

                st.markdown(f"""
                <div style="background: #f8f9fa; border-radius: 8px; padding: 0.6rem 1rem;
                            font-size: 0.8rem; color: #6c757d; margin: 0.5rem 0 1rem 0;">
                    📄 <strong>{uploaded_file.name}</strong> &nbsp;·&nbsp;
                    {image.size[0]}×{image.size[1]} px &nbsp;·&nbsp;
                    {uploaded_file.size / 1024:.1f} KB &nbsp;·&nbsp;
                    {image.mode} mode
                </div>
                """, unsafe_allow_html=True)

                # Analyze button
                btn_cols = st.columns([1, 1])
                with btn_cols[0]:
                    analyze_clicked = st.button(
                        "🔬 Analyze Image",
                        use_container_width=True,
                        type="primary",
                    )
                with btn_cols[1]:
                    reset_clicked = st.button(
                        "🔄 Reset",
                        use_container_width=True,
                    )

                if reset_clicked:
                    st.session_state.analysis_done = False
                    st.session_state.predictions = None
                    st.session_state.analyzed_image = None
                    st.rerun()

                if analyze_clicked:
                    with st.spinner("🧠 Loading model and analyzing image..."):
                        try:
                            model = load_model()
                            img_array = preprocess_image(image)
                            predictions = run_prediction(model, img_array)
                            st.session_state.predictions = predictions
                            st.session_state.analysis_done = True
                            st.session_state.analyzed_image = image.copy()
                        except Exception as e:
                            st.error(f"❌ Analysis failed: {str(e)}")
                            st.session_state.analysis_done = False

            except Exception as e:
                st.error(f"❌ Could not open image file: {str(e)}")
        else:
            # Empty state
            st.markdown("""
            <div style="background: #ffffff; border: 2px dashed #95d5b2;
                        border-radius: 16px; padding: 3rem 2rem; text-align: center;
                        margin-top: 0.5rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📷</div>
                <div style="font-size: 1.1rem; font-weight: 600; color: #2d6a4f;
                            margin-bottom: 0.5rem;">No Image Uploaded Yet</div>
                <div style="font-size: 0.88rem; color: #6c757d; line-height: 1.6;">
                    Upload a clear, well-lit photo of your dog's skin area.<br>
                    Supported formats: JPG, JPEG, PNG
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Results column
    with col_results:
        if st.session_state.analysis_done and st.session_state.predictions is not None:
            predictions = st.session_state.predictions
            top_idx = int(np.argmax(predictions))
            top_score = float(predictions[top_idx])
            top_label = CLASS_LABELS[top_idx]
            top_icon = CLASS_ICONS[top_idx]
            top_color = CLASS_COLORS[top_idx]

            # Predicted class card
            st.markdown(f"""
            <div class="result-card" style="border-left-color: {top_color};">
                <div style="font-size: 0.8rem; font-weight: 600; color: #6c757d;
                            text-transform: uppercase; letter-spacing: 1px;
                            margin-bottom: 0.5rem;">
                    Predicted Class
                </div>
                <div class="predicted-label">
                    {top_icon} {top_label}
                </div>
                <div class="confidence-value" style="color: {top_color};">
                    {top_score * 100:.1f}%
                </div>
                <div style="font-size: 0.82rem; color: #6c757d; margin-top: 0.5rem;
                            line-height: 1.5;">
                    {CLASS_DESCRIPTIONS[top_idx]}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # All scores
            st.markdown("""
            <div class="premium-card">
                <div class="section-heading" style="font-size: 1.05rem;">
                    📊 All Class Scores
                </div>
                <p style="font-size: 0.78rem; color: #6c757d; margin-bottom: 1.25rem;">
                    Model output scores across all four categories.
                    Higher values indicate stronger pattern matches.
                </p>
            """, unsafe_allow_html=True)

            # Sort by score descending for display
            sorted_indices = np.argsort(predictions)[::-1]

            bars_html = ""
            for idx in sorted_indices:
                score = float(predictions[idx])
                label = CLASS_LABELS[idx]
                icon = CLASS_ICONS[idx]
                color = CLASS_COLORS[idx]
                pct = score * 100
                is_top = idx == top_idx

                bar_bg = color if is_top else "#b0bec5"
                weight = "700" if is_top else "500"
                tag = ' <span style="background:#2d6a4f;color:#fff;font-size:0.65rem;padding:2px 8px;border-radius:4px;margin-left:6px;">TOP</span>' if is_top else ""

                bars_html += f"""
                <div class="score-bar-container">
                    <div class="score-bar-label">
                        <span style="font-weight:{weight};">{icon} {label}{tag}</span>
                        <span style="font-weight:700; color:{color};">{pct:.1f}%</span>
                    </div>
                    <div class="score-bar-bg">
                        <div class="score-bar-fill"
                             style="width:{max(pct, 0.5):.1f}%; background:{bar_bg};"></div>
                    </div>
                </div>
                """

            bars_html += "</div>"
            st.markdown(bars_html, unsafe_allow_html=True)

            # Disclaimer
            st.markdown("""
            <div class="disclaimer-box">
                <span style="font-size: 1.3rem;">⚠️</span>
                <div>
                    <strong>Important Disclaimer:</strong> Experimental AI result only.
                    This tool cannot diagnose skin conditions.
                    Consult a qualified veterinarian for any health concerns
                    regarding your pet.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            # Empty results state
            st.markdown("""
            <div class="premium-card" style="text-align: center; padding: 3rem 2rem;">
                <div style="font-size: 3.5rem; margin-bottom: 1rem;">🔍</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #2d6a4f;
                            margin-bottom: 0.75rem;">
                    Awaiting Analysis
                </div>
                <div style="font-size: 0.9rem; color: #6c757d; line-height: 1.7;
                            max-width: 350px; margin: 0 auto;">
                    Upload an image on the left panel and click
                    <strong>Analyze Image</strong> to run the AI classifier.
                    Results will appear here with detailed score breakdowns.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Info cards
            st.markdown("""
            <div class="premium-card" style="padding: 1.25rem;">
                <div style="font-size: 0.9rem; font-weight: 600; color: #2d6a4f;
                            margin-bottom: 0.75rem;">💡 Tips for Best Results</div>
                <ul style="font-size: 0.82rem; color: #6c757d; line-height: 1.8;
                           margin: 0; padding-left: 1.25rem;">
                    <li>Use natural, even lighting</li>
                    <li>Focus on the affected skin area</li>
                    <li>Avoid blurry or distant shots</li>
                    <li>Remove accessories from the area if possible</li>
                    <li>Capture the image against a clean background</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)


# ─── About Page ──────────────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.markdown("""
    <div class="hero-section" style="padding: 2.5rem;">
        <div class="hero-title" style="font-size: 2rem;">ℹ️ About PawCare AI</div>
        <div class="hero-subtitle">
            Learn about the technology, methodology, and limitations
            of the PawCare AI skin disease classifier.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="about-card">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🧠</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #1b4332;
                        margin-bottom: 0.75rem;">The Technology</div>
            <p style="font-size: 0.88rem; color: #6c757d; line-height: 1.7;">
                PawCare AI uses <strong>MobileNetV2</strong>, a lightweight convolutional
                neural network architecture developed by Google, fine-tuned on canine
                dermatology images. The model processes 224×224 pixel RGB images and
                produces scores across four skin condition categories.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🏷️</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #1b4332;
                        margin-bottom: 0.75rem;">Classification Categories</div>
            <ul style="font-size: 0.88rem; color: #6c757d; line-height: 2;">
                <li>🦠 <strong>Bacterial Dermatosis</strong> — Bacterial skin infections</li>
                <li>🍄 <strong>Fungal Infections</strong> — Fungal conditions like ringworm</li>
                <li>✅ <strong>Healthy</strong> — No significant skin disease detected</li>
                <li>⚠️ <strong>Hypersensitivity / Allergic Dermatosis</strong> — Allergic reactions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="about-card">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">⚙️</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #1b4332;
                        margin-bottom: 0.75rem;">How It Works</div>
            <ol style="font-size: 0.88rem; color: #6c757d; line-height: 2;">
                <li>Image is converted to RGB and resized to 224×224 pixels</li>
                <li>Pixel values are preprocessed using MobileNetV2 normalization</li>
                <li>The neural network processes the image through its layers</li>
                <li>Softmax output produces score distribution across 4 classes</li>
                <li>The class with the highest score is presented as the prediction</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card" style="border-top-color: #e74c3c;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">⚠️</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #c0392b;
                        margin-bottom: 0.75rem;">Limitations & Disclaimer</div>
            <p style="font-size: 0.88rem; color: #6c757d; line-height: 1.7;">
                <strong>Experimental AI result only. This tool cannot diagnose
                skin conditions. Consult a qualified veterinarian.</strong>
            </p>
            <ul style="font-size: 0.85rem; color: #6c757d; line-height: 1.9;">
                <li>This is an experimental research tool, not a medical device</li>
                <li>Results are AI-generated pattern scores, not clinical diagnoses</li>
                <li>The model may produce incorrect or misleading results</li>
                <li>Always consult a licensed veterinarian for your pet's health</li>
                <li>Do not delay professional treatment based on this tool's output</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
        <span style="font-size: 1.3rem;">⚠️</span>
        <div>
            <strong>Important Disclaimer:</strong> Experimental AI result only.
            This tool cannot diagnose skin conditions.
            Consult a qualified veterinarian for any health concerns regarding your pet.
        </div>
    </div>
    """, unsafe_allow_html=True)
