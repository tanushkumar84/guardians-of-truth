import streamlit as st
import torch
import os
import logging
import warnings
import subprocess
import json
from pathlib import Path
from utils.utils_video_input import process_video_input
from utils.utils_session import *
from utils.utils_image_input import process_image_input

# Configure memory management
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def ensure_kaggle_credentials():
    """Ensure Kaggle credentials exist for Kaggle CLI.
    Tries env vars first; if missing, writes ~/.kaggle/kaggle.json from st.secrets when available.
    """
    has_env = bool(os.getenv("KAGGLE_USERNAME")) and bool(os.getenv("KAGGLE_KEY"))
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if has_env:
        return True

    # Try to create kaggle.json from Streamlit secrets
    try:
        username = st.secrets.get("KAGGLE_USERNAME") if hasattr(st, "secrets") else None
        key = st.secrets.get("KAGGLE_KEY") if hasattr(st, "secrets") else None
    except Exception:
        username, key = None, None

    if username and key:
        # Export env vars for Kaggle CLI and write kaggle.json for redundancy
        os.environ["KAGGLE_USERNAME"] = str(username)
        os.environ["KAGGLE_KEY"] = str(key)
        try:
            kaggle_dir.mkdir(parents=True, exist_ok=True)
            with kaggle_json.open('w') as f:
                json.dump({"username": str(username), "key": str(key)}, f)
        except Exception as e:
            st.warning(f"Could not write Kaggle credentials file: {e}")

    # Return whether we have credentials available one way or another
    return bool(os.getenv("KAGGLE_USERNAME")) and bool(os.getenv("KAGGLE_KEY")) or kaggle_json.exists()

# Run setup script if models don't exist
# You can download the model from kaggle 
# For streamlit web deployement, remove if using in localhost or uncomment the line below
#os.makedirs("runs", exist_ok=True)
if not os.path.exists("runs"):
    st.info("Initializing models (first run). Preparing Kaggle credentials...")
    creds_ok = ensure_kaggle_credentials()
    # Sanity check booleans (no secrets printed)
    try:
        st.write(f"Secrets present: {bool(hasattr(st, 'secrets') and st.secrets.get('KAGGLE_USERNAME') and st.secrets.get('KAGGLE_KEY'))}")
        creds_path = Path.home() / ".kaggle" / "kaggle.json"
        st.write(f"Kaggle credentials file present: {creds_path.exists()}")
        st.write(f"Env vars present: {bool(os.getenv('KAGGLE_USERNAME')) and bool(os.getenv('KAGGLE_KEY'))}")
    except Exception:
        pass

    if creds_ok:
        try:
            project_dir = Path(__file__).parent
            st.info("Downloading model weights from Kaggle (this may take a few minutes)...")
            subprocess.run(['bash', 'setup.sh'], check=True, cwd=str(project_dir))
            st.success("Models downloaded and extracted.")
        except subprocess.CalledProcessError as e:
            st.error(f"Error running setup script: {str(e)}")
            # Do not stop the app; let UI load so users see guidance
    else:
        st.warning("Kaggle credentials not detected. Set KAGGLE_USERNAME and KAGGLE_KEY in Streamlit Secrets to enable automatic model download, or upload models to runs/models/... manually.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.serialization')

# Set page config
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS with dark background and animations matching the reference image
st.markdown("""
    <style>
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Dark background with starfield - exact match to reference */
        .stApp {
            background: #000000 !important;
            position: relative;
            overflow-x: hidden;
        }
        
        /* Navbar styling */
        .navbar {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 70px;
            background: rgba(10, 10, 15, 0.95);
            backdrop-filter: blur(10px);
            border-bottom: 2px solid rgba(23, 198, 216, 0.3);
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 3rem;
            z-index: 9999;
            box-shadow: 0 4px 20px rgba(23, 198, 216, 0.1);
        }
        
        .navbar-brand {
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, #17c6d8, #0ea5e9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: 1px;
            text-shadow: 0 0 20px rgba(23, 198, 216, 0.3);
        }
        
        .navbar-menu {
            display: flex;
            gap: 2rem;
            align-items: center;
        }
        
        .nav-link {
            color: #ffffff !important;
            text-decoration: none;
            font-size: 1.1rem;
            font-weight: 500;
            padding: 0.5rem 1.5rem;
            border-radius: 8px;
            transition: all 0.3s ease;
            cursor: pointer;
            border: 1px solid transparent;
        }
        
        .nav-link:hover {
            background: rgba(23, 198, 216, 0.2);
            border-color: rgba(23, 198, 216, 0.5);
            transform: translateY(-2px);
        }
        
        .nav-link.active {
            background: rgba(23, 198, 216, 0.3);
            border-color: #17c6d8;
        }
        
        /* Add padding to main content to account for fixed navbar */
        .main .block-container {
            padding-top: 100px !important;
        }
        
        /* About modal styling */
        .about-modal {
            background: rgba(15, 15, 20, 0.98);
            border: 2px solid rgba(23, 198, 216, 0.5);
            border-radius: 20px;
            padding: 2.5rem;
            margin: 2rem auto;
            max-width: 900px;
            box-shadow: 0 0 40px rgba(23, 198, 216, 0.3);
        }
        
        .about-section {
            margin-bottom: 2rem;
        }
        
        .about-section h2 {
            color: #17c6d8 !important;
            font-size: 1.8rem;
            margin-bottom: 1rem;
            border-bottom: 2px solid rgba(23, 198, 216, 0.3);
            padding-bottom: 0.5rem;
        }
        
        .about-section h3 {
            color: #17c6d8 !important;
            font-size: 1.4rem;
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
        }
        
        .about-section p, .about-section li {
            color: #ffffff !important;
            font-size: 1.05rem;
            line-height: 1.8;
            margin-bottom: 0.8rem;
        }
        
        .about-section ul {
            list-style-position: inside;
            margin-left: 1rem;
        }
        
        .model-card {
            background: rgba(23, 198, 216, 0.1);
            border: 1px solid rgba(23, 198, 216, 0.3);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .model-card h4 {
            color: #17c6d8 !important;
            font-size: 1.2rem;
            margin-bottom: 0.5rem;
        }
        
        .stat-badge {
            display: inline-block;
            background: rgba(23, 198, 216, 0.2);
            color: #17c6d8 !important;
            padding: 0.3rem 0.8rem;
            border-radius: 6px;
            margin-right: 0.5rem;
            font-weight: 600;
            font-size: 0.95rem;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            background: rgba(15, 15, 20, 0.9);
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid rgba(23, 198, 216, 0.3);
        }
        
        .stTabs [data-baseweb="tab"] {
            background: rgba(23, 198, 216, 0.1);
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            color: #ffffff !important;
            border: 1px solid transparent;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(23, 198, 216, 0.2);
            border-color: rgba(23, 198, 216, 0.5);
        }
        
        .stTabs [aria-selected="true"] {
            background: rgba(23, 198, 216, 0.3) !important;
            border-color: #17c6d8 !important;
        }
        
        .stTabs [data-baseweb="tab-panel"] {
            background: transparent;
            padding-top: 1.5rem;
        }
        
        /* CRITICAL: Force all content to be visible */
        .main, .main .block-container, [data-testid="stAppViewContainer"] {
            position: relative !important;
            z-index: 1000 !important;
            background: transparent !important;
        }
        
        /* Make ALL text white by default */
        body, p, div, span, label, h1, h2, h3, h4, h5, h6 {
            color: #ffffff !important;
        }
        
        /* Force radio button visibility */
        [role="radiogroup"] label {
            background: rgba(23, 198, 216, 0.25) !important;
            padding: 12px 30px !important;
            border: 2px solid #17c6d8 !important;
            border-radius: 10px !important;
            margin: 0 10px !important;
            color: #ffffff !important;
            font-size: 1.2rem !important;
            font-weight: 600 !important;
        }
        
        [role="radiogroup"] label:hover {
            background: rgba(23, 198, 216, 0.4) !important;
        }
        
        /* Force file uploader visibility */
        [data-testid="stFileUploader"] section {
            background: rgba(23, 198, 216, 0.15) !important;
            border: 3px dashed #17c6d8 !important;
            padding: 30px !important;
            border-radius: 15px !important;
            min-height: 120px !important;
        }
        
        [data-testid="stFileUploader"] * {
            color: #ffffff !important;
        }
        
        [data-testid="stFileUploader"] button {
            background: #17c6d8 !important;
            color: #000000 !important;
            font-weight: 700 !important;
            padding: 10px 25px !important;
            border-radius: 8px !important;
        }
        
        /* Animated starfield background */
        .stApp::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                radial-gradient(1px 1px at 10% 20%, white, transparent),
                radial-gradient(1px 1px at 20% 80%, white, transparent),
                radial-gradient(1px 1px at 30% 50%, white, transparent),
                radial-gradient(1px 1px at 40% 10%, white, transparent),
                radial-gradient(1px 1px at 50% 70%, white, transparent),
                radial-gradient(1px 1px at 60% 30%, white, transparent),
                radial-gradient(1px 1px at 70% 90%, white, transparent),
                radial-gradient(1px 1px at 80% 40%, white, transparent),
                radial-gradient(1px 1px at 90% 60%, white, transparent),
                radial-gradient(1px 1px at 15% 45%, white, transparent),
                radial-gradient(1px 1px at 85% 15%, white, transparent),
                radial-gradient(1px 1px at 25% 65%, white, transparent);
            background-size: 100% 100%;
            opacity: 0.5;
            z-index: 0;
            pointer-events: none;
        }
        
        /* Floating hexagonal shapes - matching reference image */
        .floating-hex {
            position: fixed;
            opacity: 0.12;
            z-index: 0;
            pointer-events: none;
        }
        
        .hex-top-left {
            top: 8%;
            left: 7%;
            width: 90px;
            height: 90px;
            background: linear-gradient(135deg, #1a8a9a, #0ea5e9);
            clip-path: polygon(30% 0%, 70% 0%, 100% 30%, 100% 70%, 70% 100%, 30% 100%, 0% 70%, 0% 30%);
            animation: floatHex1 20s ease-in-out infinite;
        }
        
        .hex-top-right {
            top: 3%;
            right: 5%;
            width: 110px;
            height: 110px;
            background: linear-gradient(135deg, #17c6d8, #1a8a9a);
            clip-path: polygon(30% 0%, 70% 0%, 100% 30%, 100% 70%, 70% 100%, 30% 100%, 0% 70%, 0% 30%);
            animation: floatHex2 25s ease-in-out infinite;
        }
        
        .hex-center-left {
            top: 45%;
            left: 25%;
            width: 85px;
            height: 85px;
            background: linear-gradient(135deg, #0f5f6a, #17c6d8);
            clip-path: polygon(30% 0%, 70% 0%, 100% 30%, 100% 70%, 70% 100%, 30% 100%, 0% 70%, 0% 30%);
            animation: floatHex3 22s ease-in-out infinite;
        }
        
        .hex-bottom-right {
            bottom: 25%;
            right: 10%;
            width: 95px;
            height: 95px;
            background: linear-gradient(135deg, #17c6d8, #0ea5e9);
            clip-path: polygon(30% 0%, 70% 0%, 100% 30%, 100% 70%, 70% 100%, 30% 100%, 0% 70%, 0% 30%);
            animation: floatHex1 18s ease-in-out infinite;
        }
        
        @keyframes floatHex1 {
            0%, 100% { transform: translate(0, 0) rotate(0deg); }
            25% { transform: translate(15px, -15px) rotate(90deg); }
            50% { transform: translate(0, -25px) rotate(180deg); }
            75% { transform: translate(-15px, -15px) rotate(270deg); }
        }
        
        @keyframes floatHex2 {
            0%, 100% { transform: translate(0, 0) rotate(0deg); }
            25% { transform: translate(-20px, 20px) rotate(90deg); }
            50% { transform: translate(-30px, 0) rotate(180deg); }
            75% { transform: translate(-20px, -20px) rotate(270deg); }
        }
        
        @keyframes floatHex3 {
            0%, 100% { transform: translate(0, 0) rotate(0deg); }
            33% { transform: translate(20px, 15px) rotate(120deg); }
            66% { transform: translate(-15px, 20px) rotate(240deg); }
        }
        
        /* Small accent dots - cyan/blue dots visible in reference */
        .accent-dot {
            position: fixed;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            z-index: 0;
            pointer-events: none;
        }
        
        .dot-1 {
            top: 60%;
            left: 35%;
            background: #00ff9d;
            opacity: 0.6;
            animation: pulse1 3s ease-in-out infinite;
        }
        
        .dot-2 {
            top: 75%;
            right: 45%;
            background: #17c6d8;
            opacity: 0.5;
            animation: pulse2 4s ease-in-out infinite;
        }
        
        @keyframes pulse1 {
            0%, 100% { transform: scale(1); opacity: 0.6; }
            50% { transform: scale(1.3); opacity: 0.8; }
        }
        
        @keyframes pulse2 {
            0%, 100% { transform: scale(1); opacity: 0.5; }
            50% { transform: scale(1.4); opacity: 0.7; }
        }
        
        /* Header styling */
        .header-container {
            text-align: center;
            padding: 3rem 0;
            margin-bottom: 2rem;
            position: relative;
            z-index: 100;
            background: transparent;
        }
        
        .main-title {
            color: #ffffff !important;
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 1rem;
            text-shadow: 0 0 20px rgba(255, 255, 255, 0.5), 0 0 40px rgba(23, 198, 216, 0.3);
            z-index: 100;
            position: relative;
        }
        
        .subtitle {
            color: #ffffff !important;
            font-size: 1.3rem;
            font-weight: 400;
            margin-bottom: 2rem;
            text-shadow: 0 0 10px rgba(255, 255, 255, 0.2);
            z-index: 100;
            position: relative;
        }
        
        /* Input section styling */
        .input-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
            position: relative;
            z-index: 100;
            background: transparent;
        }
        
        /* Radio button styling */
        div[data-testid="stHorizontalBlock"] {
            background: rgba(255, 255, 255, 0.12) !important;
            border-radius: 15px;
            padding: 1.5rem;
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin-bottom: 2rem;
            border: 2px solid rgba(23, 198, 216, 0.6) !important;
            box-shadow: 0 0 20px rgba(23, 198, 216, 0.2);
            position: relative;
            z-index: 100;
        }
        
        div.row-widget.stRadio > div {
            flex-direction: row;
            justify-content: center;
            gap: 2rem;
        }
        
        div.row-widget.stRadio > div[role="radiogroup"] > label {
            background: rgba(255, 255, 255, 0.15) !important;
            padding: 0.75rem 2.5rem;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            color: #ffffff !important;
            font-weight: 500;
            border: 2px solid rgba(255, 255, 255, 0.3) !important;
            font-size: 1.1rem !important;
        }
        
        div.row-widget.stRadio > div[role="radiogroup"] > label:hover {
            background: rgba(23, 198, 216, 0.3);
            border-color: rgba(23, 198, 216, 0.5);
            transform: translateY(-2px);
        }
        
        div.row-widget.stRadio > div[role="radiogroup"] > label[data-checked="true"] {
            background: rgba(23, 198, 216, 0.4);
            border-color: #17c6d8;
        }
        
        /* File uploader styling */
        .uploadFile {
            margin-top: 1rem;
            border: 2px dashed rgba(23, 198, 216, 0.5);
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            background: rgba(255, 255, 255, 0.08);
        }
        
        section[data-testid="stFileUploadDropzone"] {
            background: rgba(255, 255, 255, 0.12) !important;
            border: 3px dashed rgba(23, 198, 216, 0.8) !important;
            border-radius: 12px !important;
            padding: 2rem !important;
            box-shadow: 0 0 20px rgba(23, 198, 216, 0.3) !important;
            min-height: 100px !important;
        }
        
        section[data-testid="stFileUploadDropzone"] label,
        section[data-testid="stFileUploadDropzone"] span,
        section[data-testid="stFileUploadDropzone"] button,
        section[data-testid="stFileUploadDropzone"] p,
        section[data-testid="stFileUploadDropzone"] div,
        section[data-testid="stFileUploadDropzone"] small {
            color: #ffffff !important;
            font-weight: 500 !important;
            font-size: 1rem !important;
            text-shadow: 0 0 5px rgba(255, 255, 255, 0.3);
        }
        
        .upload-text {
            color: #ffffff !important;
            font-size: 1rem !important;
            margin-top: 0.5rem;
            text-align: center;
        }
        
        /* Images */
        .stImage > img {
            max-height: 300px !important;
            max-width: 100% !important;
            width: auto !important;
            object-fit: contain;
        }
        
        .face-grid-image > img {
            max-height: 200px !important;
            max-width: 100% !important;
            width: auto !important;
            object-fit: contain;
        }
        
        div.stMarkdown {
            max-width: 100%;
        }
        
        /* Result columns */
        div[data-testid="column"] {
            background: rgba(30, 30, 30, 0.9);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 12px;
            margin: 5px;
            border: 1px solid rgba(23, 198, 216, 0.3);
            position: relative;
            z-index: 100;
        }
        
        /* Text colors for dark background */
        h1, h2, h3, h4, h5, h6 {
            color: white !important;
            text-shadow: 0 0 10px rgba(255, 255, 255, 0.2);
        }
        
        p, div, span, label {
            color: #ffffff !important;
        }
        
        /* Ensure all Streamlit text is visible */
        .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span, .stMarkdown label {
            color: #ffffff !important;
        }
        
        /* Radio button text - comprehensive */
        div.row-widget.stRadio > div > label > div,
        div.row-widget.stRadio > div > label > div > p,
        div.row-widget.stRadio > div > label > div > span,
        div.row-widget.stRadio label {
            color: #ffffff !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
            text-shadow: 0 0 5px rgba(255, 255, 255, 0.3);
        }
        
        /* File uploader text - comprehensive */
        [data-testid="stFileUploader"],
        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] p,
        [data-testid="stFileUploader"] div,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploadDropzone"] small {
            color: #ffffff !important;
            background: transparent !important;
        }
        
        /* Main content wrapper */
        .main .block-container {
            padding-top: 3rem !important;
            max-width: 1200px !important;
        }
        
        /* Button text */
        .stButton > button {
            color: #ffffff !important;
            background: rgba(23, 198, 216, 0.8) !important;
            border: 1px solid rgba(23, 198, 216, 0.5) !important;
            padding: 0.5rem 2rem;
            border-radius: 8px;
            font-weight: 600 !important;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background: rgba(23, 198, 216, 1);
            transform: translateY(-2px);
        }
        
        /* Info/Alert boxes */
        .stAlert {
            background: rgba(255, 255, 255, 0.1) !important;
            color: #ffffff !important;
            border-left: 3px solid #17c6d8;
        }
        
        /* Make sure all labels are white */
        label {
            color: #ffffff !important;
            font-weight: 500 !important;
        }
    </style>
    
    <!-- Floating shapes HTML -->
    <div class="floating-hex hex-top-left"></div>
    <div class="floating-hex hex-top-right"></div>
    <div class="floating-hex hex-center-left"></div>
    <div class="floating-hex hex-bottom-right"></div>
    <div class="accent-dot dot-1"></div>
    <div class="accent-dot dot-2"></div>
""", unsafe_allow_html=True)
        
def show_home_page():
    """Display the landing/home page"""
    # Custom CSS for the header and input section
    st.markdown("""
        <style>
            /* Header styling */
            .header-container {
                text-align: center;
                padding: 2rem 0;
                margin-bottom: 2rem;
            }
            .main-title {
                color: #ffffff;
                font-size: 2.5rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
            }
            .subtitle {
                color: #a0a0a0;
                font-size: 1.1rem;
                font-weight: 300;
                margin-bottom: 2rem;
            }
            
            /* Input section styling */
            .input-container {
                max-width: 800px;
                margin: 0 auto;
                padding: 2rem;
            }
            
            /* Radio button styling */
            div[data-testid="stHorizontalBlock"] {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 15px;
                padding: 1rem;
                display: flex;
                justify-content: center;
                gap: 2rem;
                margin-bottom: 1rem;
            }
            
            div.row-widget.stRadio > div {
                flex-direction: row;
                justify-content: center;
                gap: 2rem;
            }
            
            div.row-widget.stRadio > div[role="radiogroup"] > label {
                background: rgba(255, 255, 255, 0.15);
                padding: 0.75rem 2.5rem;
                border-radius: 12px;
                cursor: pointer;
                transition: all 0.3s ease;
                color: #ffffff !important;
                font-weight: 500;
                border: 2px solid transparent;
            }
            
            div.row-widget.stRadio > div[role="radiogroup"] > label:hover {
                background: rgba(23, 198, 216, 0.3);
                border-color: rgba(23, 198, 216, 0.5);
                transform: translateY(-2px);
            }
            
            div.row-widget.stRadio > div[role="radiogroup"] > label[data-checked="true"] {
                background: rgba(23, 198, 216, 0.4);
                border-color: #17c6d8;
            }
            
            /* File uploader styling */
            .uploadFile {
                margin-top: 1rem;
                border: 2px dashed rgba(23, 198, 216, 0.5);
                border-radius: 12px;
                padding: 2rem;
                text-align: center;
                background: rgba(255, 255, 255, 0.08);
            }
            
            section[data-testid="stFileUploadDropzone"] {
                background: rgba(255, 255, 255, 0.08) !important;
                border: 2px dashed rgba(23, 198, 216, 0.5) !important;
                border-radius: 12px !important;
                padding: 2rem !important;
            }
            
            section[data-testid="stFileUploadDropzone"] label,
            section[data-testid="stFileUploadDropzone"] span,
            section[data-testid="stFileUploadDropzone"] button,
            section[data-testid="stFileUploadDropzone"] p,
            section[data-testid="stFileUploadDropzone"] div {
                color: #ffffff !important;
                font-weight: 500 !important;
            }
            
            .upload-text {
                color: #ffffff !important;
                font-size: 1rem !important;
                margin-top: 0.5rem;
                text-align: center;
            }
        </style>
        
        <!-- Navbar -->
        <div class="navbar">
            <div class="navbar-brand">🛡️ Guardians of Truth</div>
            <div class="navbar-menu">
                <a href="?page=home" class="nav-link" style="text-decoration: none;" target="_self">Home</a>
                <a href="?page=about" class="nav-link" style="text-decoration: none;" target="_self">About Us</a>
            </div>
        </div>
        
        <div class="header-container">
            <h1 class="main-title">Deepfake🔍Detection</h1>
            <p class="subtitle">Analyze images and videos for potential deepfake manipulation</p>
        </div>
        
        <style>
        /* AGGRESSIVE OVERRIDES - Force all text and elements to be visible */
        * {
            color: inherit !important;
        }
        
        /* Force all Streamlit widgets to have visible text */
        [class*="st"] label,
        [class*="st"] p,
        [class*="st"] span,
        [class*="st"] div {
            color: #ffffff !important;
        }
        
        /* Radio buttons - nuclear option */
        [data-baseweb="radio"] {
            background: rgba(255, 255, 255, 0.2) !important;
            padding: 1rem !important;
            border-radius: 10px !important;
            border: 2px solid #17c6d8 !important;
            margin: 0.5rem !important;
        }
        
        [data-baseweb="radio"] * {
            color: #ffffff !important;
            font-size: 1.2rem !important;
            font-weight: 600 !important;
        }
        
        /* File uploader - maximum visibility */
        [data-testid="stFileUploader"] > div > div {
            background: rgba(255, 255, 255, 0.15) !important;
            border: 3px dashed #17c6d8 !important;
            padding: 2rem !important;
            border-radius: 15px !important;
            min-height: 150px !important;
        }
        
        [data-testid="stFileUploader"] * {
            color: #ffffff !important;
            font-size: 1.1rem !important;
        }
        
        /* Browse files button */
        button[kind="secondary"] {
            background: #17c6d8 !important;
            color: #000000 !important;
            border: none !important;
            padding: 0.75rem 2rem !important;
            font-weight: 700 !important;
            font-size: 1.1rem !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Input container
    st.markdown("""
        <div style="
            max-width: 900px;
            margin: 0 auto;
            background: rgba(20, 20, 30, 0.85);
            border: 2px solid rgba(23, 198, 216, 0.5);
            border-radius: 20px;
            padding: 2.5rem;
            box-shadow: 0 0 30px rgba(23, 198, 216, 0.2);
            position: relative;
            z-index: 1000;
        ">
        <h3 style="color: #17c6d8; text-align: center; margin-bottom: 1.5rem; font-size: 1.5rem;">
            📤 Upload Media for Analysis
        </h3>
        </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        # Input type selection
        input_options = ["Image", "Video", "News"]
        input_type = st.radio(
            "Select media type to analyze:",
            input_options,
            horizontal=True,
            label_visibility="visible"
        )
        
        # File uploader for image and video, iframe for news
        if input_type == "Image":
            uploaded_file = st.file_uploader(
                "Upload Image",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )
            st.markdown(
                '<p class="upload-text">Supported formats: JPG, JPEG, PNG</p>',
                unsafe_allow_html=True
            )
        elif input_type == "Video":
            uploaded_file = st.file_uploader(
                "Upload Video",
                type=["mp4", "avi", "mov"],
                label_visibility="collapsed"
            )
            st.markdown(
                '<p class="upload-text">Supported formats: MP4, AVI, MOV • Max size: 200MB</p>',
                unsafe_allow_html=True
            )
        else:  # News
            uploaded_file = None
            st.markdown("""
                <div style="
                    margin-top: 1.5rem;
                    border: 2px solid rgba(23, 198, 216, 0.5);
                    border-radius: 12px;
                    overflow: hidden;
                    background: rgba(255, 255, 255, 0.05);
                ">
                    <iframe 
                        src="https://multipurposeagent.netlify.app/" 
                        width="100%" 
                        height="600" 
                        style="border: none; display: block;"
                        title="News"
                    ></iframe>
                </div>
            """, unsafe_allow_html=True)
            
        return input_type, uploaded_file

def show_about_page():
    """Display the About Us page with project and model information"""
    
    # Main header
    st.markdown("""
        <div style="background: rgba(15, 15, 20, 0.98); border: 2px solid rgba(23, 198, 216, 0.5); 
                    border-radius: 20px; padding: 2.5rem; margin: 2rem auto; max-width: 900px; 
                    box-shadow: 0 0 40px rgba(23, 198, 216, 0.3);">
            <h1 style="color: #17c6d8; text-align: center; margin-bottom: 2rem;">
                🛡️ About Guardians of Truth
            </h1>
        </div>
    """, unsafe_allow_html=True)
    
    # About section
    st.markdown("""
        <div style="background: rgba(15, 15, 20, 0.95); border: 2px solid rgba(23, 198, 216, 0.3); 
                    border-radius: 15px; padding: 2rem; margin: 1rem auto; max-width: 900px;">
            <p style="color: #ffffff; font-size: 1.1rem; line-height: 1.8; margin-bottom: 1rem;">
                Guardians of Truth is an advanced deepfake detection system designed to combat 
                the growing threat of AI-generated misinformation. Using state-of-the-art deep 
                learning models, we analyze images and videos to determine their authenticity 
                with industry-leading accuracy.
            </p>
            <p style="color: #ffffff; font-size: 1.1rem; line-height: 1.8;">
                Our mission is to protect digital media integrity and empower users with tools 
                to verify the authenticity of visual content in an era of sophisticated AI manipulation.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Project Overview
    st.markdown("""
        <div style="background: rgba(15, 15, 20, 0.95); border: 2px solid rgba(23, 198, 216, 0.3); 
                    border-radius: 15px; padding: 2rem; margin: 1rem auto; max-width: 900px;">
            <h2 style="color: #17c6d8; font-size: 1.8rem; margin-bottom: 1rem; border-bottom: 2px solid rgba(23, 198, 216, 0.3); padding-bottom: 0.5rem;">
                🎯 Project Overview
            </h2>
            <p style="color: #ffffff; font-size: 1.05rem; line-height: 1.8; margin-bottom: 1rem;">
                This system employs an ensemble approach combining three complementary neural 
                network architectures to achieve robust detection performance:
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Features in columns
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div style="background: rgba(23, 198, 216, 0.1); border: 1px solid rgba(23, 198, 216, 0.3); 
                        border-radius: 12px; padding: 1.5rem; margin: 0.5rem;">
                <h4 style="color: #17c6d8;">✓ Multi-Model Ensemble</h4>
                <p style="color: #ffffff; font-size: 0.95rem;">Combines CNN and Vision Transformer architectures</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div style="background: rgba(23, 198, 216, 0.1); border: 1px solid rgba(23, 198, 216, 0.3); 
                        border-radius: 12px; padding: 1.5rem; margin: 0.5rem;">
                <h4 style="color: #17c6d8;">✓ Face Detection</h4>
                <p style="color: #ffffff; font-size: 0.95rem;">MediaPipe-based with full-frame fallback</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div style="background: rgba(23, 198, 216, 0.1); border: 1px solid rgba(23, 198, 216, 0.3); 
                        border-radius: 12px; padding: 1.5rem; margin: 0.5rem;">
                <h4 style="color: #17c6d8;">✓ Production-Ready</h4>
                <p style="color: #ffffff; font-size: 0.95rem;">CPU optimized with efficient caching</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style="background: rgba(23, 198, 216, 0.1); border: 1px solid rgba(23, 198, 216, 0.3); 
                        border-radius: 12px; padding: 1.5rem; margin: 0.5rem;">
                <h4 style="color: #17c6d8;">✓ Weighted Voting</h4>
                <p style="color: #ffffff; font-size: 0.95rem;">Optimized weights: 0.4, 0.4, 0.2</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div style="background: rgba(23, 198, 216, 0.1); border: 1px solid rgba(23, 198, 216, 0.3); 
                        border-radius: 12px; padding: 1.5rem; margin: 0.5rem;">
                <h4 style="color: #17c6d8;">✓ Video Analysis</h4>
                <p style="color: #ffffff; font-size: 0.95rem;">Temporal aggregation across frames</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div style="background: rgba(23, 198, 216, 0.1); border: 1px solid rgba(23, 198, 216, 0.3); 
                        border-radius: 12px; padding: 1.5rem; margin: 0.5rem;">
                <h4 style="color: #17c6d8;">✓ Free Hosting</h4>
                <p style="color: #ffffff; font-size: 0.95rem;">Streamlit Cloud deployment</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Stats
    st.markdown("""
        <div style="text-align: center; margin: 2rem auto; max-width: 900px;">
            <span style="display: inline-block; background: rgba(23, 198, 216, 0.2); color: #17c6d8; 
                         padding: 0.5rem 1rem; border-radius: 8px; margin: 0.5rem; font-weight: 600; font-size: 1.1rem;">
                93.1% Accuracy
            </span>
            <span style="display: inline-block; background: rgba(23, 198, 216, 0.2); color: #17c6d8; 
                         padding: 0.5rem 1rem; border-radius: 8px; margin: 0.5rem; font-weight: 600; font-size: 1.1rem;">
                3 Models
            </span>
            <span style="display: inline-block; background: rgba(23, 198, 216, 0.2); color: #17c6d8; 
                         padding: 0.5rem 1rem; border-radius: 8px; margin: 0.5rem; font-weight: 600; font-size: 1.1rem;">
                2.5s Inference
            </span>
        </div>
    """, unsafe_allow_html=True)
    
    # Models section
    st.markdown("""
        <div style="background: rgba(15, 15, 20, 0.95); border: 2px solid rgba(23, 198, 216, 0.3); 
                    border-radius: 15px; padding: 2rem; margin: 1rem auto; max-width: 900px;">
            <h2 style="color: #17c6d8; font-size: 1.8rem; margin-bottom: 1.5rem; border-bottom: 2px solid rgba(23, 198, 216, 0.3); padding-bottom: 0.5rem;">
                🤖 Our Models
            </h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Model 1
    st.markdown("""
        <div style="background: rgba(23, 198, 216, 0.1); border: 1px solid rgba(23, 198, 216, 0.3); 
                    border-radius: 12px; padding: 1.5rem; margin: 1rem auto; max-width: 900px;">
            <h3 style="color: #17c6d8; margin-bottom: 1rem;">1. EfficientNet-B3 (Primary Model)</h3>
            <p style="color: #ffffff;"><strong>Architecture:</strong> Compound-scaled CNN with efficient scaling</p>
            <p style="color: #ffffff; margin: 0.5rem 0;"><strong>Performance:</strong></p>
            <ul style="color: #ffffff; margin-left: 1.5rem;">
                <li>Accuracy: 90-92%</li>
                <li>Parameters: 12 million | Size: 50MB</li>
                <li>Input: 300×300×3 RGB | Inference: ~0.3s</li>
            </ul>
            <p style="color: #ffffff;"><strong>Strengths:</strong> Balanced performance, efficient architecture, strong generalization</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Model 2
    st.markdown("""
        <div style="background: rgba(23, 198, 216, 0.1); border: 1px solid rgba(23, 198, 216, 0.3); 
                    border-radius: 12px; padding: 1.5rem; margin: 1rem auto; max-width: 900px;">
            <h3 style="color: #17c6d8; margin-bottom: 1rem;">2. Swin Transformer Base (Highest Accuracy)</h3>
            <p style="color: #ffffff;"><strong>Architecture:</strong> Hierarchical Vision Transformer with shifted windows</p>
            <p style="color: #ffffff; margin: 0.5rem 0;"><strong>Performance:</strong></p>
            <ul style="color: #ffffff; margin-left: 1.5rem;">
                <li>Accuracy: 92-94% (Best single model)</li>
                <li>Parameters: 88 million | Size: 338MB</li>
                <li>Input: 224×224×3 RGB | Inference: ~0.5s</li>
            </ul>
            <p style="color: #ffffff;"><strong>Strengths:</strong> Highest accuracy, captures global context, robust to manipulations</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Model 3
    st.markdown("""
        <div style="background: rgba(23, 198, 216, 0.1); border: 1px solid rgba(23, 198, 216, 0.3); 
                    border-radius: 12px; padding: 1.5rem; margin: 1rem auto; max-width: 900px;">
            <h3 style="color: #17c6d8; margin-bottom: 1rem;">3. Custom Lightweight CNN (Fast Inference)</h3>
            <p style="color: #ffffff;"><strong>Architecture:</strong> 4 convolutional blocks (32→64→128→256)</p>
            <p style="color: #ffffff; margin: 0.5rem 0;"><strong>Performance:</strong></p>
            <ul style="color: #ffffff; margin-left: 1.5rem;">
                <li>Accuracy: 81-83%</li>
                <li>Parameters: 430K (Smallest) | Size: 1.7MB</li>
                <li>Input: 128×128×3 RGB | Inference: ~0.1s</li>
            </ul>
            <p style="color: #ffffff;"><strong>Strengths:</strong> Ultra-fast inference, minimal memory footprint, resource-efficient</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Ensemble Strategy
    st.markdown("""
        <div style="background: rgba(15, 15, 20, 0.95); border: 2px solid rgba(23, 198, 216, 0.3); 
                    border-radius: 15px; padding: 2rem; margin: 1rem auto; max-width: 900px;">
            <h2 style="color: #17c6d8; font-size: 1.8rem; margin-bottom: 1rem;">⚙️ Ensemble Strategy</h2>
            <p style="color: #ffffff; font-size: 1.05rem; line-height: 1.8;">
                Our final prediction combines all three models using weighted voting:
            </p>
            <p style="text-align: center; font-size: 1.3rem; color: #17c6d8; font-family: monospace; margin: 1.5rem 0; font-weight: 600;">
                Final Score = 0.4 × EfficientNet + 0.4 × Swin + 0.2 × Custom CNN
            </p>
            <p style="color: #ffffff; font-size: 1.05rem; line-height: 1.8;">
                <strong>Why Ensemble?</strong> Different architectures capture complementary features. 
                Selected weights based on validation accuracy improvement from 91.2% → 93.1%.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Technical Stack
    st.markdown("""
        <div style="background: rgba(15, 15, 20, 0.95); border: 2px solid rgba(23, 198, 216, 0.3); 
                    border-radius: 15px; padding: 2rem; margin: 1rem auto; max-width: 900px;">
            <h2 style="color: #17c6d8; font-size: 1.8rem; margin-bottom: 1rem;">📊 Technical Stack</h2>
            <ul style="color: #ffffff; font-size: 1.05rem; line-height: 2; margin-left: 1.5rem;">
                <li><strong>Deep Learning:</strong> PyTorch 2.0+, timm (pre-trained models)</li>
                <li><strong>Computer Vision:</strong> OpenCV, MediaPipe, Pillow</li>
                <li><strong>Web Framework:</strong> Streamlit (interactive UI)</li>
                <li><strong>Data Processing:</strong> NumPy, Pandas</li>
                <li><strong>Visualization:</strong> Matplotlib, Plotly</li>
                <li><strong>Deployment:</strong> Streamlit Cloud, Docker support</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Use Cases
    st.markdown("""
        <div style="background: rgba(15, 15, 20, 0.95); border: 2px solid rgba(23, 198, 216, 0.3); 
                    border-radius: 15px; padding: 2rem; margin: 1rem auto; max-width: 900px;">
            <h2 style="color: #17c6d8; font-size: 1.8rem; margin-bottom: 1rem;">🎓 Use Cases</h2>
            <ul style="color: #ffffff; font-size: 1.05rem; line-height: 2; margin-left: 1.5rem;">
                <li><strong>Media Verification:</strong> Journalists and fact-checkers verifying authenticity</li>
                <li><strong>Social Media Monitoring:</strong> Detecting manipulated content before viral spread</li>
                <li><strong>Legal Evidence:</strong> Verifying digital evidence in court proceedings</li>
                <li><strong>Education:</strong> Teaching digital literacy and AI manipulation awareness</li>
                <li><strong>Research:</strong> Studying deepfake detection techniques</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
        <div style="text-align: center; margin: 2rem auto; padding: 2rem; max-width: 900px;">
            <p style="color: #17c6d8; font-size: 1.3rem; font-weight: 600; margin-bottom: 1rem;">
                🛡️ Protecting Truth in the Age of AI 🛡️
            </p>
            <p style="color: #ffffff; font-size: 1rem;">
                Developed with ❤️ by the Guardians of Truth Team
            </p>
        </div>
    """, unsafe_allow_html=True)

def show_models_page():
    """Display the Models page with complete workflow visualization"""
    
    # Main header
    st.markdown("""
        <div style="background: rgba(15, 15, 20, 0.98); border: 2px solid rgba(23, 198, 216, 0.5); 
                    border-radius: 20px; padding: 2.5rem; margin: 2rem auto; max-width: 1200px; 
                    box-shadow: 0 0 40px rgba(23, 198, 216, 0.3);">
            <h1 style="color: #17c6d8; text-align: center; margin-bottom: 1rem;">
                🤖 Complete System Workflow Visualization
            </h1>
            <p style="color: #ffffff; text-align: center; font-size: 1.1rem;">
                From Upload to Final Prediction - How Our 3-Model Ensemble Works
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Step 1: Input
    st.markdown("""
        <div style="background: rgba(15, 15, 20, 0.95); border: 2px solid rgba(23, 198, 216, 0.5); 
                    border-radius: 15px; padding: 2rem; margin: 1.5rem auto; max-width: 1200px;">
            <h2 style="color: #17c6d8; margin-bottom: 1.5rem;">
                📤 Step 1: User Input
            </h2>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(23, 198, 216, 0.2), rgba(14, 165, 233, 0.2)); 
                        border: 2px solid #17c6d8; border-radius: 12px; padding: 2rem; text-align: center; height: 200px; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🖼️</div>
                <h3 style="color: #17c6d8; margin: 0;">Image Upload</h3>
                <p style="color: #ffffff; margin-top: 0.5rem;">JPG, PNG, JPEG</p>
                <p style="color: #aaa; font-size: 0.9rem;">Max: 200MB</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(23, 198, 216, 0.2), rgba(14, 165, 233, 0.2)); 
                        border: 2px solid #17c6d8; border-radius: 12px; padding: 2rem; text-align: center; height: 200px; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🎬</div>
                <h3 style="color: #17c6d8; margin: 0;">Video Upload</h3>
                <p style="color: #ffffff; margin-top: 0.5rem;">MP4, AVI, MOV</p>
                <p style="color: #aaa; font-size: 0.9rem;">10-20 frames sampled</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Arrow down
    st.markdown("""
        <div style="text-align: center; margin: 1rem 0;">
            <div style="font-size: 3rem; color: #17c6d8;">⬇️</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Step 2: Preprocessing
    st.markdown("""
        <div style="background: rgba(15, 15, 20, 0.95); border: 2px solid rgba(23, 198, 216, 0.5); 
                    border-radius: 15px; padding: 2rem; margin: 1.5rem auto; max-width: 1200px;">
            <h2 style="color: #17c6d8; margin-bottom: 1.5rem;">
                🔍 Step 2: Face Detection & Preprocessing
            </h2>
            <div style="background: rgba(23, 198, 216, 0.1); border-radius: 10px; padding: 1.5rem; margin-top: 1rem;">
                <p style="color: #ffffff; font-size: 1.05rem; margin-bottom: 1rem;">
                    <strong>MediaPipe Face Detection</strong> (Confidence > 0.5)
                </p>
                <ul style="color: #ffffff; margin-left: 1.5rem; line-height: 2;">
                    <li>Detect face in image/video frame</li>
                    <li>Extract face region with 20% padding</li>
                    <li>If no face detected → Use full image (fallback)</li>
                    <li>Convert BGR to RGB color space</li>
                </ul>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Arrow down
    st.markdown("""
        <div style="text-align: center; margin: 1rem 0;">
            <div style="font-size: 3rem; color: #17c6d8;">⬇️</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Step 3: Parallel Processing
    st.markdown("""
        <div style="background: rgba(15, 15, 20, 0.95); border: 2px solid rgba(23, 198, 216, 0.5); 
                    border-radius: 15px; padding: 2rem; margin: 1.5rem auto; max-width: 1200px;">
            <h2 style="color: #17c6d8; margin-bottom: 1.5rem;">
                ⚡ Step 3: Parallel Model Processing (3 Models Simultaneously)
            </h2>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div style="background: rgba(23, 198, 216, 0.15); border: 2px solid rgba(23, 198, 216, 0.5); border-radius: 12px; padding: 1.5rem; height: 520px;">', unsafe_allow_html=True)
        st.markdown('<div style="text-align: center; margin-bottom: 1rem;"><div style="font-size: 3rem;">🔷</div><h3 style="color: #17c6d8; margin: 0.5rem 0;">EfficientNet-B3</h3></div>', unsafe_allow_html=True)
        
        st.markdown('<div style="background: rgba(0, 0, 0, 0.3); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">', unsafe_allow_html=True)
        st.markdown('<p style="color: #17c6d8; font-weight: 600; margin-bottom: 0.5rem;">Preprocessing:</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #ffffff; font-size: 0.9rem; margin: 0.3rem 0;">• Resize: 300×300</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #ffffff; font-size: 0.9rem; margin: 0.3rem 0;">• Center Crop</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #ffffff; font-size: 0.9rem; margin: 0.3rem 0;">• ImageNet Normalize</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div style="background: rgba(0, 0, 0, 0.3); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">', unsafe_allow_html=True)
        st.markdown('<p style="color: #17c6d8; font-weight: 600; margin-bottom: 0.5rem;">Model Info:</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #ffffff; font-size: 0.9rem; margin: 0.3rem 0;">• 12M parameters</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #ffffff; font-size: 0.9rem; margin: 0.3rem 0;">• 50MB size</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #ffffff; font-size: 0.9rem; margin: 0.3rem 0;">• ~0.3s inference</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div style="background: rgba(0, 0, 0, 0.3); padding: 1rem; border-radius: 8px;">', unsafe_allow_html=True)
        st.markdown('<p style="color: #17c6d8; font-weight: 600; margin-bottom: 0.5rem;">Output:</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #ffffff; font-size: 0.9rem; margin: 0.3rem 0;">• Accuracy: 90-92%</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #ffffff; font-size: 0.9rem; margin: 0.3rem 0;">• Weight: 0.4</p>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div style="background: rgba(23, 198, 216, 0.15); border: 2px solid rgba(23, 198, 216, 0.5); border-radius: 12px; padding: 1.5rem; height: 520px;">', unsafe_allow_html=True)
        st.markdown('<div style="text-align: center; margin-bottom: 1rem;"><div style="font-size: 3rem;">🔶</div><h3 style="color: #17c6d8; margin: 0.5rem 0;">Swin Transformer</h3></div>', unsafe_allow_html=True)
        
        st.markdown('<div style="background: rgba(0, 0, 0, 0.3); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">', unsafe_allow_html=True)
        st.markdown('<p style="color: #17c6d8; font-weight: 600; margin-bottom: 0.5rem;">Preprocessing:</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #ffffff; font-size: 0.9rem; margin: 0.3rem 0;">• Resize: 224×224</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #ffffff; font-size: 0.9rem; margin: 0.3rem 0;">• Center Crop</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #ffffff; font-size: 0.9rem; margin: 0.3rem 0;">• ImageNet Normalize</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div style="background: rgba(0, 0, 0, 0.3); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">', unsafe_allow_html=True)
        st.markdown('<p style="color: #17c6d8; font-weight: 600; margin-bottom: 0.5rem;">Model Info:</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #ffffff; font-size: 0.9rem; margin: 0.3rem 0;">• 88M parameters</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #ffffff; font-size: 0.9rem; margin: 0.3rem 0;">• 338MB size</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #ffffff; font-size: 0.9rem; margin: 0.3rem 0;">• ~0.5s inference</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div style="background: rgba(0, 0, 0, 0.3); padding: 1rem; border-radius: 8px;">', unsafe_allow_html=True)
        st.markdown('<p style="color: #17c6d8; font-weight: 600; margin-bottom: 0.5rem;">Output:</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #ffffff; font-size: 0.9rem; margin: 0.3rem 0;">• Accuracy: 92-94%</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #ffffff; font-size: 0.9rem; margin: 0.3rem 0;">• Weight: 0.4</p>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div style="background: rgba(23, 198, 216, 0.15); border: 2px solid rgba(23, 198, 216, 0.5); border-radius: 12px; padding: 1.5rem; height: 520px;">', unsafe_allow_html=True)
        st.markdown('<div style="text-align: center; margin-bottom: 1rem;"><div style="font-size: 3rem;">🔹</div><h3 style="color: #17c6d8; margin: 0.5rem 0;">Custom CNN</h3></div>', unsafe_allow_html=True)
        
        st.markdown('<div style="background: rgba(0, 0, 0, 0.3); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">', unsafe_allow_html=True)
        st.markdown('<p style="color: #17c6d8; font-weight: 600; margin-bottom: 0.5rem;">Preprocessing:</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #ffffff; font-size: 0.9rem; margin: 0.3rem 0;">• Resize: 128×128</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #ffffff; font-size: 0.9rem; margin: 0.3rem 0;">• Center Crop</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #ffffff; font-size: 0.9rem; margin: 0.3rem 0;">• [0,1] Normalize</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div style="background: rgba(0, 0, 0, 0.3); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">', unsafe_allow_html=True)
        st.markdown('<p style="color: #17c6d8; font-weight: 600; margin-bottom: 0.5rem;">Model Info:</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #ffffff; font-size: 0.9rem; margin: 0.3rem 0;">• 430K parameters</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #ffffff; font-size: 0.9rem; margin: 0.3rem 0;">• 1.7MB size</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #ffffff; font-size: 0.9rem; margin: 0.3rem 0;">• ~0.1s inference</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div style="background: rgba(0, 0, 0, 0.3); padding: 1rem; border-radius: 8px;">', unsafe_allow_html=True)
        st.markdown('<p style="color: #17c6d8; font-weight: 600; margin-bottom: 0.5rem;">Output:</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #ffffff; font-size: 0.9rem; margin: 0.3rem 0;">• Accuracy: 81-83%</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #ffffff; font-size: 0.9rem; margin: 0.3rem 0;">• Weight: 0.2</p>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
    
    # Each model outputs
    st.markdown("""
        <div style="text-align: center; margin: 1.5rem auto; max-width: 1200px;">
            <div style="background: rgba(23, 198, 216, 0.1); border-radius: 10px; padding: 1rem;">
                <p style="color: #17c6d8; font-weight: 600; font-size: 1.1rem; margin: 0;">
                    Each Model Outputs: [Probability_Real, Probability_Fake]
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Arrow down
    st.markdown("""
        <div style="text-align: center; margin: 1rem 0;">
            <div style="font-size: 3rem; color: #17c6d8;">⬇️</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Step 4: Ensemble
    st.markdown("""
        <div style="background: rgba(15, 15, 20, 0.95); border: 2px solid rgba(23, 198, 216, 0.5); 
                    border-radius: 15px; padding: 2rem; margin: 1.5rem auto; max-width: 1200px;">
            <h2 style="color: #17c6d8; margin-bottom: 1.5rem;">
                🎯 Step 4: Ensemble Aggregation
            </h2>
            
            <div style="background: linear-gradient(135deg, rgba(23, 198, 216, 0.2), rgba(14, 165, 233, 0.2)); 
                        border: 2px solid #17c6d8; border-radius: 12px; padding: 2rem; text-align: center; margin: 1rem 0;">
                <p style="color: #ffffff; font-size: 1.1rem; margin-bottom: 1rem;">Weighted Voting Formula:</p>
                <p style="color: #17c6d8; font-size: 1.5rem; font-family: monospace; font-weight: 700; margin: 1rem 0;">
                    Final Score = 0.4 × EfficientNet + 0.4 × Swin + 0.2 × Custom
                </p>
            </div>
            
            <div style="background: rgba(23, 198, 216, 0.1); border-radius: 10px; padding: 1.5rem; margin-top: 1.5rem;">
                <h3 style="color: #17c6d8; margin-bottom: 1rem;">Example Calculation:</h3>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1rem;">
                    <div style="background: rgba(0, 0, 0, 0.3); padding: 1rem; border-radius: 8px;">
                        <p style="color: #ffffff; margin: 0.3rem 0;">EfficientNet: 0.15</p>
                        <p style="color: #17c6d8; margin: 0.3rem 0;">0.4 × 0.15 = 0.06</p>
                    </div>
                    <div style="background: rgba(0, 0, 0, 0.3); padding: 1rem; border-radius: 8px;">
                        <p style="color: #ffffff; margin: 0.3rem 0;">Swin: 0.12</p>
                        <p style="color: #17c6d8; margin: 0.3rem 0;">0.4 × 0.12 = 0.048</p>
                    </div>
                    <div style="background: rgba(0, 0, 0, 0.3); padding: 1rem; border-radius: 8px;">
                        <p style="color: #ffffff; margin: 0.3rem 0;">Custom: 0.45</p>
                        <p style="color: #17c6d8; margin: 0.3rem 0;">0.2 × 0.45 = 0.09</p>
                    </div>
                </div>
                <p style="color: #ffffff; font-size: 1.2rem; text-align: center; margin: 1rem 0;">
                    Final Score = 0.06 + 0.048 + 0.09 = <span style="color: #17c6d8; font-weight: 700;">0.198</span>
                </p>
                <p style="color: #ffffff; font-size: 1.1rem; text-align: center; margin-top: 1.5rem;">
                    Decision: Score < 0.5 → <span style="color: #00ff00; font-weight: 700; font-size: 1.3rem;">✅ REAL</span>
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Arrow down
    st.markdown("""
        <div style="text-align: center; margin: 1rem 0;">
            <div style="font-size: 3rem; color: #17c6d8;">⬇️</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Step 5: Final Output
    st.markdown("""
        <div style="background: rgba(15, 15, 20, 0.95); border: 2px solid rgba(23, 198, 216, 0.5); 
                    border-radius: 15px; padding: 2rem; margin: 1.5rem auto; max-width: 1200px;">
            <h2 style="color: #17c6d8; margin-bottom: 1.5rem;">
                📊 Step 5: Results Display
            </h2>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div style="background: rgba(23, 198, 216, 0.1); border: 1px solid rgba(23, 198, 216, 0.3); 
                        border-radius: 12px; padding: 1.5rem; height: 280px;">
                <h3 style="color: #17c6d8; text-align: center; margin-bottom: 1rem;">Individual Predictions</h3>
                <div style="background: rgba(0, 0, 0, 0.3); padding: 0.8rem; border-radius: 6px; margin: 0.5rem 0;">
                    <p style="color: #ffffff; margin: 0;">🔷 EfficientNet</p>
                    <p style="color: #00ff00; font-weight: 600; margin: 0;">REAL 85.0%</p>
                </div>
                <div style="background: rgba(0, 0, 0, 0.3); padding: 0.8rem; border-radius: 6px; margin: 0.5rem 0;">
                    <p style="color: #ffffff; margin: 0;">🔶 Swin Transformer</p>
                    <p style="color: #00ff00; font-weight: 600; margin: 0;">REAL 88.0%</p>
                </div>
                <div style="background: rgba(0, 0, 0, 0.3); padding: 0.8rem; border-radius: 6px; margin: 0.5rem 0;">
                    <p style="color: #ffffff; margin: 0;">🔹 Custom CNN</p>
                    <p style="color: #ff6b6b; font-weight: 600; margin: 0;">FAKE 55.0%</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(0, 255, 0, 0.2), rgba(0, 200, 0, 0.2)); 
                        border: 2px solid #00ff00; border-radius: 12px; padding: 1.5rem; text-align: center; height: 280px; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">✅</div>
                <h3 style="color: #00ff00; font-size: 2rem; margin: 0.5rem 0;">REAL</h3>
                <p style="color: #ffffff; font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0;">80.2%</p>
                <p style="color: #aaa; font-size: 1rem; margin-top: 1rem;">Final Ensemble Verdict</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style="background: rgba(23, 198, 216, 0.1); border: 1px solid rgba(23, 198, 216, 0.3); 
                        border-radius: 12px; padding: 1.5rem; height: 280px;">
                <h3 style="color: #17c6d8; text-align: center; margin-bottom: 1rem;">Visualizations</h3>
                <ul style="color: #ffffff; line-height: 2; margin-left: 1rem;">
                    <li>Original Image/Video</li>
                    <li>Face Detection Box</li>
                    <li>Extracted Face</li>
                    <li>Confidence Chart</li>
                    <li>Model Comparison</li>
                    <li>Temporal Analysis (Video)</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # Summary
    st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(23, 198, 216, 0.2), rgba(14, 165, 233, 0.2)); 
                    border: 2px solid #17c6d8; border-radius: 15px; padding: 2rem; margin: 2rem auto; max-width: 1200px; text-align: center;">
            <h2 style="color: #17c6d8; margin-bottom: 1rem;">⚡ Complete Process Time</h2>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 1.5rem;">
                <div style="background: rgba(0, 0, 0, 0.3); padding: 1rem; border-radius: 8px;">
                    <p style="color: #17c6d8; font-size: 1.5rem; font-weight: 700; margin: 0;">~0.5s</p>
                    <p style="color: #ffffff; font-size: 0.9rem; margin: 0.5rem 0;">Preprocessing</p>
                </div>
                <div style="background: rgba(0, 0, 0, 0.3); padding: 1rem; border-radius: 8px;">
                    <p style="color: #17c6d8; font-size: 1.5rem; font-weight: 700; margin: 0;">~0.9s</p>
                    <p style="color: #ffffff; font-size: 0.9rem; margin: 0.5rem 0;">Model Inference</p>
                </div>
                <div style="background: rgba(0, 0, 0, 0.3); padding: 1rem; border-radius: 8px;">
                    <p style="color: #17c6d8; font-size: 1.5rem; font-weight: 700; margin: 0;">~0.1s</p>
                    <p style="color: #ffffff; font-size: 0.9rem; margin: 0.5rem 0;">Ensemble</p>
                </div>
                <div style="background: rgba(0, 0, 0, 0.3); padding: 1rem; border-radius: 8px;">
                    <p style="color: #00ff00; font-size: 1.5rem; font-weight: 700; margin: 0;">~2.5s</p>
                    <p style="color: #ffffff; font-size: 0.9rem; margin: 0.5rem 0;">Total (Image)</p>
                </div>
            </div>
            <p style="color: #aaa; margin-top: 1.5rem; font-size: 0.95rem;">
                Video processing: ~25s for 10 frames | Combined Accuracy: 93.1%
            </p>
        </div>
    """, unsafe_allow_html=True)

def main():
    init_session_state()

    if check_session_timeout():
        st.warning("Your session has timed out. Please reload the page.")
        return

    # Initialize current page state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'

    # Check URL parameters for page navigation (do this first)
    try:
        query_params = st.query_params
        if 'page' in query_params:
            page_param = query_params['page']
            if page_param == 'about':
                st.session_state.current_page = 'about'
            elif page_param == 'home':
                st.session_state.current_page = 'home'
    except Exception as e:
        pass  # If query params fail, use session state
    
    # Add navigation buttons in sidebar
    with st.sidebar:
        st.markdown("### 📍 Navigation")
        if st.button("🏠 Home", use_container_width=True, key="nav_home"):
            st.session_state.current_page = 'home'
            st.query_params.clear()
            st.rerun()
        
        if st.button("ℹ️ About Us", use_container_width=True, key="nav_about"):
            st.session_state.current_page = 'about'
            st.query_params['page'] = 'about'
            st.rerun()

    # Show back button if not on the home page
    if st.session_state.current_page not in ['home', 'about']:
        if st.button('← Back to Home'):
            clear_session_state()
            st.session_state.current_page = 'home'
            st.query_params.clear()
            st.rerun()

    # Display appropriate page
    if st.session_state.current_page == 'home':
        input_type, uploaded_file = show_home_page()
        # Only process if not News and file is uploaded
        if input_type != 'News' and uploaded_file is not None:
            st.session_state.input_type = input_type
            st.session_state.uploaded_file = uploaded_file
            st.session_state.current_page = 'results'
            st.rerun()
    elif st.session_state.current_page == 'about':
        show_about_page()
    elif st.session_state.current_page == 'results':
        with cleanup_on_exit():
            if st.session_state.input_type == "Image":
                process_image_input(st.session_state.uploaded_file)
            else:
                process_video_input(st.session_state.uploaded_file)

if __name__ == "__main__":
    main()