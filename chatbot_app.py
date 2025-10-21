"""
SmartRAG ChatBot - Minimalistic ChatGPT-style Interface
A clean, modern chat interface for the SmartRAG system with file management.
"""

import streamlit as st
import os
import time
from datetime import datetime
from pathlib import Path
import json
from typing import List, Dict
import tempfile

# Import SmartRAG system
from multimodal_rag.system import MultimodalRAGSystem
from multimodal_rag.base import QueryRequest

# Import new configuration system
try:
    from config_schema import load_config, SmartRAGConfig
    USE_NEW_CONFIG = True
except ImportError:
    USE_NEW_CONFIG = False
    print("⚠️  config_schema not found, using legacy configuration")

# Import for speech-to-text
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# Import for file handling and database
import sqlite3
import base64
from io import BytesIO
from PIL import Image
import mimetypes

# Page configuration
st.set_page_config(
    page_title="SmartRAG ChatBot",
    page_icon="🤖",  # Traditional bot icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-like styling
st.markdown("""
<style>
    /* Main chat interface background and text */
    /* Target the main page body/container */
    .stApp {
        background-color: #f0f2f6 !important; /* Very light grey/off-white for the main background */
        color: #333333 !important; /* Dark text for general page content */
    }

    /* Sidebar styling (already somewhat light, but confirming) */
    .css-1d391kg, .css-1544g2n, .stSidebar,
    .css-1lcbmhc, .css-12oz5g7, .css-1y4p8pa,
    section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
        background-color: #ffffff !important; /* White sidebar */
        color: #333333 !important; /* Dark text in sidebar */
        min-width: 250px !important;
    }

    /* Sidebar toggle button (using a clear blue accent) */
    button[kind="header"], .css-1rs6os button, [data-testid="collapsedControl"],
    section[data-testid="stSidebar"] button[kind="header"] {
        background-color: #4a90e2 !important;
        color: white !important;
        border-radius: 8px !important;
    }

    /* Collapsed sidebar */
    section[data-testid="stSidebar"][aria-expanded="false"] {
        width: 50px !important;
        min-width: 50px !important;
    }

    /* Hide Streamlit defaults */
    #MainMenu, footer, header {visibility: hidden;}

    /* Custom button (clear blue) */
    .stButton > button {
        background-color: #4a90e2;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        background-color: #357abd;
    }

    /* System Status / Ready / Speech-to-Text Available boxes (light backgrounds with clear text) */
    .stSuccess {
        background-color: #e6f0fa; /* Very light blue */
        color: #1f4e79; /* Dark blue text */
        border: 1px solid #4a90e2;
        border-radius: 8px;
    }
    .stError {
        background-color: #fcebea; /* Very light red */
        color: #721c24; /* Dark red text */
        border: 1px solid #e24a4a;
        border-radius: 8px;
    }

    /* File uploader area */
    .stFileUploader {
        border: 2px dashed #4a90e2;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        background: #ffffff; /* White background */
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #4a90e2;
    }

    /* Chat with your Documents title area / Main area background (was dark blue/green) */
    /* Targeting the main container where the "Chat with your Documents" header lives */
    /* This might be a container div that was set to dark. We'll set it to white/light grey */
    /* If the header itself is in a custom component, you might need to find its specific class. */
    /* Assuming a general container for the main chat area */
    .main-chat-container {
        background-color: #ffffff !important; /* White background for the main card/area */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05); /* Soft shadow */
    }

    /* The "SmartRAG system initialized successfully!" box (was dark green) */
    /* This uses the stSuccess styling, which is now light blue/white */
    
    /* The SmartRAG welcome box (was dark blue) */
    .st-bv .stMarkdown { /* Target stMarkdown if it's used for this box */
        background-color: #e6f0fa !important; /* Light blue background for info box */
        color: #1f4e79 !important; /* Dark text */
        padding: 15px !important;
        border-radius: 10px !important;
        border-left: 4px solid #4a90e2 !important;
    }

    /* The "Drag and drop file here" box (was using stFileUploader) */
    /* It is now white with blue dashed border */


    /* Chat messages - User messages (Using light green for a clear contrast) */
    .user-message {
        background: #dcf8c6 !important; /* Very light green bubble */
        color: #333333 !important; /* Dark text */
        padding: 15px !important;
        border-radius: 15px !important;
        margin: 10px 0 !important;
        border-left: 4px solid #4caf50 !important; /* Clear green accent */
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }

    /* Chat messages - Assistant messages (Using light blue for a clear contrast) */
    .assistant-message {
        background: #e9f5ff !important; /* Very light blue bubble */
        color: #333333 !important; /* Dark text */
        padding: 15px !important;
        border-radius: 15px !important;
        margin: 10px 0 !important;
        border-left: 4px solid #4a90e2 !important; /* Clear blue accent */
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }

    /* Input styling - Use light theme colors for the text input */
    .stTextInput > div > div > input {
        background-color: #ffffff;
        border: 2px solid #cccccc; /* Soft grey border */
        border-radius: 25px;
        color: #000000; /* Black text */
        padding: 12px 20px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #4a90e2; /* Blue border on focus */
        box-shadow: 0 0 10px rgba(74, 144, 226, 0.3);
        outline: none;
    }

    /* Chat input container */
    .chat-input-container {
        display: flex;
        align-items: center;
        background: #ffffff; /* White background for the bar */
        border: 2px solid #ccc; /* Light border */
        border-radius: 25px;
        padding: 8px 12px;
        margin: 20px auto;
        max-width: 800px;
    }

    /* Expander headers */
    .streamlit-expanderHeader {
        background: #f1f1f1; /* Light grey header */
        color: #333333; /* Dark text */
        border-radius: 8px;
    }

    /* Ensure all text elements have proper contrast in light theme */
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50 !important; /* Dark blue-grey for headers */
    }

    .stMarkdown {
        color: #333333 !important; /* Dark text for markdown */
    }

    .stText {
        color: #333333 !important; /* Dark text for general text */
    }

    /* Info boxes and alerts */
    .stInfo {
        background-color: #e6f3ff !important;
        color: #0066cc !important;
        border: 1px solid #b3d9ff !important;
    }

    .stWarning {
        background-color: #fff3cd !important;
        color: #856404 !important;
        border: 1px solid #ffeaa7 !important;
    }

    /* Ensure form submit buttons are visible */
    .stForm button {
        background-color: #4a90e2 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: 500 !important;
        margin-top: 10px !important;
    }

    .stForm button:hover {
        background-color: #357abd !important;
    }

    /* Chat container styling for light theme - remove frame */
    .chat-container {
        background-color: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Sources text in responses */
    .assistant-message small {
        color: #555555 !important;
        font-style: italic;
    }

    /* Main title and subtitle styling */
    [data-testid="stMarkdownContainer"] h1 {
        color: #2c3e50 !important;
    }

    /* Italic subtitle styling */
    [data-testid="stMarkdownContainer"] em {
        color: #555555 !important;
    }

    /* File uploader text */
    .stFileUploader label {
        color: #333333 !important;
    }

    /* Selectbox and other input labels */
    .stSelectbox label, .stTextInput label, .stTextArea label {
        color: #333333 !important;
    }
    
    /* File uploader styling for light theme - fix dark dropzone */
    .stFileUploader [data-testid="stFileUploaderDropzone"] {
        background-color: #f5f5f5 !important;
        color: #333333 !important;
        border: 2px dashed #4a90e2 !important;
    }
    
    .stFileUploader [data-testid="stFileUploaderDropzone"] * {
        color: #333333 !important;
    }
    
    /* Fix file uploader instruction text */
    .stFileUploader [data-testid="stFileUploaderInstructions"] {
        color: #333333 !important;
    }
    
    /* Button text colors for light gray buttons */
    button[style*="background-color: rgb(43, 43, 43)"] {
        background-color: #e0e0e0 !important;
        color: #333333 !important;
        border: 1px solid #cccccc !important;
    }
    
    /* Sidebar button styling */
    .css-1d391kg button {
        background-color: #e0e0e0 !important;
        color: #333333 !important;
    }
    
    /* Status container text should be visible */
    .element-container[style*="background"] {
        color: white !important;
    }
    
    .element-container[style*="background"] * {
        color: white !important;
    }
    
    /* Fix secondary buttons and dark containers */
    .stButton button[kind="secondary"] {
        background-color: #e0e0e0 !important;
        color: #333333 !important;
        border: 1px solid #cccccc !important;
    }
    
    /* Fix expander headers that might be dark */
    .streamlit-expanderHeader {
        background-color: #f5f5f5 !important;
        color: #333333 !important;
    }
    
    /* Additional targeting for expander text visibility */
    .streamlit-expanderHeader p,
    .streamlit-expanderHeader span,
    .streamlit-expanderHeader div,
    .streamlit-expanderHeader summary {
        color: #333333 !important;
    }
    
    /* Target Streamlit's data-testid attributes for expanders */
    [data-testid="stExpanderDetails"] summary {
        color: #333333 !important;
    }
    
    [data-testid="stExpanderDetails"] summary > div {
        color: #333333 !important;
    }
    
    /* Fix any potential white text in expanders */
    .streamlit-expanderHeader * {
        color: #333333 !important;
    }
    
    /* Fix any sidebar components that might be dark */
    .css-1d391kg .stSelectbox > div > div {
        background-color: #f5f5f5 !important;
        color: #333333 !important;
    }
    
    /* Fix Data Management section */
    div[data-testid="column"] button {
        background-color: #e0e0e0 !important;
        color: #333333 !important;
    }
    
    /* Additional file uploader fixes */
    .stFileUploader {
        background-color: #f5f5f5 !important;
        color: #333333 !important;
    }
    
    .stFileUploader label {
        color: #333333 !important;
    }
    
    /* Upload Documents expander header */
    .stExpander [data-testid="stExpanderToggleIcon"] + div {
        background-color: #e8e8e8 !important;
        color: #333333 !important;
        border-radius: 8px !important;
        padding: 8px !important;
    }
    
    /* Expander summary/header */
    .stExpander summary {
        background-color: #e8e8e8 !important;
        color: #333333 !important;
        border-radius: 8px !important;
        padding: 10px !important;
        font-weight: bold !important;
    }
    
    /* Fix Streamlit metrics text color - targeting the sidebar status elements */
    .stMetric {
        color: #333333 !important;
    }
    
    .stMetric label {
        color: #333333 !important;
    }
    
    .stMetric [data-testid="metric-label"] {
        color: #333333 !important;
    }
    
    .stMetric [data-testid="metric-value"] {
        color: #333333 !important;
    }
    
    /* Additional targeting for sidebar metrics */
    .css-1d391kg .stMetric,
    .css-1d391kg .stMetric label,
    .css-1d391kg .stMetric div {
        color: #333333 !important;
    }
    
    /* More comprehensive metric targeting */
    [data-testid="metric-container"],
    [data-testid="metric-container"] *,
    .metric-container,
    .metric-container *,
    .stMetric *,
    .metric * {
        color: #333333 !important;
    }
    
    /* Target metric values specifically */
    .metric .metric-value,
    [data-testid="metric-value"],
    .stMetric-value,
    .css-1xarl3l,
    .css-16huue1 {
        color: #333333 !important;
    }
    
    /* Sidebar specific overrides */
    .css-1d391kg * {
        color: #333333 !important;
    }
    
    /* File uploader browse button */
    .stFileUploader button,
    .stFileUploader [data-testid="baseButton-secondary"] {
        background-color: #e8e8e8 !important;
        color: #333333 !important;
        border: 1px solid #cccccc !important;
        border-radius: 6px !important;
    }
    
    /* All secondary buttons (like Browse Files) */
    button[kind="secondary"] {
        background-color: #e8e8e8 !important;
        color: #333333 !important;
        border: 1px solid #cccccc !important;
    }
</style>
""", unsafe_allow_html=True)

# Custom CSS and JavaScript for sidebar and light theme fixes
st.markdown("""
<style>
    /* Force dark text for main content areas but preserve white text in dark containers */
    .main .block-container {
        color: #333333 !important;
    }
    
    /* Preserve white text in dark containers */
    .main .block-container [style*="background-color: rgb(43, 43, 43)"] * {
        color: white !important;
    }
    
    .main .block-container [style*="background-color: rgb(70, 130, 180)"] * {
        color: white !important;
    }
    
    .main .block-container [style*="background-color: rgb(34, 139, 34)"] * {
        color: white !important;
    }
    
    /* Fix sidebar text visibility */
    .css-1d391kg {
        color: #333333 !important;
    }
    
    /* But preserve white text in dark sidebar elements */
    .css-1d391kg [style*="background-color: rgb(43, 43, 43)"] * {
        color: white !important;
    }
    
    .css-1d391kg [style*="background-color: rgb(70, 130, 180)"] * {
        color: white !important;
    }
    
    .css-1d391kg [style*="background-color: rgb(34, 139, 34)"] * {
        color: white !important;
    }
</style>

<script>
// Ensure sidebar toggle is always accessible
document.addEventListener('DOMContentLoaded', function() {
    // Find and ensure sidebar toggle is visible
    function ensureSidebarToggle() {
        const toggleButtons = document.querySelectorAll('button[kind="header"], button[data-testid="collapsedControl"]');
        toggleButtons.forEach(button => {
            if (button) {
                button.style.display = 'block';
                button.style.visibility = 'visible';
                button.style.opacity = '1';
            }
        });
    }
    
    ensureSidebarToggle();
    setInterval(ensureSidebarToggle, 1000);
});
</script>
""", unsafe_allow_html=True)

# File Storage Database Functions
def init_file_storage_db():
    """Initialize SQLite database for file storage."""
    conn = sqlite3.connect('file_storage.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stored_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            file_content BLOB NOT NULL,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_hash TEXT UNIQUE NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()

def store_file_in_db(filename: str, file_content: bytes, file_type: str, file_size: int, file_hash: str, upload_time: str = None) -> bool:
    """Store file in SQLite database."""
    from datetime import datetime
    try:
        conn = sqlite3.connect('file_storage.db')
        cursor = conn.cursor()
        
        # Use provided upload_time or current local time
        if upload_time is None:
            upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.execute('''
            INSERT OR REPLACE INTO stored_files 
            (filename, file_type, file_size, file_content, file_hash, upload_time)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (filename, file_type, file_size, file_content, file_hash, upload_time))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error storing file in database: {str(e)}")
        return False

def get_file_from_db(filename: str) -> tuple:
    """Retrieve file from SQLite database."""
    try:
        conn = sqlite3.connect('file_storage.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT filename, file_type, file_size, file_content, upload_time
            FROM stored_files 
            WHERE filename = ?
            ORDER BY upload_time DESC
            LIMIT 1
        ''', (filename,))
        
        result = cursor.fetchone()
        conn.close()
        return result
    except Exception as e:
        st.error(f"Error retrieving file from database: {str(e)}")
        return None

def get_all_stored_files() -> list:
    """Get list of all stored files."""
    try:
        conn = sqlite3.connect('file_storage.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT filename, file_type, file_size, upload_time
            FROM stored_files 
            ORDER BY upload_time DESC
        ''')
        
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        st.error(f"Error getting file list: {str(e)}")
        return []

def get_files_by_time_range(start_time: str, end_time: str = None) -> list:
    """Get files uploaded within a specific time range."""
    try:
        conn = sqlite3.connect('file_storage.db')
        cursor = conn.cursor()
        
        if end_time:
            cursor.execute('''
                SELECT filename, file_type, file_size, upload_time
                FROM stored_files 
                WHERE upload_time BETWEEN ? AND ?
                ORDER BY upload_time DESC
            ''', (start_time, end_time))
        else:
            cursor.execute('''
                SELECT filename, file_type, file_size, upload_time
                FROM stored_files 
                WHERE upload_time >= ?
                ORDER BY upload_time DESC
            ''', (start_time,))
        
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        st.error(f"Error getting files by time range: {str(e)}")
        return []

def search_files_by_timestamp(query_time: str) -> str:
    """Search for files uploaded around a specific time and return formatted response."""
    from datetime import datetime, timedelta
    import re
    
    try:
        # Parse different time formats
        time_patterns = [
            r'(\d{1,2}):(\d{2})',  # HH:MM
            r'(\d{1,2})\.(\d{2})',  # HH.MM
            r'(\d{1,2}):(\d{2}):(\d{2})',  # HH:MM:SS
        ]
        
        hour, minute = None, None
        for pattern in time_patterns:
            match = re.search(pattern, query_time)
            if match:
                hour = int(match.group(1))
                minute = int(match.group(2))
                break
        
        if hour is None or minute is None:
            return "❌ Could not parse time format. Please use formats like '14:20', '2:30 PM', or '14.20'."
        
        # Get today's date with the specified time
        today = datetime.now().date()
        target_time = datetime.combine(today, datetime.min.time().replace(hour=hour, minute=minute))
        
        # Search within ±30 minutes of the target time
        start_time = target_time - timedelta(minutes=30)
        end_time = target_time + timedelta(minutes=30)
        
        # Query database
        conn = sqlite3.connect('file_storage.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT filename, file_type, upload_time
            FROM stored_files 
            WHERE upload_time BETWEEN ? AND ?
            ORDER BY ABS(strftime('%s', upload_time) - strftime('%s', ?))
        ''', (start_time.strftime('%Y-%m-%d %H:%M:%S'), 
              end_time.strftime('%Y-%m-%d %H:%M:%S'),
              target_time.strftime('%Y-%m-%d %H:%M:%S')))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return f"No documents were uploaded around {query_time}."
        
        # For single result, give direct answer
        if len(results) == 1:
            filename, file_type, upload_time = results[0]
            upload_dt = datetime.strptime(upload_time, '%Y-%m-%d %H:%M:%S')
            formatted_time = upload_dt.strftime('%H:%M')
            return f"At {query_time}, you uploaded **{filename}** ({file_type}) at {formatted_time}."
        
        # For multiple results, list them
        response = f"Around {query_time}, you uploaded:\n\n"
        for filename, file_type, upload_time in results:
            upload_dt = datetime.strptime(upload_time, '%Y-%m-%d %H:%M:%S')
            formatted_time = upload_dt.strftime('%H:%M')
            response += f"• **{filename}** at {formatted_time}\n"
        
        return response
        
    except Exception as e:
        return f"❌ Error searching files by timestamp: {str(e)}"

# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    if 'whisper_model' not in st.session_state:
        st.session_state.whisper_model = None
    if 'transcribed_text' not in st.session_state:
        st.session_state.transcribed_text = ""
    if 'input_hash' not in st.session_state:
        st.session_state.input_hash = ""
    if 'viewing_file' not in st.session_state:
        st.session_state.viewing_file = None
    if 'show_file_viewer' not in st.session_state:
        st.session_state.show_file_viewer = False
    
    # Initialize file storage database
    init_file_storage_db()
    if 'audio_processing' not in st.session_state:
        st.session_state.audio_processing = False
    if 'show_audio_upload' not in st.session_state:
        st.session_state.show_audio_upload = False
    if 'transcribed_message' not in st.session_state:
        st.session_state.transcribed_message = ""
    if 'last_processed_input' not in st.session_state:
        st.session_state.last_processed_input = ""
    if 'input_hash' not in st.session_state:
        st.session_state.input_hash = ""

def get_whisper_model():
    """Get or initialize the Whisper model."""
    if st.session_state.whisper_model is None and WHISPER_AVAILABLE:
        with st.spinner("🎙️ Loading Whisper model..."):
            try:
                st.session_state.whisper_model = whisper.load_model("base")
                st.success("✅ Whisper model loaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to load Whisper model: {str(e)}")
                return None
    return st.session_state.whisper_model

def transcribe_audio(audio_bytes):
    """Transcribe audio bytes using Whisper."""
    if not WHISPER_AVAILABLE:
        return "Error: Whisper not available"
    
    whisper_model = get_whisper_model()
    if whisper_model is None:
        return "Error: Could not load Whisper model"
    
    try:
        # Save audio bytes to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
        
        # Transcribe with Whisper
        st.session_state.audio_processing = True
        result = whisper_model.transcribe(temp_audio_path)
        transcript = result.get('text', '').strip()
        
        # Clean up temp file
        os.unlink(temp_audio_path)
        st.session_state.audio_processing = False
        
        return transcript if transcript else "No speech detected"
        
    except Exception as e:
        st.session_state.audio_processing = False
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        return f"Error transcribing audio: {str(e)}"

def render_file_viewer():
    """Render file viewer for displaying stored documents."""
    if st.session_state.viewing_file:
        filename = st.session_state.viewing_file
        file_data = get_file_from_db(filename)
        
        if file_data:
            filename, file_type, file_size, file_content, upload_time = file_data
            
            # Create file viewer modal
            with st.container():
                col1, col2, col3 = st.columns([1, 8, 1])
                with col2:
                    st.markdown(f"### 📄 Viewing: {filename}")
                    st.markdown(f"**Type:** {file_type} | **Size:** {file_size:,} bytes | **Uploaded:** {upload_time}")
                    
                    # Close button
                    if st.button("❌ Close Viewer", key="close_viewer"):
                        st.session_state.viewing_file = None
                        st.session_state.show_file_viewer = False
                        st.rerun()
                    
                    st.markdown("---")
                    
                    # Display file content based on type
                    if file_type.startswith('image/'):
                        # Display image
                        image = Image.open(BytesIO(file_content))
                        st.image(image, caption=filename, use_column_width=True)
                        
                    elif file_type == 'application/pdf':
                        # Enhanced PDF viewer with embedded display
                        st.markdown("**📄 PDF Document**")
                        
                        # Create tabs for different viewing options
                        tab1, tab2 = st.tabs(["📖 View PDF", "📥 Download"])
                        
                        with tab1:
                            try:
                                # Create base64 encoded PDF for embedded viewing
                                import base64
                                base64_pdf = base64.b64encode(file_content).decode('utf-8')
                                
                                # Embed PDF using iframe
                                pdf_display = f"""
                                <div style="border: 2px solid #4a90e2; border-radius: 10px; overflow: hidden;">
                                    <iframe src="data:application/pdf;base64,{base64_pdf}" 
                                            width="100%" height="600" 
                                            style="border: none;"
                                            type="application/pdf">
                                        <p style="padding: 20px; text-align: center;">
                                            Your browser does not support inline PDF viewing. 
                                            Please use the Download tab to get the PDF file.
                                        </p>
                                    </iframe>
                                </div>
                                """
                                st.markdown(pdf_display, unsafe_allow_html=True)
                                
                                # Additional controls
                                col_pdf1, col_pdf2 = st.columns(2)
                                with col_pdf1:
                                    if st.button("🔗 Open in New Tab", key="open_new_tab"):
                                        st.markdown(f"""
                                        <script>
                                            window.open("data:application/pdf;base64,{base64_pdf}", "_blank");
                                        </script>
                                        """, unsafe_allow_html=True)
                                        st.success("PDF opened in new tab!")
                                
                                with col_pdf2:
                                    st.download_button(
                                        label="� Save PDF",
                                        data=file_content,
                                        file_name=filename,
                                        mime=file_type,
                                        key="pdf_download_inline"
                                    )
                                    
                            except Exception as e:
                                st.error(f"❌ Error displaying PDF: {str(e)}")
                                st.info("📋 Please use the Download tab to access the PDF.")
                        
                        with tab2:
                            st.markdown("### 📥 Download Options")
                            st.download_button(
                                label="📥 Download PDF",
                                data=file_content,
                                file_name=filename,
                                mime=file_type,
                                use_container_width=True,
                                key="pdf_download_tab"
                            )
                            st.markdown(f"""
                            **File Information:**
                            - 📄 **Name**: {filename}
                            - 📏 **Size**: {file_size:,} bytes ({file_size/1024:.1f} KB)
                            - 🕒 **Uploaded**: {upload_time}
                            """)
                            st.info("💡 **Tip**: You can also view the PDF directly in the 'View PDF' tab above!")
                        
                    elif file_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 
                                     'application/msword']:
                        # Display Word document download link
                        st.markdown("**Word Document**")
                        st.download_button(
                            label="📥 Download Document",
                            data=file_content,
                            file_name=filename,
                            mime=file_type
                        )
                        st.info("Word documents can be downloaded and viewed in Microsoft Word or compatible applications.")
                        
                    elif file_type.startswith('audio/'):
                        # Display audio player
                        st.markdown("**Audio File**")
                        st.audio(file_content, format=file_type)
                        
                    elif file_type.startswith('text/'):
                        # Display text content
                        try:
                            text_content = file_content.decode('utf-8')
                            st.text_area("File Content:", text_content, height=400)
                        except:
                            st.error("Unable to display text content")
                            st.download_button(
                                label="📥 Download File",
                                data=file_content,
                                file_name=filename,
                                mime=file_type
                            )
                    else:
                        # Generic file download
                        st.markdown("**File Download**")
                        st.download_button(
                            label="📥 Download File",
                            data=file_content,
                            file_name=filename,
                            mime=file_type
                        )
        else:
            st.error(f"File '{filename}' not found in database.")
            if st.button("❌ Close"):
                st.session_state.viewing_file = None
                st.session_state.show_file_viewer = False
                st.rerun()

def render_custom_chat_input():
    """Render clean chat input bar with speech-to-text functionality."""
    
    # Create a clean input container
    col1, col2, col3 = st.columns([1, 10, 1])
    
    with col1:
        # Microphone/Speech-to-text button
        if WHISPER_AVAILABLE:
            if st.button("🎙️", help="Upload audio for speech-to-text", key="stt_btn"):
                st.session_state.show_audio_upload = True
        else:
            st.button("🎙️", disabled=True, help="Whisper not available - install openai-whisper")
    
    with col2:
        # Use a form to prevent continuous re-evaluation with unique key based on message count
        message_count = len(st.session_state.messages)
        with st.form(key=f"chat_form_{message_count}", clear_on_submit=True):
            text_message = st.text_input(
                "message",
                placeholder="Type your message or upload audio for speech-to-text...",
                label_visibility="collapsed",
                key=f"chat_input_field_{message_count}"
            )
            submit_button = st.form_submit_button("Send", use_container_width=True)
            
            if submit_button and text_message.strip():
                return text_message.strip()
    
    with col3:
        # Visual send indicator (form handles submission now)
        st.markdown("<div style='text-align: center; padding: 8px; color: #333333;'>📝</div>", unsafe_allow_html=True)
    
    # Audio upload modal when microphone is clicked - more compact
    if st.session_state.get('show_audio_upload', False):
        with st.expander("🎙️ Speech-to-Text", expanded=True):
            audio_file = st.file_uploader(
                "Choose an audio file",
                type=['wav', 'mp3', 'm4a', 'ogg', 'flac', 'webm'],
                key=f"speech_to_text_upload_{message_count}"
            )
            
            col_transcribe, col_cancel = st.columns([1, 1])
            
            with col_transcribe:
                if audio_file and st.button("🎯 Transcribe", key="transcribe_btn"):
                    with st.spinner("🎙️ Converting speech to text..."):
                        transcript = transcribe_audio_file(audio_file)
                    if transcript and not transcript.startswith("Error"):
                        st.session_state.show_audio_upload = False
                        st.session_state.transcribed_message = transcript
                        st.rerun()
                    elif transcript:
                        st.error(transcript)
            
            with col_cancel:
                if st.button("❌ Cancel", key="cancel_stt"):
                    st.session_state.show_audio_upload = False
                    st.rerun()
    
    # Return transcribed message if available
    if st.session_state.get('transcribed_message', ''):
        result = st.session_state.transcribed_message
        st.session_state.transcribed_message = ""
        return result
    
    return None

def transcribe_audio_file(audio_file):
    """Transcribe an uploaded audio file."""
    if not WHISPER_AVAILABLE:
        st.error("Whisper not available")
        return None
    
    whisper_model = get_whisper_model()
    if whisper_model is None:
        st.error("Could not load Whisper model")
        return None
    
    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_file.read())
            temp_path = temp_file.name
        
        # Transcribe
        with st.spinner("🎙️ Transcribing..."):
            result = whisper_model.transcribe(temp_path)
            transcript = result.get('text', '').strip()
        
        # Cleanup
        os.unlink(temp_path)
        
        if transcript:
            st.success(f"🎙️ Transcribed: {transcript}")
            return transcript
        else:
            st.warning("No speech detected")
            return None
            
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return None

def get_rag_system():
    """Get or initialize the RAG system with new configuration."""
    if st.session_state.rag_system is None:
        with st.spinner("🔧 Initializing SmartRAG system..."):
            try:
                # Load and apply environment variable overrides if available
                if USE_NEW_CONFIG:
                    # Allow runtime overrides from environment or session state
                    config = load_config(
                        config_path="config.yaml",
                        # Example: can override from session state or env vars
                    )
                    st.session_state.rag_system = MultimodalRAGSystem(config_path="config.yaml")
                    st.info("✅ Using validated configuration schema")
                else:
                    st.session_state.rag_system = MultimodalRAGSystem(config_path="config.yaml")
                
                st.session_state.system_initialized = True
                st.success("✅ SmartRAG system initialized successfully!")
            except Exception as e:
                st.error(f"❌ Failed to initialize SmartRAG: {str(e)}")
                st.session_state.system_initialized = False
                return None
    return st.session_state.rag_system

def save_uploaded_files_list():
    """Save the list of uploaded files to disk."""
    try:
        os.makedirs("user_data", exist_ok=True)
        with open("user_data/streamlit_uploaded_files.json", "w") as f:
            json.dump(st.session_state.uploaded_files, f, indent=2, default=str)
    except Exception as e:
        st.error(f"Error saving uploaded files list: {e}")

def load_uploaded_files_list():
    """Load the list of uploaded files from disk."""
    try:
        if os.path.exists("user_data/streamlit_uploaded_files.json"):
            with open("user_data/streamlit_uploaded_files.json", "r") as f:
                st.session_state.uploaded_files = json.load(f)
    except Exception as e:
        st.error(f"Error loading uploaded files list: {e}")

def process_uploaded_file(uploaded_file, rag_system):
    """Process a single uploaded file."""
    try:
        file_type = uploaded_file.type
        
        # For images, process with OCR and BLIP captioning
        if file_type.startswith('image/'):
            try:
                # Reset file pointer to beginning
                uploaded_file.seek(0)
                
                # Save image temporarily for processing
                temp_dir = Path("temp_uploads")
                temp_dir.mkdir(exist_ok=True)
                temp_path = temp_dir / uploaded_file.name
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process image with RAG system (OCR + BLIP, stores in vector DB)
                result = rag_system.ingest_file(temp_path)
                
                # Clean up temp file
                temp_path.unlink()
                
                if result.success:
                    # Store file info for tracking
                    file_info = {
                        "name": uploaded_file.name,
                        "size": uploaded_file.size,
                        "type": uploaded_file.type,
                        "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "chunks": len(result.chunks),
                        "processing_time": result.processing_time
                    }
                    st.session_state.uploaded_files.append(file_info)
                    save_uploaded_files_list()
                    
                    return True, f"✅ Successfully processed image {uploaded_file.name} ({len(result.chunks)} chunks created)"
                else:
                    return False, f"❌ Failed to process image {uploaded_file.name}: {result.error_message}"
                
            except Exception as e:
                return False, f"❌ Failed to process image {uploaded_file.name}: {str(e)}"
        
        # For documents and audio, use standard temp file processing
        else:
            # Create temp directory
            temp_dir = Path("temp_uploads")
            temp_dir.mkdir(exist_ok=True)
            
            # Save uploaded file temporarily
            temp_path = temp_dir / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process with RAG system
            result = rag_system.ingest_file(temp_path)
            
            # Clean up temp file
            temp_path.unlink()
            
            if result.success:
                # Add to uploaded files list
                file_info = {
                    "name": uploaded_file.name,
                    "size": uploaded_file.size,
                    "type": uploaded_file.type,
                    "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "chunks": len(result.chunks),
                    "processing_time": result.processing_time
                }
                st.session_state.uploaded_files.append(file_info)
                save_uploaded_files_list()
                return True, f"✅ Successfully processed {uploaded_file.name} ({len(result.chunks)} chunks created)"
            else:
                return False, f"❌ Failed to process {uploaded_file.name}: {result.error_message}"
            
    except Exception as e:
        return False, f"❌ Error processing {uploaded_file.name}: {str(e)}"

def delete_uploaded_file(file_index):
    """Delete an uploaded file from the list."""
    try:
        if 0 <= file_index < len(st.session_state.uploaded_files):
            deleted_file = st.session_state.uploaded_files.pop(file_index)
            save_uploaded_files_list()
            st.success(f"✅ Deleted {deleted_file['name']}")
        else:
            st.error("❌ Invalid file index")
    except Exception as e:
        st.error(f"❌ Error deleting file: {str(e)}")
        
def clear_recent_uploads():
    """Clear the recent uploads list only."""
    try:
        st.session_state.uploaded_files = []
        save_uploaded_files_list()
        # Add a flag to hide recent uploads section
        st.session_state.hide_recent_uploads = True
        st.success("✅ Recent uploads list cleared!")
    except Exception as e:
        st.error(f"❌ Error clearing recent uploads: {str(e)}")

def clear_database_files():
    """Clear all files from the database."""
    try:
        conn = sqlite3.connect('file_storage.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM stored_files')
        conn.commit()
        conn.close()
        st.session_state.uploaded_files = []
        save_uploaded_files_list()
        st.session_state.hide_recent_uploads = True
        st.success("✅ All stored files cleared from database!")
    except Exception as e:
        st.error(f"❌ Error clearing database files: {str(e)}")

def show_recent_uploads():
    """Show recent uploads section again."""
    st.session_state.hide_recent_uploads = False
    st.success("✅ Recent uploads section restored!")

def clear_all_data():
    """Clear all uploaded files and chat history."""
    try:
        st.session_state.uploaded_files = []
        st.session_state.messages = []
        save_uploaded_files_list()
        st.success("✅ All data cleared!")
    except Exception as e:
        st.error(f"❌ Error clearing data: {str(e)}")

def sidebar_content():
    """Render sidebar content."""
    from datetime import datetime  # Local import to avoid scope issues
    
    with st.sidebar:
        # Always visible header - even when collapsed
        st.title(" SmartRAG Multimodal ChatBot")
        st.markdown("**Talk to Your Files, Get Instant Answers**")
        st.markdown("---")
        
        # System Status
        st.subheader("📊 System Status")
        if st.session_state.system_initialized:
            st.success("🟢 System Ready")
            
            # Show current time for reference
            current_time = datetime.now().strftime("%H:%M")
            st.markdown(f"🕒 **Current Time**: {current_time}")
            
            # Audio capabilities status
            if WHISPER_AVAILABLE:
                st.info("🎙️ Speech-to-Text Available")
            else:
                st.warning("🎙️ Install openai-whisper for speech-to-text")
            
            # Get system stats
            try:
                if st.session_state.rag_system:
                    stats = st.session_state.rag_system.get_system_stats()
                    st.metric("Documents", len(st.session_state.uploaded_files))
                    st.metric("LLM Available", "✅ Yes" if stats.get('llm_available', False) else "❌ No")
            except:
                pass
        else:
            st.error("🔴 System Not Ready")
            if st.button("🔄 Retry Initialization"):
                st.session_state.rag_system = None
                st.rerun()
        
        st.markdown("---")
        
        # Recent Uploads with Timestamps
        if not st.session_state.get('hide_recent_uploads', False):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader("🕒 Recent Uploads")
            with col2:
                if st.button("🗑️", key="clear_uploads_btn", help="Clear recent uploads list"):
                    clear_recent_uploads()
                    st.rerun()
            
            try:
                recent_files = get_all_stored_files()[:5]  # Show last 5 uploads
                if recent_files:
                    for filename, file_type, file_size, upload_time in recent_files:
                        from datetime import datetime
                        upload_dt = datetime.strptime(upload_time, '%Y-%m-%d %H:%M:%S')
                        time_str = upload_dt.strftime('%H:%M')
                        
                        # Truncate long filenames
                        display_name = filename if len(filename) <= 20 else f"{filename[:17]}..."
                        st.markdown(f"📄 **{display_name}**")
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;🕒 {time_str} | 📊 {file_size//1024}KB")
                else:
                    st.markdown("*No uploads yet*")
            except Exception as e:
                st.markdown("*Could not load recent uploads*")
        else:
            # Show a restore button when hidden
            st.subheader("🕒 Recent Uploads")
            st.markdown("*Recent uploads hidden*")
            if st.button("👁️ Show Recent Uploads", key="show_uploads_btn", help="Show recent uploads again"):
                show_recent_uploads()
                st.rerun()
        
        st.markdown("---")
        
        # File Upload Section - Clean and Simple
        with st.expander("📁 Upload Documents", expanded=True):
            uploaded_files = st.file_uploader(
                "Choose files to upload",
                accept_multiple_files=True,
                type=['pdf', 'docx', 'doc', 'txt', 'md', 'rtf', 'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp', 'mp3', 'wav', 'm4a', 'ogg', 'flac', 'aac'],
                help="📄 Documents: PDF, DOCX, DOC, TXT, MD, RTF | 🖼️ Images: JPG, PNG, BMP, TIFF, WEBP | 🎵 Audio: MP3, WAV, M4A, OGG, FLAC, AAC",
                key="main_file_uploader"
            )
            
            if uploaded_files and st.session_state.system_initialized:
                if st.button("📤 Process Files", type="primary", key="process_files_btn"):
                    from datetime import datetime  # Import for timestamp handling
                    rag_system = get_rag_system()
                    if rag_system:
                        # Create a processing container
                        processing_container = st.container()
                        with processing_container:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            success_count = 0
                            for i, uploaded_file in enumerate(uploaded_files):
                                status_text.text(f"Processing {uploaded_file.name}...")
                                
                                # Store file in database first
                                uploaded_file.seek(0)
                                file_content = uploaded_file.read()
                                import hashlib
                                file_hash = hashlib.md5(file_content).hexdigest()
                                
                                # Store in database
                                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                stored = store_file_in_db(
                                    filename=uploaded_file.name,
                                    file_content=file_content,
                                    file_type=uploaded_file.type,
                                    file_size=uploaded_file.size,
                                    file_hash=file_hash,
                                    upload_time=current_time
                                )
                                
                                if stored:
                                    # Reset file pointer and process with RAG system
                                    uploaded_file.seek(0)
                                    success, message = process_uploaded_file(uploaded_file, rag_system)
                                    
                                    if success:
                                        success_count += 1
                                        # Add to uploaded files list for UI
                                        file_info = {
                                            "name": uploaded_file.name,
                                            "size": uploaded_file.size,
                                            "type": uploaded_file.type,
                                            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        }
                                        if file_info not in st.session_state.uploaded_files:
                                            st.session_state.uploaded_files.append(file_info)
                                else:
                                    st.error(f"Failed to store {uploaded_file.name} in database")
                                
                                progress_bar.progress((i + 1) / len(uploaded_files))
                            
                            # Final status
                            if success_count == len(uploaded_files):
                                status_text.success(f"✅ Successfully processed all {len(uploaded_files)} files!")
                            else:
                                status_text.warning(f"⚠️ Processed {success_count} of {len(uploaded_files)} files")
                            
                            time.sleep(2)
                            # Clear processing UI elements
                            processing_container.empty()
                            st.rerun()
        
        st.markdown("---")
        
        # Uploaded Files Management - Show both session files and database files
        try:
            # Get files from database
            db_files = get_all_stored_files()
            session_files = st.session_state.uploaded_files
            
            # Combine and deduplicate files
            all_files = []
            file_names_seen = set()
            
            # Add session files first
            for file_info in session_files:
                if file_info['name'] not in file_names_seen:
                    all_files.append({
                        'name': file_info['name'],
                        'size': file_info['size'],
                        'type': file_info.get('type', 'unknown'),
                        'chunks': file_info.get('chunks', 'Unknown'),
                        'upload_time': file_info.get('upload_time', 'Unknown'),
                        'source': 'session'
                    })
                    file_names_seen.add(file_info['name'])
            
            # Add database files that aren't already in session
            for filename, file_type, file_size, upload_time in db_files:
                if filename not in file_names_seen:
                    all_files.append({
                        'name': filename,
                        'size': file_size,
                        'type': file_type,
                        'chunks': 'Unknown',
                        'upload_time': upload_time,
                        'source': 'database'
                    })
                    file_names_seen.add(filename)
            
            if all_files:
                with st.expander(f"📋 All Stored Files ({len(all_files)})", expanded=False):
                    for i, file_info in enumerate(all_files):
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.write(f"📄 **{file_info['name']}**")
                            size_display = f"{file_info['size']:,} bytes" if isinstance(file_info['size'], int) else f"{file_info['size']//1024}KB"
                            st.caption(f"Size: {size_display} | Chunks: {file_info.get('chunks', 'Unknown')}")
                        
                        with col2:
                            if st.button("👁️", key=f"view_all_{i}", help=f"View {file_info['name']}"):
                                st.session_state.viewing_file = file_info['name']
                                st.session_state.show_file_viewer = True
                                st.rerun()
                        
                        with col3:
                            if st.button("🗑️", key=f"delete_all_{i}", help=f"Delete {file_info['name']}"):
                                # Remove from both session and database
                                if file_info['source'] == 'session':
                                    # Find and remove from session
                                    for j, session_file in enumerate(st.session_state.uploaded_files):
                                        if session_file['name'] == file_info['name']:
                                            delete_uploaded_file(j)
                                            break
                                else:
                                    # Remove from database
                                    try:
                                        conn = sqlite3.connect('file_storage.db')
                                        cursor = conn.cursor()
                                        cursor.execute('DELETE FROM stored_files WHERE filename = ?', (file_info['name'],))
                                        conn.commit()
                                        conn.close()
                                        st.success(f"✅ Deleted {file_info['name']} from database")
                                    except Exception as e:
                                        st.error(f"❌ Error deleting from database: {str(e)}")
                                st.rerun()
        except Exception as e:
            # Fallback to original implementation
            if st.session_state.uploaded_files:
                with st.expander(f"📋 Uploaded Files ({len(st.session_state.uploaded_files)})", expanded=False):
                    for i, file_info in enumerate(st.session_state.uploaded_files):
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.write(f"📄 **{file_info['name']}**")
                            st.caption(f"Size: {file_info['size']:,} bytes | Chunks: {file_info.get('chunks', 'Unknown')}")
                        
                        with col2:
                            if st.button("👁️", key=f"view_{i}", help=f"View {file_info['name']}"):
                                st.session_state.viewing_file = file_info['name']
                                st.session_state.show_file_viewer = True
                                st.rerun()
                        
                        with col3:
                            if st.button("🗑️", key=f"delete_{i}", help=f"Delete {file_info['name']}"):
                                delete_uploaded_file(i)
                                st.rerun()
        
        st.markdown("---")
        
        # Clear Chat Button
        st.markdown("### 🧹 Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear Chat", help="Clear chat history only", use_container_width=True):
                st.session_state.messages = []
                st.success("✅ Chat history cleared!")
                st.rerun()
        
        with col2:
            if st.button("Hide Uploads", help="Hide recent uploads section", use_container_width=True):
                clear_recent_uploads()
                st.rerun()
        
        # Second row for more destructive actions
        col3, col4 = st.columns(2)
        
        with col3:
            if st.button("Clear DB Files", help="Clear all files from database", type="secondary", use_container_width=True):
                clear_database_files()
                st.rerun()
        
        with col4:
            if st.button("Clear All Data", help="Clear all files and chat history", type="secondary", use_container_width=True):
                clear_all_data()
                st.rerun()

def main_chat_interface():
    """Render the main chat interface."""
    # Show file viewer if active
    if st.session_state.show_file_viewer and st.session_state.viewing_file:
        render_file_viewer()
        return
    
    # Top header with menu button as backup for sidebar access
    col1, col2, col3 = st.columns([1, 6, 1])
    
    with col1:
        # Enhanced backup menu button with full sidebar functionality
        if st.button("🔧", help="Access sidebar tools if sidebar is not visible", key="sidebar_tools_toggle"):
            st.session_state.show_emergency_sidebar = not st.session_state.get('show_emergency_sidebar', False)
    
    with col2:
        st.title("Chat with your Documents")
        st.markdown("*“Smart Conversations with Text, Images & Audio”*")
    
    with col3:
        # Quick status indicator
        if st.session_state.system_initialized:
            st.markdown("🟢 **Ready**")
        else:
            st.markdown("🔴 **Loading**")
    
    # Emergency sidebar panel if regular sidebar is not working
    if st.session_state.get('show_emergency_sidebar', False):
        with st.container():
            st.markdown("### 🔧 Emergency Sidebar Access")
            
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                # File upload
                st.markdown("**📁 Upload Files:**")
                emergency_upload = st.file_uploader(
                    "Choose files to upload",
                    type=['pdf', 'docx', 'txt', 'png', 'jpg', 'jpeg', 'mp3', 'wav', 'mp4'],
                    accept_multiple_files=True,
                    key="emergency_upload"
                )
                
                if emergency_upload:
                    st.success(f"Ready to process {len(emergency_upload)} file(s)")
                    if st.button("Process Uploaded Files"):
                        # Process files here (you can add the actual processing logic)
                        for file in emergency_upload:
                            st.info(f"Processing: {file.name}")
                
                # Quick actions
                st.markdown("**🧹 Quick Actions:**")
                col_clear1, col_clear2, col_close = st.columns(3)
                
                with col_clear1:
                    if st.button("Clear Chat", key="emergency_clear_chat"):
                        st.session_state.messages = []
                        st.success("✅ Chat cleared!")
                        st.rerun()
                
                with col_clear2:
                    if st.button("Clear All", key="emergency_clear_all"):
                        st.session_state.messages = []
                        st.success("✅ All data cleared!")
                        st.rerun()
                
                with col_close:
                    if st.button("Close Panel", key="emergency_close"):
                        st.session_state.show_emergency_sidebar = False
                        st.rerun()
            
            st.markdown("---")  # Separator
    
    # Chat container with scrollable messages
    chat_container = st.container()
    with chat_container:
        # Add some sample messages if chat is empty to show styling
        if not st.session_state.messages:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            st.markdown("""
            <div class="assistant-message">
                <strong>SmartRAG:</strong><br>
                Welcome! I'm ready to help you with your documents. Upload some files using the sidebar and start asking questions!
                <br><br>
                <small><em>System ready | No documents uploaded yet</em></small>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Quick upload option if no files are uploaded (backup if sidebar not accessible)
            if not st.session_state.uploaded_files and st.session_state.system_initialized:
                st.info("💡 **Quick Start:** If you can't see the sidebar, you can upload a file here:")
                quick_file = st.file_uploader(
                    "Upload a document, image, or audio file",
                    type=['pdf', 'docx', 'txt', 'jpg', 'png', 'mp3', 'wav'],
                    key="quick_upload"
                )
                if quick_file:
                    rag_system = get_rag_system()
                    if rag_system:
                        with st.spinner("Processing file..."):
                            success, message = process_uploaded_file(quick_file, rag_system)
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)
        else:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for message in st.session_state.messages:
                if message["role"] == "user":
                    # User message in green bubble - clean display
                    st.markdown(f"""
                    <div class="user-message">
                        <strong>You:</strong><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Assistant message in blue bubble with clickable sources
                    content = message["content"]
                    metadata = message.get("metadata", "")
                    
                    # Make source document names clickable
                    import re
                    def make_sources_clickable(text):
                        # Pattern to find "Source: filename" or "from filename"
                        pattern = r'(Source:|from)\s+([^\s,;.!?]+\.(pdf|docx?|txt|png|jpe?g|gif|bmp|tiff?|wav|mp3|m4a|ogg|flac))'
                        
                        def replace_with_button(match):
                            prefix = match.group(1)
                            filename = match.group(2)
                            return f'{prefix} <span style="color: #4a90e2; cursor: pointer; text-decoration: underline;" onclick="window.parent.postMessage({{type: \'file_click\', filename: \'{filename}\'}})">📄 {filename}</span>'
                        
                        return re.sub(pattern, replace_with_button, text, flags=re.IGNORECASE)
                    
                    clickable_content = make_sources_clickable(content)
                    clickable_metadata = make_sources_clickable(metadata)
                    
                    st.markdown(f"""
                    <div class="assistant-message">
                        <strong>SmartRAG:</strong><br>
                        {clickable_content}
                        <br><br>
                        <small><em>{clickable_metadata}</em></small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add JavaScript to handle file clicks
                    st.markdown("""
                    <script>
                    window.addEventListener('message', function(event) {
                        if (event.data.type === 'file_click') {
                            // This would ideally trigger a Streamlit callback
                            console.log('File clicked:', event.data.filename);
                        }
                    });
                    </script>
                    """, unsafe_allow_html=True)
                    
                    # Alternative: Add clickable buttons for each source mentioned
                    sources = re.findall(r'(Source:|from)\s+([^\s,;.!?]+\.(pdf|docx?|txt|png|jpe?g|gif|bmp|tiff?|wav|mp3|m4a|ogg|flac))', content + " " + metadata, re.IGNORECASE)
                    if sources:
                        st.markdown("**📎 View Sources:**")
                        cols = st.columns(min(len(sources), 3))
                        for i, (_, filename, _) in enumerate(sources):
                            with cols[i % 3]:
                                if st.button(f"📄 {filename}", key=f"view_{filename}_{len(st.session_state.messages)}_{i}"):
                                    st.session_state.viewing_file = filename
                                    st.session_state.show_file_viewer = True
                                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Show thinking indicator above chat input when processing
    thinking_placeholder = st.empty()
    
    # Custom chat input with mic button
    if st.session_state.system_initialized:
        # Use custom chat input without extra headings
        user_input = render_custom_chat_input()
        
        # Create a hash of the input to prevent processing the same input multiple times
        import hashlib
        if user_input:
            input_hash = hashlib.md5(f"{user_input}_{len(st.session_state.messages)}".encode()).hexdigest()
            
            # Only process if this is a new, unique input
            if input_hash != st.session_state.input_hash:
                st.session_state.input_hash = input_hash
                
                # Add user message to chat history
                st.session_state.messages.append({
                    "role": "user", 
                    "content": user_input
                })
                
                # Clear any stored transcribed text to prevent loops
                if 'transcribed_text' in st.session_state:
                    del st.session_state['transcribed_text']
                
                # Show thinking indicator above chat input
                with thinking_placeholder.container():
                    with st.spinner("🤔 Thinking..."):
                        # Check if this is a time-based query first
                        import re
                        time_query_patterns = [
                            r'what.*uploaded.*at\s*(\d{1,2}[:.]\d{2})',
                            r'what.*uploaded.*(\d{1,2}[:.]\d{2})',
                            r'documents.*uploaded.*(\d{1,2}[:.]\d{2})',
                            r'files.*uploaded.*(\d{1,2}[:.]\d{2})',
                            r'uploaded.*at\s*(\d{1,2}[:.]\d{2})',
                        ]
                        
                        is_time_query = False
                        timestamp_response = None
                        
                        for pattern in time_query_patterns:
                            match = re.search(pattern, user_input.lower())
                            if match:
                                time_str = match.group(1)
                                timestamp_response = search_files_by_timestamp(time_str)
                                is_time_query = True
                                break
                        
                        if is_time_query and timestamp_response:
                            # Handle timestamp query directly - don't go through RAG
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": timestamp_response,
                                "metadata": "Search based on upload timestamp"
                            })
                        else:
                            # Regular RAG query
                            rag_system = get_rag_system()
                            if rag_system:
                                try:
                                    query_request = QueryRequest(
                                        query=user_input,
                                        top_k=5,
                                        include_metadata=True
                                    )
                                    
                                    start_time = time.time()
                                    response = rag_system.query(query_request)
                                    processing_time = time.time() - start_time
                                    
                                    # Format metadata with specific document names
                                    if response.sources:
                                        # Extract unique document names from sources
                                        doc_names = set()
                                        for source in response.sources:
                                            source_file = source.source_file or source.metadata.get('filename', 'Unknown')
                                            if source_file and source_file != 'Unknown':
                                                # Get just the filename without path
                                                doc_name = source_file.split('/')[-1].split('\\')[-1]
                                                doc_names.add(doc_name)
                                        
                                        if doc_names:
                                            doc_list = ', '.join(sorted(doc_names))
                                            metadata = f"Sources: {doc_list}"
                                        else:
                                            metadata = f"Sources: {len(response.sources)} documents"
                                    else:
                                        metadata = "Sources: No documents referenced"
                                    
                                    # Add assistant response to chat history
                                    st.session_state.messages.append({
                                        "role": "assistant", 
                                        "content": response.answer,
                                        "metadata": metadata
                                    })
                                    
                                except Exception as e:
                                    st.session_state.messages.append({
                                        "role": "assistant", 
                                        "content": f"Sorry, I encountered an error: {str(e)}",
                                        "metadata": ""
                                    })
                
                # Clear the thinking indicator and rerun
                thinking_placeholder.empty()
                st.rerun()
    else:
        st.warning("🚫 System not initialized. Please check the sidebar for initialization status.")
        st.info("💡 Make sure you have sufficient RAM and the LLama model is properly configured")

def main():
    """Main application function."""
    init_session_state()
    load_uploaded_files_list()
    
    # Initialize RAG system on first load
    if not st.session_state.system_initialized and st.session_state.rag_system is None:
        get_rag_system()
    
    # Render UI
    sidebar_content()
    main_chat_interface()

if __name__ == "__main__":
    main()