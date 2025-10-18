"""
SmartRAG ChatBot Launcher
Simple script to start the Streamlit chatbot interface.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the SmartRAG Streamlit chatbot."""
    print("🤖 Starting SmartRAG ChatBot...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("chatbot_app.py"):
        print("❌ Error: chatbot_app.py not found!")
        print("Please run this script from the SmartRAG project directory.")
        sys.exit(1)
    
    # Check if config exists
    if not os.path.exists("config.yaml"):
        print("❌ Error: config.yaml not found!")
        print("Please ensure the configuration file exists.")
        sys.exit(1)
    
    print("✅ Configuration found")
    print("🚀 Launching Streamlit interface...")
    print("\n📝 Instructions:")
    print("1. The app will open in your default browser")
    print("2. Upload documents using the sidebar")
    print("3. Start chatting with your documents!")
    print("4. Press Ctrl+C to stop the server")
    print("\n" + "=" * 50)
    
    try:
        # Launch streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "chatbot_app.py",
            "--server.address", "localhost",
            "--server.port", "8501",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n\n👋 ChatBot stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error starting Streamlit: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Make sure Streamlit is installed: pip install streamlit")
        print("2. Ensure Ollama is running with required models")
        print("3. Check that all dependencies are installed")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()