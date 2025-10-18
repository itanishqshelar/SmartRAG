#!/usr/bin/env python3
"""
SmartRAG - Start the web application
Simple launcher script for the SmartRAG web interface
"""

import subprocess
import sys
import os

def main():
    """Start the SmartRAG web application."""
    print("🚀 Starting SmartRAG LLaVA ChatBot...")
    print("=" * 50)
    print("📁 Working directory:", os.getcwd())
    print("🌐 Web interface: http://localhost:8501")
    print("🤖 Using LLaVA 7B for multimodal processing")
    print("🔒 Running completely OFFLINE")
    print("")
    print("💡 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Start the Streamlit chatbot app
        subprocess.run([sys.executable, "-m", "streamlit", "run", "chatbot_app.py", "--server.port=8501"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 SmartRAG stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔧 Make sure you have:")
        print("   1. Installed requirements: pip install -r requirements.txt")
        print("   2. Streamlit installed: pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()