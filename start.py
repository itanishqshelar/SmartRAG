#!/usr/bin/env python3
"""
SmartRAG - Application Launcher
Validates environment and starts the SmartRAG web interface
"""

import subprocess
import sys
import os
from pathlib import Path

def check_environment():
    """Pre-flight checks before starting."""
    errors = []
    
    # Check if we're in the right directory
    if not Path("chatbot_app.py").exists():
        errors.append("❌ chatbot_app.py not found. Run this script from the project root directory.")
    
    # Check if config exists
    if not Path("config.yaml").exists():
        errors.append("❌ config.yaml not found. Configuration file is required.")
    
    # Check if requirements are installed
    try:
        import streamlit
    except ImportError:
        errors.append("❌ Streamlit not installed. Run: pip install -r requirements.txt")
    
    return errors

def main():
    """Start the SmartRAG web application."""
    print("🚀 Starting SmartRAG Multimodal ChatBot...")
    print("=" * 60)
    
    # Run pre-flight checks
    errors = check_environment()
    if errors:
        print("\n⚠️  Pre-flight checks failed:\n")
        for error in errors:
            print(f"   {error}")
        print("\n� Fix the issues above and try again.")
        sys.exit(1)
    
    print("✅ Environment validated")
    print(f"�📁 Working directory: {os.getcwd()}")
    print("🌐 Web interface: http://localhost:8501")
    print("🤖 Model: Llama 3.1 8B via Ollama (local)")
    print("🔒 Privacy: Completely OFFLINE")
    print("")
    print("� Instructions:")
    print("   1. The app will open in your default browser")
    print("   2. Upload documents using the sidebar")
    print("   3. Start chatting with your documents!")
    print("   4. Press Ctrl+C to stop the server")
    print("")
    print("=" * 60)
    print("")
    
    try:
        # Start the Streamlit chatbot app
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "chatbot_app.py", 
             "--server.port=8501",
             "--server.address=localhost"],
            check=True
        )
    except KeyboardInterrupt:
        print("\n\n🛑 SmartRAG stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error starting application: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Check if Ollama is running: ollama list")
        print("   2. Verify Python packages: pip install -r requirements.txt")
        print("   3. Check logs in ./logs/ for detailed errors")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\n🔧 Make sure you have:")
        print("   1. Installed requirements: pip install -r requirements.txt")
        print("   2. Ollama running with required models")
        sys.exit(1)

if __name__ == "__main__":
    main()