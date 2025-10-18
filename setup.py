"""
Setup script for SmartRAG multimodal RAG system.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    requirements = requirements_file.read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]
else:
    requirements = [
        "langchain>=0.1.0",
        "chromadb>=0.4.0", 
        "sentence-transformers>=2.2.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "PyPDF2>=3.0.1",
        "python-docx>=1.0.0",
        "Pillow>=10.0.0",
        "requests>=2.31.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0.0",
        "click>=8.1.0"
    ]

setup(
    name="smartrag",
    version="1.0.0",
    author="SmartRAG Team",
    author_email="team@smartrag.com",
    description="Multimodal Retrieval-Augmented Generation system for documents, images, and audio",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/smartrag",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Indexing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "all": [
            "pdfplumber>=0.9.0",
            "pytesseract>=0.3.10",
            "opencv-python>=4.8.0",
            "openai-whisper>=20231117",
            "pydub>=0.25.1",
            "librosa>=0.10.0",
            "faiss-cpu>=1.7.4"
        ],
        "audio": [
            "openai-whisper>=20231117",
            "pydub>=0.25.1", 
            "librosa>=0.10.0",
            "SpeechRecognition>=3.10.0"
        ],
        "image": [
            "pytesseract>=0.3.10",
            "opencv-python>=4.8.0"
        ],
        "pdf": [
            "pdfplumber>=0.9.0"
        ],
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "coverage>=7.0.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "smartrag=cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "smartrag": ["config/*.yaml"],
    },
    keywords=[
        "rag", "retrieval", "generation", "multimodal", "llm", 
        "document-processing", "semantic-search", "ai", "nlp"
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-org/smartrag/issues",
        "Source": "https://github.com/your-org/smartrag",
        "Documentation": "https://github.com/your-org/smartrag/wiki",
    },
)