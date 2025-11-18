#!/usr/bin/env python3
"""Setup script for pdf2epub package."""

from setuptools import setup, find_packages

setup(
    name="pdf2epub",
    version="2.0.0",
    description="A robust Python tool for converting scanned PDF books into EPUB ebooks using OCR",
    author="jguyot",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pillow>=10.1.0",
        "opencv-python>=4.8.1",
        "pytesseract>=0.3.10",
        "easyocr>=1.7.1",
        "pdf2image>=1.16.3",
        "ebooklib>=0.18",
        "colorlog>=6.8.0",
        "imutils>=0.5.4",
        "pyyaml>=6.0.1",
        "google-generativeai>=0.3.2",
        "openai>=1.6.1",
        "anthropic>=0.8.1",
        "tenacity>=8.2.3",
        "tqdm>=4.66.1",
        "pydantic>=2.5.3",
        "pydantic-settings>=2.1.0",
        "pandas>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "pdf2epub=pdf2epub.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
