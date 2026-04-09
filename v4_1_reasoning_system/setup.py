"""
Minimal setup.py for backwards-compatible pip install.
Prefer pyproject.toml for modern tooling.
"""
from setuptools import setup, find_packages

setup(
    name="v4_1_reasoning_system",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "numpy>=1.24.0",
        "ortools>=9.7,<9.12",
        "faiss-cpu>=1.7.4",
        "pydantic>=2.5.0",
        "rich>=13.0.0",
    ],
)
