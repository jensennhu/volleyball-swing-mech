"""
Setup configuration for volleyball-swing-mech package
"""

from setuptools import setup, find_packages
import os

# Read long description from README
def read_file(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Read requirements
def read_requirements(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return [
                line.strip()
                for line in f
                if line.strip() and not line.startswith("#")
            ]
    return []

setup(
    name="volleyball-swing-mech",
    version="1.0.0",
    author="Jensen Hu",
    author_email="your.email@example.com",  # Update with your email
    description="Volleyball spike biomechanics analysis using computer vision and machine learning",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/jensennhu/volleyball-swing-mech",
    project_urls={
        "Bug Reports": "https://github.com/jensennhu/volleyball-swing-mech/issues",
        "Source": "https://github.com/jensennhu/volleyball-swing-mech",
        "Documentation": "https://github.com/jensennhu/volleyball-swing-mech/tree/main/docs",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Multimedia :: Video",
    ],
    keywords="volleyball biomechanics pose-estimation computer-vision machine-learning lstm sports-analytics",
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "mypy>=0.950",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
            "seaborn>=0.11.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
            "seaborn>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "volleyball-analyze=scripts.05_realtime_detection:main",
            "volleyball-train=scripts.04_train_lstm_model:main",
            "volleyball-process=scripts.01_process_cvat_annotations:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
