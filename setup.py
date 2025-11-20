from setuptools import setup, find_packages

setup(
    name="copy_move_detection",
    version="0.1.0",
    description="A system for detecting copy-move forgeries in scientific images.",
    author="The Research Integrity project",
    packages=["copy_move_detection", "copy_move_detection.utility"],
    package_dir={"copy_move_detection": "src"},
    package_data={
        "copy_move_detection": ["*.so"],
    },
    install_requires=[
        "numpy==1.23.5",
        "Pillow>=9,<10",
        "scipy==1.10.1",
        "scikit-learn==1.3.2",
        "matplotlib==3.7.5",
        "svgwrite==1.4.3",
        "opencv-python==4.12.0.88",
    ],
    entry_points={
        "console_scripts": [
            "cmfd-detect=copy_move_detection.run_detection:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
)
