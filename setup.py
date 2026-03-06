from setuptools import setup, find_packages

setup(
    name="villa_cog_assist",
    version="0.1.0",
    description="RealSense cognitive state monitoring pipeline",
    packages=find_packages(),
    install_requires=[
        "pyrealsense2",
        "numpy",
        "opencv-python",
        "mediapipe",
        "ultralytics",
    ],
    entry_points={
        "console_scripts": [
            "villa-cog-assist=demo:main",
        ],
    },
)
