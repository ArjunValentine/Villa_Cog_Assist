#!/bin/bash
# Build a standalone executable using PyInstaller

# ensure pyinstaller is installed
pip install pyinstaller

# create a one-file bundle
pyinstaller --onefile demo.py --name villa_cog_assist

echo "Executable generated in dist/"