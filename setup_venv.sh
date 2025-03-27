#!/bin/bash

# Setup virtual environment for the project
echo "Setting up virtual environment..."

# Create a virtual environment named 'optiviz_venv'
python3 -m venv optiviz_venv

# Activate the virtual environment
source optiviz_venv/bin/activate

# Install main project dependencies
pip install -r requirements.txt

# Install development dependencies (for tools like linters, formatters, etc.)
pip install -r requirements-dev.txt

echo "Virtual environment setup complete."
