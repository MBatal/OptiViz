@echo off

REM Setup virtual environment for the project
echo Setting up virtual environment...

REM Create a virtual environment named 'optiviz_venv'
python -m venv optiviz_venv

REM Activate the virtual environment
call optiviz_venv\Scripts\activate.bat

REM Install main project dependencies
pip install -r requirements.txt

REM Install development dependencies (for tools like linters, formatters, etc.)
pip install -r requirements-dev.txt

echo Virtual environment setup complete.
