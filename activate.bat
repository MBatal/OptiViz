@echo off

REM Check if virtual environment exists
if exist optiviz_venv\Scripts\activate.bat (
    REM Activate the virtual environment
    call optiviz_venv\Scripts\activate.bat
    echo Virtual environment 'optiviz_venv' activated.
) else (
    echo Virtual environment 'optiviz_venv' not found. Please create it first by running setup_venv.bat.
)
