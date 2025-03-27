#!/bin/bash

# Check if virtual environment exists
if [ -d "optiviz_venv" ]; then
  # Activate the virtual environment
  source optiviz_venv/bin/activate
  echo "Virtual environment 'optiviz_venv' activated."
else
  echo "Virtual environment 'optiviz_venv' not found. Please create it first by running setup_venv.sh."
fi
