# Makefile for Adaptive PDS Anomaly Detection Project

# Define the virtual environment directory
VENV = env
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

# Default target: Create venv, install deps, and run the code
all: install run

# Create virtual environment and install dependencies
install: requirements.txt
	test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Environment setup complete."

# Run the experiment using the virtual environment's python
run:
	$(PYTHON) main.py

# Clean up results and the virtual environment
clean:
	rm -f final_project_plot.png final_metrics.json
	rm -rf __pycache__
	rm -rf $(VENV)
	@echo "Cleaned up generated files and virtual environment."