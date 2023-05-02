# Makefile
SHELL = /bin/bash

.PHONY: help
help:
	@echo "Commands:"
	@echo "venv    : creates a virtual environment."
	@echo "style   : executes style formatting."
	@echo "clean   : cleans all unnecessary files."

# Environment
venv:
	python3 -m venv venv
	source venv/bin/activate && \
	python3 -m pip install --upgrade pip setuptools wheel && \
	python3 -m pip install -e ".[dev]"
	@echo -e "\nVirtual environment created -> run 'source venv/bin/activate'"

# Testing
test:
	python3 -m pytest --cov src/madewithml --cov-report html --disable-warnings

# Styling
.PHONY: style
style:
	black .
	flake8
	python3 -m isort .
	pyupgrade

# Cleaning
.PHONY: clean
clean: style
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E ".trash" | xargs rm -rf
	rm -f .coverage*
