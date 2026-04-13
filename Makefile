.PHONY: install dev build help

# Variables
VENV_DIR = backend/.venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip

help:
	@echo "Makefile commands:"
	@echo "  make install - Setup Python venv, install Python dependencies, install Node dependencies"
	@echo "  make dev     - Run Vite dev server and FastAPI dev server concurrently"
	@echo "  make build   - Build the React frontend for production"

$(VENV_DIR)/bin/activate:
	python3 -m venv $(VENV_DIR)

install: $(VENV_DIR)/bin/activate
	$(PIP) install --upgrade pip
	$(PIP) install -r backend/requirements.txt
	cd frontend && npm install

dev:
	@echo "Starting backend and frontend concurrently..."
	# Run Uvicorn and Vite. Use Ctrl+C to stop both.
	(cd backend && ../$(VENV_DIR)/bin/uvicorn main:app --reload --port 8000) & PID1=$$!; \
	(cd frontend && npm run dev) & PID2=$$!; \
	wait $$PID1 $$PID2

build:
	cd frontend && npm run build
