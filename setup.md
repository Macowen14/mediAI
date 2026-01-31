# Project setup

## Prerequisites

- Python 3.11 or newer
  Create and activate a virtual environment (Linux)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Important note about the package manager

- This project uses `uv` to install and manage project libraries (not `pip`). When working on this repository prefer `uv` for dependency operations; if you must use `pip` please coordinate with the project maintainer.
  Upgrade pip and install the project

```bash
python -m pip install --upgrade pip
pip install -e .
```

Notes

- The repository includes a `pyproject.toml` and `setup.py`; `pip install -e .` will install the project and the declared dependencies into the active venv.
  Optional: register an IPython kernel for notebooks

```bash
python -m ipykernel install --user --name mediai --display-name "Python (mediai)"
```

Running the app (example)
``bash

# If the project exposes a FastAPI app in app.py as `app`:

uvicorn app:app --reload
``

If you run into dependency build issues

- Ensure you have a recent `pip` and a working build toolchain. Installing a wheel-builder like `pip install build` may help.

# Project setup

## Clone the repository (first step)

```bash
git clone https://github.com/Macowen14/mediAI.git
cd mediAI
```

## Prerequisites

- Python 3.11 or newer
- Git
- An OpenAI API key (required by services used in this project)
- An ollama model (preferably `ministral-3:8b` or a cloud based model `deepseek-v3.1:671b-cloud`)

Create and activate a virtual environment (Linux)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Important note about the package manager

- This project uses `uv` to install and manage project libraries (not `pip`). When working on this repository prefer `uv` for dependency operations; if you must use `pip` please coordinate with the project maintainer.

Upgrade pip and install the project (optional when using `uv`)

```bash
python -m pip install --upgrade pip
pip install -e .
```

Notes

- The repository includes a `pyproject.toml` and `setup.py`; `pip install -e .` will install the project and the declared dependencies into the active venv.
- If you prefer a non-editable install: `pip install .`

Optional: register an IPython kernel for notebooks

```bash
python -m ipykernel install --user --name mediai --display-name "Python (mediai)"
```

Running the app (example)

```bash
# If the project exposes a FastAPI app in app.py as `app`:
uvicorn app:app --reload
```

If you run into dependency build issues

- Ensure you have a recent `pip` and a working build toolchain. Installing a wheel-builder like `pip install build` may help.

If you want me to add CI, a requirements file, or a dedicated Makefile, tell me and I will add one.
