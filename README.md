# Adv_ML_Group4_Project_Code

Code for project in ECE 8936

# Setup

## Prerequisites

- Python (recommended: Python 3.8 or newer)

## Setup Steps

1. Install [`uv`](https://astral.sh/uv/install/):
	```bash
	curl -Ls https://astral.sh/uv/install.sh | bash
	```

    Make sure to follow all instructions and setup properly

3. Sync dependencies:
	```bash
	uv sync
	```

This will install all required packages as specified in `pyproject.toml`.

## Running Notebooks and Python Files (with uv)

### Run Jupyter Notebooks

1. Install Jupyter using uv:
	```bash
	uv pip install jupyter
	```

2. Start Jupyter:
	```bash
	uv pip run jupyter notebook
	```
	Open the desired notebook from the `notebooks/` directory in your browser.

### Run Python Files

To run a Python script from the `src/` directory using uv:

```bash
uv pip run python src/<filename>.py
```

Replace `<filename>` with the name of the Python file you want to execute.


**Note**: We used VSCode Jupyter Extension to run this, so the setup is simpler. Contact me at harnoors@mun.ca if help is needed with setup.