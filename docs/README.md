# Machine Learning Model to Predict Turbine Energy Yield (TEY)

This prototype predicts Turbine Energy Yield (TEY) using a machine learning model. It consists of an interactive Streamlit application that makes predictions using an API, which serves the best model obtained from experiments in MLflow.


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Trobleshooting](#troubleshooting)
## Installation

### Clone the Repository 

    git clone https://github.com/cristianBMJ/Predictive_Maintenance_ML.git
    cd <repository_directory>

### Install Poetry

If you haven't installed Poetry yet, you can do so by running:

    curl -sSL https://install.python-poetry.org | python3 -

Navigate to the project directory where the `pyproject.toml` file is located and run:

    poetry install

Ensure that you are using a compatible Python version `>=3.9 and <4.0`. You can check your Python version with:

       python --version

## Usage 

### Run MLflow

To start the MLflow UI to make experiments, run the following command:

    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

Running on http://127.0.0.1:5000

Training a new experiment:

    python src/model_training

### Run API app

    python api/app.py 

Running on http://127.0.0.1:5001


### Run Streamlit app


After installing the dependencies, make mlflow experiment and save best model in Flask API, you can start the Streamlit application. Use the following command:

       streamlit run streamlit_app/streamlit_app.py

Open your web browser and go to http://localhost:8501 to access the Streamlit application.


## Troubleshooting

### - Streamlit/python Version Mismatch `3.9.7`

*Solution:*

- Install a compatible version `3.9.8` using pyenv
- Reinstall all dependecies with Poetry
- Add Streamlit with Poetry