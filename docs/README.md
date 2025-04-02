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
    `bash
    git clone https://github.com/cristianBMJ/Predictive_Maintenance_ML.git
    cd <repository_directory>
    `
### Install Poetry
    If you haven't installed Poetry yet, you can do so by running:
    `bash
    curl -sSL https://install.python-poetry.org | python3 -
    `

## Usage 

## Troubleshooting

### - Streamlit/python Version Mismatch (3.9.7)

*Solution:*

- Install a compatible version (3.9.8) using pyenv
- Reinstall all dependecies with Poetry
-  Add Streamlit with Poetry