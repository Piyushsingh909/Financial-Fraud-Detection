Financial Fraud Detection
A project to build and evaluate a machine learning model for detecting financial fraud.

🚀 Getting Started
Follow these steps to set up the project on your local machine.

1. Create and Activate Virtual Environment
It's highly recommended to work within a virtual environment.

On Windows (PowerShell):

py -3 -m venv .venv
./.venv/Scripts/Activate.ps1

On macOS/Linux:

python3 -m venv .venv
source .venv/bin/activate

2. Install Dependencies
Install all the required Python packages from the requirements.txt file.

pip install -r requirements.txt

3. Place Raw Data
Move your raw dataset, named Fraud.csv, into the data/raw/ directory. For example:

# On Windows
move Fraud.csv data/raw/

# On macOS/Linux
mv Fraud.csv data/raw/

📂 Project Workflow
The project is structured into three main command-line scripts.

1. Prepare Data
This script processes the raw Fraud.csv file from data/raw/ and saves a processed version in Parquet format, which is efficient for analysis.

python scripts/prepare_data.py

Output: This command creates the file data/processed/fraud.parquet.

2. Train Baseline Model
This script trains a baseline XGBoost model on the processed data.

python -m src.modeling.train

Outputs: This saves the following files:

Model: models/xgb_baseline.joblib

Reports: reports/metrics.txt, reports/roc_curve.png, and reports/pr_curve.png.

3. Score New Data
Use this script to apply the trained model to a dataset and generate predictions.

python scripts/score.py --input data/processed/fraud.parquet --output reports/scored.parquet

Output: This command creates reports/scored.parquet, which contains the input data along with the model's fraud predictions.

📓 Notebooks
For interactive exploration and model development, you can use the provided Jupyter Notebooks.

notebooks/01_eda.ipynb: Contains a quick exploratory data analysis (EDA) of the dataset.

notebooks/02_modeling.ipynb: Provides an alternative, notebook-based environment to run the model training process.
