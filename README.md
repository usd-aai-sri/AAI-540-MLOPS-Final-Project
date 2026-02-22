# AAI-540-MLOPS-Final-Project-Team-3

# Credit Card Fraud Detection using AWS SageMaker

[![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-orange?style=flat-square&logo=amazon-aws)](https://aws.amazon.com/sagemaker/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-green?style=flat-square)](https://xgboost.readthedocs.io/)

## Overview
I built this end-to-end Machine Learning system to detect fraudulent credit card transactions with high recall while strictly controlling false positives. Credit card fraud is a severe, highly imbalanced binary classification problem. To solve this, I designed a production-ready pipeline entirely within AWS SageMaker. 

My system ingests raw data, processes features, trains an optimized XGBoost model, registers the model for MLOps governance, runs asynchronous batch predictions, and actively monitors the model for algorithmic bias using SageMaker Clarify.

## Architecture

My pipeline consists of the following sequential stages:
1. **Data Ingestion & S3 Storage:** Securely uploading raw Kaggle transaction data to an Amazon S3 bucket.
2. **Data Engineering:** Handling missing values, scaling numeric features, and applying stratified splitting to preserve the severe class imbalance (0.17% fraud).
3. **Model Training:** Provisioning transient ml.m5.xlarge instances to train an XGBoost Estimator. I utilized the `scale_pos_weight` hyperparameter to natively and heavily penalize missed fraud cases.
4. **MLOps & Governance:** Registering the finalized model in the SageMaker Model Registry and dynamically generating a Model Card to document training metadata and risk levels.
5. **Batch Inference:** Deploying a SageMaker Batch Transform job to asynchronously score thousands of unlabelled transactions.
6. **Bias Monitoring:** Running a SageMaker Clarify job to analyze algorithmic bias against specific transaction thresholds.

## Repository Structure

```text
├── README.md                           # Project documentation
├── data/                               # Dummy/sample data snippets
├── fraud_detection_pipeline.ipynb      # Main Jupyter Notebook with end-to-end SageMaker blocks
└── requirements.txt                    # Python dependencies
```

## Getting Started

### Prerequisites
To run our pipeline, you will need:
* An active AWS Account with SageMaker Studio or Notebook Instances enabled.
* The proper IAM Execution Role with permissions for SageMaker, S3, and CloudWatch.
* The Credit Card Fraud Detection dataset from Kaggle downloaded and converted to a standard CSV format (e.g., `my_project_dataset.csv`).

### Execution
1. Clone this repository into your SageMaker Studio environment.
2. Upload `fraud_detection_dataset.csv` to the root directory.
3. Open `fraud_detection_pipeline.ipynb` and execute the blocks sequentially. The notebook will automatically spin up the required AWS infrastructure, run the jobs, and spin them down to avoid unnecessary billing.

## Results & Evaluation
Because this dataset is wildly imbalanced (284,807 legitimate vs. 492 fraudulent), standard accuracy is a misleading metric. Instead, we evaluated our model using Precision-Recall AUC (PR-AUC) and the Confusion Matrix.

By tuning the decision threshold and utilizing XGBoost's `scale_pos_weight`, our model successfully isolates fraudulent spikes while minimizing friction for legitimate customers.

## Governance & Bias Check
We take responsible AI seriously. In the final phase of our pipeline, we explicitly spin up a SageMaker Clarify container to generate HTML/JSON reports checking the model for feature bias. Additionally, every successful training run is cataloged in the Model Registry with an attached Model Card detailing its intended use and limitations.
