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
