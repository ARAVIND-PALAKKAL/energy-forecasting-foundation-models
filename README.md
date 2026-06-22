# Project 1: Time Series Foundation Model Benchmark for Energy Load Forecasting

## Overview
A benchmarking study comparing pretrained time series foundation models 
(zero-shot) against a trained LSTM and a classical ARIMA baseline for 
energy load forecasting. Foundation models are evaluated purely zero-shot — 
no fine-tuning or training on this dataset.

## Results

| Model | MAPE | RMSE | Training Required |
|-------|------|------|-------------------|
| ARIMA (classical baseline) | 15.67% | 2844.99 MW | No |
| TimesFM 2.5 (Google) | 6.27% | 1213.88 MW | No (zero-shot) |
| Chronos-T5-Small (Amazon) | 4.91% | 1035.98 MW | No (zero-shot) |
| **LSTM (trained)** | **3.45%** | **696.64 MW** | Yes |

**Key findings:**
- Both foundation models beat the classical ARIMA baseline zero-shot, 
  without seeing any of this dataset during inference
- Chronos outperforms TimesFM on this dataset by 1.36% MAPE
- The trained LSTM still achieves the best overall accuracy, but foundation 
  models offer a compelling alternative when labelled data is scarce or 
  training time is limited
- Foundation models reduce MAPE by 60-69% over ARIMA with no training at all

## Dataset
- **Source:** [PJM Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) (Kaggle)
- **Region:** AEP (American Electric Power)
- **Period:** 2016–2018 (subset used for modelling)
- **Granularity:** Daily average (resampled from hourly)
- **Test set:** Last 90 days held out

## Models

### ARIMA (5,1,0)
Classical statistical baseline. Fitted on training data, 
forecasts 90 steps ahead.

### Chronos (Amazon)
Pretrained language-model-style foundation model for time series. 
Uses `amazon/chronos-t5-small` checkpoint. Zero-shot inference — 
training data used as context only.

### TimesFM 2.5 (Google)
Decoder-only transformer foundation model pretrained on 100B+ 
real-world time series points. Uses `google/timesfm-2.5-200m-pytorch` 
checkpoint. Zero-shot inference.

### LSTM (Trained)
2-layer LSTM with hidden size 64, trained for 50 epochs on the 
training set with sequence length of 30 days.

## Metrics
- **MAPE** (Mean Absolute Percentage Error) — primary metric
- **RMSE** (Root Mean Squared Error) — secondary metric

## Project Structure
energy-forecasting-foundation-models/

├── data/               # Raw CSV files (not tracked in Git)

├── notebooks/

│   └── project1_foundation_models.ipynb

└── README.md
## Requirements
pandas

numpy

matplotlib

torch

statsmodels

chronos-forecasting

timesfm

jupyter
## How to Run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Install Chronos: `pip install git+https://github.com/amazon-science/chronos-forecasting.git`
4. Download AEP_hourly.csv from Kaggle and place in `data/`
5. Open and run `notebooks/project1_foundation_models.ipynb`

## Context
Part of a series of energy forecasting projects. This project focuses 
on benchmarking state-of-the-art pretrained foundation models against 
trained deep learning and classical statistical approaches, exploring 
the trade-off between zero-shot convenience and task-specific accuracy.

## Related Projects
- [Project 2: Multivariate LSTM with Weather Covariates](https://github.com/ARAVIND-PALAKKAL/energy-load-forecasting-multivariate)
- [Project 3: Classical vs Deep Learning Benchmark](https://github.com/ARAVIND-PALAKKAL/energy-forecasting-projects)