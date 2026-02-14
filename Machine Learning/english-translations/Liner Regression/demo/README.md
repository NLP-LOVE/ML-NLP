## Table of Contents
- [1. Problem Statement](#1-problem-statement)
- [2. Steps](#2-steps)
- [3. Model Selection](#3-model-selection)
- [4. Environment Setup](#4-environment-setup)
- [5. CSV Data Processing](#5-csv-data-processing)
- [6. Data Processing](#6-data-processing)
- [7. Model Training](#7-model-training)
- [8. Full Code](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/Liner%20Regression/demo/housing_price.py)

This document describes the Python code for a housing price prediction model. Housing price prediction is a classic introductory ML problem; this demo shows how to build such a model step by step.

## 1. Problem Statement

Given basic housing information and sales data, build a regression model to predict housing sales prices.

Data download: [Download](https://pan.baidu.com/share/init?surl=kVdwI3d), password: mfqy.

**Data description:**
- Data includes housing sales prices and basic information from King County, USA (May 2014–May 2015).
- Data is split into training and test sets (kc_train.csv, kc_test.csv).
- Training data: ~10,000 records, 14 fields. Main fields:
  - Column 1: Sale date
  - Column 2: Sale price (target)
  - Columns 3–14: bedrooms, bathrooms, living area, parking area, floors, grade, sqft_above, sqft_basement, year_built, year_renovated, lat, long
- Test data: ~3,000 records, 13 fields (no sale price). Use the trained model to predict prices.

## 2. Steps

- 1. Choose and evaluate the model.
- 2. Impute missing values (e.g., mean).
- 3. Normalize features (e.g., feature scaling) for comparable dimensions.
- 4. Train the model.
- 5. Evaluate on test data and visualize predictions.

## 3. Model Selection

Use **multiple linear regression**. Formula: y = w₁x₁ + w₂x₂ + … + b. Use sklearn for training.

## 4. Environment Setup

- Python 3.5+
- numpy, pandas, matplotlib, seaborn, sklearn

## 5. CSV Data Processing

Create separate files: one for features (without sale price), one for sale price (target) for evaluation.

## 6. Data Processing

Read data, check for missing values, apply feature scaling (e.g., MinMaxScaler).

## 7. Model Training

Use sklearn's LinearRegression. Evaluate with MSE. Plot predicted vs. actual for visualization.

## [8. Full Code](https://github.com/NLP-LOVE/ML-NLP/blob/master/Machine%20Learning/Liner%20Regression/demo/housing_price.py)

------

> Author: [@mantchs](https://github.com/mantchs)
>
> Welcome to join the discussion! <a target="_blank" href="//shang.qq.com/wpa/qunwpa?idkey=863f915b9178560bd32ca07cd090a7d9e6f5f90fcff5667489697b1621cecdb3"><img border="0" src="http://pub.idqqimg.com/wpa/images/group.png" alt="NLP Interview Study Group" title="NLP Interview Study Group"></a>
