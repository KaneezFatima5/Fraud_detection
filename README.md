🌲 Random Forest from Scratch for Fraud Detection 
===================================================
Building an ensemble learning algorithm from the ground up and applying it to real-world financial data

🚀 Overview
---------------
This project focuses on implementing the Random Forest algorithm from scratch using Python and NumPy, followed by applying it to a bank fraud detection dataset.

The goal was to:
-   Understand ensemble learning beyond library abstractions
-   Build a production-ready ML pipeline
-   Evaluate model behavior on highly imbalanced, real-world data

🎯 Objective
--------------------
Detect fraudulent banking transactions by training a custom-built Random Forest classifier and evaluating its performance using industry-standard metrics.

Fraud Detection Dataset
-------------------------
Real-world banking transaction data
Highly imbalanced classes
Includes:
-   Transaction amount
-   Account features
-   Temporal behavior
-   Transaction metadata

🔄 End-to-End Pipeline
Raw Bank Transaction Data
   ↓
Data Cleaning & Preprocessing
   ↓
Custom Decision Trees
   ↓
Random Forest Ensemble
   ↓
Fraud Prediction
   ↓
Evaluation & Analysis

### 1\. Random Forest — Built From Scratch
🔧 Core Components Implemented

Decision Tree classifier
-   Gini impurity & entropy-based splits
-   Feature subsampling per tree
-   Bootstrapped sampling (bagging)
-   Majority voting for final prediction

⚠️ No scikit-learn RandomForest used

### 2\. Model Architecture

-   Number of trees (configurable)
-   Maximum tree depth  
-   Minimum samples per split
-   Random feature selection at each split

All hyperparameters were manually controlled to analyze bias–variance tradeoffs.


### 3\. Data Preprocessing

-   Handling missing values
-   Feature scaling and normalization
-   Train–test split
-   Class imbalance analysis

### 4\. Evaluation Metrics

Due to class imbalance, multiple metrics were used:

-   Precision
-   Recall
-   F1-score
-   Confusion Matrix
-   Accuracy (used cautiously)

### 5\. Model Analysis

-   Tree diversity analysis
-   Effect of number of trees on performance
-   Overfitting vs generalization comparison
-   Feature importance insights

🔍 Key Learnings
---------------
-   Ensemble learning significantly reduces variance
-   Bootstrapping improves robustness
-   Recall is more critical than accuracy for fraud detection
-   Custom implementations deepen understanding of ML internals
-   Random Forest performs well on non-linear tabular data

🛠️ Tech Stack
--------------
-   Python
-   NumPy
-   Pandas
-   Matplotlib
