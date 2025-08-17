# DS261MachineLearningAtScale
UC Berkeley Masters of Data Science course DS261: Machine Learning at Scale

This repository contains the final project for UC Berkeley’s Machine Learning Systems at Scale (DS261). The project focuses on building large-scale machine learning pipelines to predict U.S. domestic flight departure delays using PySpark on Databricks.

We work with 2015–2021 U.S. flight data enriched with weather, traffic, and airport network features. The problem is framed as both a regression task (predicting departure delay in minutes) and a classification task (predicting whether a flight is delayed ≥15 minutes).

The challenge is not only predictive accuracy but also scalability, robustness, and temporal generalization — ensuring models trained on historical data can predict well on future unseen years.

Objectives

Predict departure delay (DEP_DELAY) in minutes.

Explore classification + regression hybrid pipelines (e.g., delay filter → regression).

Improve model R² and MAE/RMSE across years.

Build scalable ML pipelines for distributed execution on Spark.

Compare traditional ML models with ensemble and deep learning approaches.

Dataset

Source: U.S. Department of Transportation (BTS) + FAA weather data.

Years: 2015–2021.

Size: ~60M rows.

Key Columns:

Flight identifiers: ORIGIN, DEST, TAIL_NUM, OP_UNIQUE_CARRIER

Temporal: YEAR, MONTH, DAY_OF_WEEK, CRS_DEP_TIME

Target: DEP_DELAY

Engineered Features:

PREV_DEP_DELAY, PREV2_DEP_DELAY, NEXT_FLIGHT_FLAG

Network centrality: ORIGIN_CLOSENESS, DEST_BETWEENNESS

Weather & traffic: ELEVATION, RECENT_DEST_CANCELLED_COUNT_6_2H

Methods
Feature Engineering

Conversion of time strings (e.g., 16:50) into numeric features (CRS_DEP_TIME_MINUTES).

Delay history features (PREV_DEP_DELAY, PREV2_ARRIVAL_DELAY).

Graph-based centrality metrics (degree, closeness, betweenness) for airports.

Rare category handling and target encodings for categorical features.

Pocketization: treat delays >60 min as "major delay" for stability.

Models Implemented

Regression Models:

Random Forest Regressor (PySpark MLlib)

XGBoost (Spark-XGBoost)

CatBoost Regressor (with categorical encodings)

Ensemble: Random Forest + XGBoost stacked/blended

Neural Networks: MLP regressors with PyTorch/Sklearn

Classification Models:

Binary classification (delay ≥15 minutes) using RF & CatBoost

Evaluation with ROC AUC, PR AUC, F1

Hybrid Pipeline:

Use classifier as a gatekeeper → regression invoked only for predicted delays.

Evaluation Metrics
Regression

R²: variance explained by the model

MAE: mean absolute error — interpretable in minutes

RMSE: root mean squared error — penalizes large misses

Formulas:

M
A
E
=
1
𝑛
∑
𝑖
=
1
𝑛
∣
𝑦
𝑖
−
𝑦
^
𝑖
∣
MAE=
n
1
	​

i=1
∑
n
	​

∣y
i
	​

−
y
^
	​

i
	​

∣
R
M
S
E
=
1
𝑛
∑
𝑖
=
1
𝑛
(
𝑦
𝑖
−
𝑦
^
𝑖
)
2
RMSE=
n
1
	​

i=1
∑
n
	​

(y
i
	​

−
y
^
	​

i
	​

)
2
	​


Where:

𝑛
n = number of flights

𝑦
𝑖
y
i
	​

 = actual delay

𝑦
^
𝑖
y
^
	​

i
	​

 = predicted delay

Classification

Accuracy

Precision, Recall, F1

ROC AUC & PR AUC

Results (Highlights)

Best Random Forest Regression: R² ≈ 0.106 on 2019 test data.

Best CatBoost Regression: R² ≈ 0.108 (log-transformed target).

Ensemble RF + XGBoost: R² ≈ 0.0168 (more stable across years).

Random Forest Classifier (≥15 min delay): ROC AUC ≈ 0.67, F1 ≈ 0.63.

Visualizations

Feature importances (RF, XGBoost, CatBoost).

Residual/error plots vs. features (carrier, month, dep_hour).

Confusion matrix and ROC/PR curves for classification.

Prediction vs actual scatterplots.

Pipeline Structure

PySpark ML Pipeline with:

Feature engineering (UDFs, transformations).

StringIndexer, VectorAssembler, StandardScaler.

Model training (RF, XGB, CatBoost).

Evaluation on temporal splits (train = 2015–2018, test = 2019; additional eval on 2020–2021).

Lessons Learned

Temporal splits are crucial — random splits overestimate performance.

MAE vs RMSE: both needed to balance interpretability and robustness.

Classification-first pipelines can help focus regression only where relevant.

Feature engineering (especially delay history + network metrics) was as important as model choice.

Ensemble methods improved stability, but at scale, cost-performance trade-offs must be managed.

Tech Stack

PySpark (MLlib, DataFrames, UDFs)

Databricks (distributed training environment)

XGBoost-Spark, CatBoost, Scikit-learn, PyTorch

Matplotlib, Seaborn, Pandas for visualizations
