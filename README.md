# Customer Churn Prediction — STAT 432 Final Project

## Overview
This project builds and compares multiple classification models to predict customer churn in an e-commerce setting. Using a dataset of customer demographics, behavioral metrics, and satisfaction indicators, we evaluate logistic regression, K-nearest neighbors (KNN), support vector machines (SVM), decision trees, and ensemble methods (bagging and random forest) to identify which approach best flags at-risk customers.

Given the class imbalance in the data (~83% no-churn, ~17% churn), we prioritize **sensitivity** and **test error** over raw accuracy when comparing models, since correctly identifying churners matters more than overall correctness for retention use cases.

## Authors
- Jada Giddens
- Colin Gu 
- Adi Agarwal
- Dev Langaliya

## Dataset
- **Source:** [E-Commerce Customer Churn dataset](https://www.kaggle.com) by Ankit Verma (Kaggle)
- **Raw size:** 5,630 rows × 19 columns
- **Cleaned size:** 3,774 rows × 19 columns (rows with missing values removed — affected columns: `Tenure`, `HourSpendOnApp`, `OrderAmountHikeFromlastYear`, `CouponUsed`, `OrderCount`, `DaySinceLastOrder`, each <5% missing)
- **Target variable:** `Churn` (0 = did not churn, 1 = churned)
- **Class balance:** 83.2% no churn / 16.8% churn

### Key EDA Findings
- Most numerical features are right-skewed (exception: `CashbackAmount`)
- Customers who filed complaints and those with more registered devices churn more frequently
- Lower tenure, fewer days since last order, and more registered addresses are associated with higher churn

## Models & Results

| Model | Test Error | Sensitivity | Specificity |
|---|---|---|---|
| Logistic Regression (Best Subset / BIC) | 0.1060 | 0.5846 | 0.9584 |
| Logistic Regression (LASSO) | 0.1113 | 0.5769 | 0.9536 |
| KNN (K=3) | — | 0.5952 (Recall) | — |
| SVM — Linear | 0.10 | 0.55 | 0.97 |
| SVM — Polynomial | 0.06 | 0.86 | 0.96 |
| SVM — Radial | 0.04 | 0.79 | 0.99 |
| SVM — Radial (Weighted) | 0.05 | 0.78 | 0.99 |
| Simple Classification Tree | 0.0808 | 0.5776 | 0.9812 |
| Pruned Classification Tree | 0.0848 | 0.5776 | 0.9765 |
| Bagging | 0.0212 | 0.9397 | 0.9859 |
| **Random Forest (untuned)** | **0.0146** | **0.9138** | **0.9984** |
| Random Forest (tuned) | 0.0212 | 0.8793 | 0.9969 |

**Best performer:** Untuned Random Forest achieved the lowest test error (0.0146) and strong sensitivity (91.38%), making it the top choice for identifying at-risk customers.

**Most interpretable:** Logistic Regression (Best Subset via BIC) — provides clear coefficient-level insight into churn drivers, such as complaints, marital status, and tenure.

## Methodology Summary
- **Train/test split:** 80/20 across all models
- **Logistic Regression:** Best Subset Selection (minimizing BIC) and LASSO (via 10-fold CV to select λ) approaches, with one-hot encoded categorical variables
- **KNN:** Features scaled using training-set statistics only; K tuned via 5-fold CV optimizing ROC AUC (K=3 selected)
- **SVM:** Linear, polynomial, and radial kernels tuned via 8-fold CV; radial kernel also tested with class weighting (1:2, 1:3, 1:5, 1:10) to address imbalance
- **Decision Trees:** Single tree grown then pruned via cost-complexity pruning with the 1-SE rule
- **Bagging / Random Forest:** `randomForest` with `mtry = p` (bagging) vs. `mtry = floor(sqrt(p))` (random forest); random forest additionally tuned over `mtry` and `nodesize` via OOB error grid search

## My Contributions
- **Introduction / Abstract / Interpretations** — co-wrote the project introduction and abstract, and contributed to interpreting results across models
- **Literature Review** — researched and summarized prior work on churn prediction methods (e.g., Prabadevi et al. 2023, The Pecan Team 2026) to motivate our choice of models
- **SVM model** — implemented and tuned support vector machines across linear, polynomial, and radial kernels (via 8-fold cross-validation), including a weighted radial variant to address class imbalance; evaluated each kernel on test error, sensitivity, and specificity, and identified the polynomial kernel as the best choice from a business perspective due to its high sensitivity in catching at-risk customers

Full code for this section is in **Appendix C** of the report.
