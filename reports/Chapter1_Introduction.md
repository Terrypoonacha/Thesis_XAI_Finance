# Chapter 1: Introduction

## 1.1 Business Context: The Cost of Fraud
Financial fraud is a multi-billion dollar problem for the global banking industry. As digital transactions increase, so does the sophistication of fraudulent activities. Traditional rule-based systems are often too rigid, while advanced machine learning models, though effective, often lack transparency. This "black box" nature creates a significant barrier to adoption, especially in highly regulated environments where decisions must be justifiable to auditors and customers.

## 1.2 The Problem: Imbalanced Data & Black Boxes
Detecting fraud is akin to finding a needle in a haystack. In our dataset, valid transactions heavily outnumber fraudulent ones, with fraud accounting for only **0.17%** of all transactions. This extreme class imbalance renders standard accuracy metrics misleading; a model could predict "no fraud" for every case and achieve 99.83% accuracy while failing to detect a single fraudulent transaction.

Furthermore, high-performance algorithms like XGBoost, while capable of handling such imbalance and delivering superior predictive power, are complex and opaque. A purely predictive model is insufficient if it cannot explain *why* a transaction was flagged.

## 1.3 Thesis Objective
This thesis aims to bridge the gap between high-performance fraud detection and model interpretability. We propose a framework that leverages:
1.  **Robust Modeling**: Using XGBoost with techniques to handle class imbalance (e.g., `scale_pos_weight`).
2.  **Explainable AI (XAI)**: employing SHAP (SHapley Additive exPlanations) to provide both global and local interpretability.

By combining these, we aim to provide a solution that not only detects fraud with high precision but also provides the "why" behind every decision, satisfying both regulatory compliance and operational requirements.
