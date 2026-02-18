# Chapter 3: Methodology

## 3.1 Dataset and Preprocessing
The analysis utilizes the "Credit Card Fraud Detection" dataset.
- **Source**: Kaggle / European Credit Card Transactions.
- **Volume**: ~284,807 transactions.
- **Imbalance**: Only 492 frauds (0.172%).
- **Features**: 28 PCA-transformed features ($V1, V2, ..., V28$), plus `Time` and `Amount`.

Preprocessing steps included:
1.  **Imbalance Handling**: We utilize the `scale_pos_weight` parameter in XGBoost to penalize false negatives, effectively rebalancing the loss function without oversampling the data.
2.  **Stratified Splitting**: To ensure stable evaluation, we employ Stratified K-Fold cross-validation ($k=5$), guaranteeing that the minority class is represented in every training and validation fold.

## 3.2 Model: Extreme Gradient Boosting (XGBoost)
We selected XGBoost for its state-of-the-art performance on tabular data.
- **Objective Function**: Binary Logistic (`binary:logistic`).
- **Optimization**: Gradient Descent on the loss function, adding trees to correct residual errors.
- **Regularization**: L1 and L2 regularization to prevent overfitting on the minority class.

## 3.3 Evaluation Metrics
Given the extreme imbalance, Accuracy is discarded. We focus on:
- **Precision-Recall Curve (PRC)**: Visualizes the trade-off between Precision (positive predictive value) and Recall (sensitivity).
- **AUPRC (Area Under the Precision-Recall Curve)**: Our primary metric for model selection, as it focuses specifically on the performance on the minority (positive) class.
$$ AUPRC = \int_0^1 P(R) dR $$

## 3.4 Explainability: SHAP
To interpret the "Black Box" XGBoost model, we use SHAP (SHapley Additive exPlanations). SHAP values attribute the prediction output to each feature based on game theory.
- **TreeExplainer**: A fast, model-specific algorithm for tree ensembles.
- **Global Interpretability**: We aggregate absolute SHAP values across the dataset to identify the most important features driving fraud detection globally.
