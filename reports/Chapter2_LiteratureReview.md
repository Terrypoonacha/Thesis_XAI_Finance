# Chapter 2: Literature Review

## 2.1 Financial Fraud Detection
The detection of fraudulent transactions, such as credit card fraud, has evolved from rule-based systems to complex Machine Learning (ML) models.
- **Traditional Methods**: Expert systems relying on if-then rules (e.g., "Transaction > €5000 at 3 AM -> Flag").
- **Supervised Learning**: Models like Logistic Regression, Random Forest, and XGBoost (our chosen baseline) learn patterns from historical labeled data.
- **Unsupervised Learning**: Anomaly detection (e.g., Isolation Forest) for novel fraud patterns.

While modern ML models like Gradient Boosted Decision Trees (GBDT) offer superior predictive performance (AUPRC > 0.85), they often operate as "Black Boxes".

## 2.2 Explainable AI (XAI) in Finance
To trust these models, practitioners employ XAI techniques.
- **Global Interpretability**: Understanding the model as a whole (e.g., Feature Importance plots).
- **Local Interpretability**: Understanding individual predictions. **SHAP (SHapley Additive exPlanations)** is the gold standard, derived from cooperative game theory, providing unified measures of feature contribution. However, SHAP outputs are technical (e.g., "V14 = -2.3") and lack business context.

## 2.3 The Regulatory Landscape
Financial institutions operate under strict governance frameworks that mandate model transparency.
- **BaFin MaRisk (Minimum Requirements for Risk Management)**: specifically AT 4.3.2 requires institutions to have a comprehensive understanding of the risk models used. "Black Box" models without explanation are generally non-compliant.
- **EU AI Act (2024)**: Classifies AI used in credit scoring and risk assessment as **High-Risk**. **Article 13** explicitly mandates that such systems be "sufficiently transparent to enable users to interpret the system’s output and use it appropriately."

## 2.4 The Knowledge Gap: Automated Compliance
Current literature addresses:
1.  Improving fraud detection accuracy (the ML domain).
2.  Generating technical explanations (the XAI domain).
3.  Defining regulatory requirements (the Legal domain).

There is a significant gap in **Bridging the Divide**: Translating technical XAI outputs ("V14 is high") into legal compliance documentation ("This transaction is flagged due to V14, requiring review under MaRisk AT 4.3.2"). This thesis proposes an **Agentic Workflow** using Large Language Models (LLMs) to automate this "last mile" of explainability.
