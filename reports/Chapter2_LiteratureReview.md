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
Financial institutions operate under strict governance frameworks that mandate model transparency. This thesis focuses on two critical frameworks: The EU AI Act and BaFin's MaRisk.

### 2.3.1 The EU AI Act (Regulation 2024/1689)
The efficient functioning of the internal market requires a uniform legal framework for AI products. The **Regulation (EU) 2024/1689** (EU AI Act) categorizes AI systems by risk. Systems used for **credit scoring** and **risk assessment** in relation to natural persons are classified as **High-Risk AI Systems** (Annex III).

**Article 13: Transparency and Provision of Information to Users**
Article 13 is the cornerstone of explainability in the Act. It mandates that high-risk AI systems must be designed and developed in such a way to ensure that their operation is sufficiently transparent to enable deployers to interpret the system's output and use it appropriately.
*   **Key Requirement**: The instructions for use must include concise, complete, correct, and clear information that is relevant, accessible, and comprehensible to users.
*   **Impact on Black Boxes**: Purely opaque models (like deep neural networks or unconstrained XGBoost ensembles) may fail this requirement if they cannot provide "concise and clear" rationale for their decisions.

**Article 14: Human Oversight**
While Article 13 focuses on information, **Article 14** addresses the *agency* of the human operator. High-risk AI systems must be designed to enable effective human oversight.
*   **The "Rubber Stamp" Risk**: A key concern of the regulator is automation bias, where humans blindly accept model outputs.
*   **Agentic Mitigation**: The "Compliance Cockpit" proposed in this thesis directly addresses Article 14 by transforming the AI from a decision-maker to a decision-support system. By providing a reasoned "Compliance Memo" alongside the score, the human officer is empowered to exercise critical judgment, satisfying the requirement that oversight measures enable individuals to "interpret the system’s output" and "disregard, override or reverse the output" (Art. 14(4)).

### 2.3.2 BaFin MaRisk (Minimum Requirements for Risk Management)
In Germany, the Federal Financial Supervisory Authority (BaFin) enforces the **MaRisk** circular (10/2021).

**AT 4.3.1: Changes in risk management systems**
This section governs the modification and expansion of risk management systems, including IT systems and models.
*   **Requirement**: Before integrating new models (like ML-based fraud detection), institutions must analyze the impact on personnel and technical/organizational processes.
*   **Model Understanding**: Implicit in AT 4.3.1 is the requirement that the institution must *understand* the new system to assess its impact. A "Black Box" that cannot be audited or understood poses an operational risk that must be mitigated.
*   **Connection to AT 4.3.2**: While AT 4.3.1 focuses on the change process, AT 4.3.2 mandates the suitability of the procedures themselves. Together, they create a regulatory environment where unexplained model outputs are unacceptable for critical risk decisions.

## 2.4 The Knowledge Gap: Automated Compliance
Current literature addresses:
1.  Improving fraud detection accuracy (the ML domain).
2.  Generating technical explanations (the XAI domain).
3.  Defining regulatory requirements (the Legal domain).

There is a significant gap in **Bridging the Divide**: Translating technical XAI outputs ("V14 is high") into legal compliance documentation ("This transaction is flagged due to V14, requiring review under MaRisk AT 4.3.2"). This thesis proposes an **Agentic Workflow** using Large Language Models (LLMs) to automate this "last mile" of explainability.
