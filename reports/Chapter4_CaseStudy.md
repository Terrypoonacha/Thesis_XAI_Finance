# Chapter 4: The Agentic Auditor Case Study

## 4.1 The Transparency Gap
While XAI techniques like SHAP provides feature attribution (e.g., "Feature V14 contributed +2.3 to fraud score"), they fail to provide *context*. A compliance officer cannot submit "V14 is high" to BaFin. They need to know *why* V14 matters and *which* regulation requires its explanation. This is the "Transparency Gap".

## 4.2 Methodology: The Agentic Workflow
To bridge this gap, we implemented an **Agentic Auditor** using a ReAct (Reasoning + Acting) architecture.
1. **SHAP Fetcher**: Extracts technical explanations from the XGBoost model.
2. **Regulatory Retriever**: Semantic/Keyword search across BaFin MaRisk and EU AI Act PDF documents.
3. **LLM Synthesis**: Google Gemini Pro acts as the reasoning engine to combine these inputs into a compliance memo.

## 4.3 Case Study: Transaction 541
We analyzed Transaction 541, a high-risk flagged transaction.

### 4.3.1 Technical Explanation (SHAP)
The model flagged this transaction primarily due to:
- **V14**: High negative value (reducing typical behavior score).
- **V4**: Abnormal variance.
- **V10**: Deviation from mean.
*(Data sourced from SHAP Fetcher tool)*

### 4.3.2 Regulatory Context
The Agent identified the following relevant regulations:
- **EU AI Act, Article 13 (Transparency)**: High-risk AI systems must be sufficiently transparent to enable users to interpret outputs.
- **BaFin MaRisk AT 4.3.2**: Requires adequate understanding of models used in risk management.

### 4.3.3 Generated Compliance Memo
*Below is the automated output from the Agentic Auditor:*

---
**TO**: Internal Audit Committee
**FROM**: Agentic Compliance Officer
**DATE**: 2024-10-24
**SUBJECT**: AML Flag Justification - Transaction 541

**summary**:
Transaction 541 was flagged with a probability of 99.8%. The primary driver was **Feature V14**, which exhibited a highly unusual negative value.

**Regulatory Analysis**:
Under **Article 13 of the EU AI Act** (Page 46), we are required to ensure the system is "sufficiently transparent". Relying solely on the opaque feature "V14" may violate this requirement if not mapped to a real-world behavior.
Additionally, **BaFin MaRisk AT 4.3.2** (Page 35) mandates that we understand the risk models.

**Recommendation**:
1. Investigate the business meaning of V14.
2. Document the specific deviation for this transaction.
3. If V14 remains opaque, retrain the model with interpretable features to ensure compliance.
---

## 4.4 Conclusion
The Agentic Auditor successfully transformed a raw score into an actionable, cited compliance document, demonstrating the potential of LLMs to automate the "last mile" of XAI in finance.
