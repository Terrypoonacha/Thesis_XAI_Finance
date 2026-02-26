# Agentic XAI for Financial Compliance (Master's Thesis)

An automated framework for financial fraud auditing, designed for **EU AI Act Article 10, 13, & 14** and **BaFin MaRisk AT 4.3.1/4.3.2** compliance.

## 🚀 Interactive Visual Walkthrough
To provide a reproducible "Visual Proof" of the project's lifecycle (from modeling to audit), run the master script:

```bash
python src/full_walkthrough.py
```

This will walk you through 4 stages:
1. **Model Performance Audit**: XGBoost vs. Random Forest comparison and performance table.
2. **Global Interpretability**: SHAP Summary and Regulatory Relevance tagging.
3. **Agentic Reasoning Trace**: Flowchart visualization of the ReAct reasoning path.
4. **Bias & Generalization**: Demographic disparity and Semantic Gap analysis on BAF data.

## 🛡️ Compliance Cockpit (UI)
The project includes a Streamlit dashboard for human-centric evaluation:
```bash
streamlit run src/streamlit_app.py
```

## 📂 Project Structure
- `src/sprint_1_pipeline.py`: Baseline modeling and performance table.
- `src/xai_global.py`: Global interpretability and compliance tagging.
- `src/agentic_auditor.py`: Core Agent logic and Flowchart generator.
- `src/baf_bias_audit.py`: Bias detection and semantic dataset profiling.
- `src/full_walkthrough.py`: Master visual walkthrough command.

## 📚 Compliance Alignment
- **Article 10 (Data Governance)**: Bias detection on the BAF dataset.
- **Article 13 (Transparency)**: SHAP-driven explanations and regulatory retrieval.
- **Article 14 (Human Oversight)**: Manual override controls in the Cockpit.
- **MaRisk AT 4.3.2 (Model Risk)**: Academic feature importance analysis.
