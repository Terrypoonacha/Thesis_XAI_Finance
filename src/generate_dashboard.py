import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from pathlib import Path

# Config
PROJECT_ROOT = Path("c:/Users/terry/OneDrive/Desktop/Thesis_XAI_Finance/Thesis_XAI_Finance")
FIGURES_DIR = PROJECT_ROOT / "reports" / "Figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def generate_dashboard():
    # Set professional style
    plt.style.use('dark_background')
    sns.set_context("talk")
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Agentic XAI Project Lifecycle Dashboard', fontsize=32, fontweight='bold', color='#4A90E2')
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # Plot A: Model Performance Evolution (AUPRC)
    perf_data = {
        'Stage': ['ULB Baseline', 'ULB Optimized', 'BAF Base'],
        'AUPRC': [0.8788, 1.0000, 0.9850]
    }
    df_perf = pd.DataFrame(perf_data)
    sns.barplot(x='Stage', y='AUPRC', data=df_perf, ax=axes[0, 0], palette='Blues_r')
    axes[0, 0].set_title('Model Performance Evolution', fontsize=20, pad=20)
    axes[0, 0].set_ylim(0.8, 1.05)
    for i, v in enumerate(df_perf['AUPRC']):
        axes[0, 0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')

    # Plot B: Bias Detection (Demographic Disparity)
    bias_data = {
        'Group': ['Age < 30', 'Age > 60'],
        'Fraud Prevalence (%)': [1.2, 0.9]
    }
    df_bias = pd.DataFrame(bias_data)
    sns.barplot(x='Group', y='Fraud Prevalence (%)', data=df_bias, ax=axes[0, 1], palette='OrRd_r')
    axes[0, 1].set_title('Bias Detection (Fraud Rate Disparity)', fontsize=20, pad=20)
    axes[0, 1].set_ylim(0, 1.5)
    for i, v in enumerate(df_bias['Fraud Prevalence (%)']):
        axes[0, 1].text(i, v + 0.05, f'{v:.1f}%', ha='center', fontweight='bold')

    # Plot C: Knowledge & Feature Growth
    growth_data = {
        'Sprint': [1, 2, 3, 4, 5, 6],
        'Features': [28, 28, 30, 30, 30, 60], # 30 ULB + 30 BAF approx
        'Legal Citations': [0, 5, 8, 12, 14, 15]
    }
    df_growth = pd.DataFrame(growth_data)
    sns.lineplot(x='Sprint', y='Features', data=df_growth, ax=axes[1, 0], marker='o', label='F_Semantic Features', color='#4A90E2')
    sns.lineplot(x='Sprint', y='Legal Citations', data=df_growth, ax=axes[1, 0], marker='s', label='L_Legal Citations', color='#50E3C2')
    axes[1, 0].set_title('Growth of Knowledge Base', fontsize=20, pad=20)
    axes[1, 0].legend()

    # Plot D: Project Timeline
    sprints = [
        "S1: Foundation\n(XGBoost / SHAP)",
        "S2: Brain\n(ReAct / RAG)",
        "S3: UI\n(Cockpit / Eval)",
        "S4: Legal\n(Art 14 / Scorecards)",
        "S5: Stress\n(Edges / Override)",
        "S6: Global\n(BAF / Bias)"
    ]
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Project Timeline (Milestones)', fontsize=24, pad=20)
    
    for i, sprint in enumerate(sprints):
        y_pos = 0.85 - (i * 0.15)
        axes[1, 1].text(0.1, y_pos, f"▸ {sprint}", fontsize=18, color='#F5A623', transform=axes[1, 1].transAxes)

    # Save
    save_path = FIGURES_DIR / "project_lifecycle_dashboard.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Dashboard saved to: {save_path}")

if __name__ == "__main__":
    generate_dashboard()
