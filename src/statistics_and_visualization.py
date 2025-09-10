```python
# statistics_and_visualization.py
# Module for statistical analysis and visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from tqdm import tqdm
from datetime import datetime
from config import SIGNIFICANCE_LEVEL, OUTPUT_DIR, PLOT_FORMATS, FIGURE_SIZE, ANCOVA_FIGURE_SIZE, FONT
import os

# Configure plotting
plt.rcParams['font.sans-serif'] = [FONT]
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font=FONT)

def run_statistical_analysis(df: pd.DataFrame) -> list:
    """Perform ANOVA and Tukey's HSD tests for specified metrics."""
    print(f"[{datetime.now()}] --- Starting statistical analysis ---")
    metrics_to_test = [
        'tokens_per_second', 'shannon_entropy', 'perplexity_proxy', 'novelty_score', 'ttr',
        'graph_density', 'avg_clustering', 'psi_proxy', 'analogy_rate', 'internal_coherence', 'distinct_2'
    ]
    stats_results = []
    
    for metric in tqdm(metrics_to_test, desc="Statistical analysis"):
        print(f"\n{'='*30}\n[{datetime.now()}] --- Analyzing metric: {metric} ---\n{'='*30}")
        try:
            # ANOVA
            model = ols(f'{metric} ~ C(condition)', data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            ss_condition = anova_table['sum_sq']['C(condition)']
            ss_residual = anova_table['sum_sq']['Residual']
            partial_eta_squared = ss_condition / (ss_condition + ss_residual)
            p_value = anova_table['PR(>F)']['C(condition)']
            stats_results.append({
                'metric': metric,
                'f_value': anova_table['F']['C(condition)'],
                'p_value': p_value,
                'eta_squared': partial_eta_squared
            })
            print(f"[{datetime.now()}] ANOVA Results:")
            print(anova_table)
            print(f"[{datetime.now()}] Effect Size (Partial Eta Squared, η²): {partial_eta_squared:.4f}")
            
            # Tukey's HSD if significant
            if p_value < SIGNIFICANCE_LEVEL:
                print(f"[{datetime.now()}] Post-hoc Test (Tukey's HSD):")
                tukey_result = pairwise_tukeyhsd(endog=df[metric], groups=df['condition'], alpha=SIGNIFICANCE_LEVEL)
                print(tukey_result)
            else:
                print(f"[{datetime.now()}] ANOVA not significant, skipping post-hoc test.")
        except Exception as e:
            print(f"[{datetime.now()}] Error in statistical analysis for {metric}: {e}")
            stats_results.append({
                'metric': metric,
                'f_value': np.nan,
                'p_value': np.nan,
                'eta_squared': np.nan
            })
    
    return stats_results

def create_visualizations(df: pd.DataFrame) -> list:
    """Create bar plots and ANCOVA visualizations."""
    print(f"[{datetime.now()}] --- Starting visualization ---")
    metrics_to_plot = ['shannon_entropy', 'perplexity_proxy', 'novelty_score']
    conditions = df['condition'].unique()
    plot_files = []
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Bar plots with error bars
    print(f"[{datetime.now()}] --- Generating bar plots ---")
    for metric in tqdm(metrics_to_plot, desc="Generating bar plots"):
        plt.figure(figsize=FIGURE_SIZE)
        sns.barplot(x='condition', y=metric, data=df, order=conditions, capsize=0.1, errorbar='ci')
        plt.title(f'“{metric}”指标在不同条件下的均值 (95% CI)', fontsize=16)
        plt.ylabel(f"均值 (Mean of {metric})")
        plt.xlabel("实验条件 (Condition)")
        plt.tight_layout()
        
        for fmt in PLOT_FORMATS:
            plot_filename = f"{OUTPUT_DIR}/barchart_{metric}.{fmt}"
            plt.savefig(plot_filename)
            plot_files.append(plot_filename)
            print(f"[{datetime.now()}] Saved plot: {plot_filename}")
        plt.close()
    
    # ANCOVA visualization
    print(f"[{datetime.now()}] --- Generating ANCOVA visualization ---")
    metric_ancova = 'shannon_entropy'
    plt.figure(figsize=ANCOVA_FIGURE_SIZE)
    sns.lmplot(x='prompt_length', y=metric_ancova, hue='condition', data=df,
               hue_order=conditions, height=ANCOVA_FIGURE_SIZE[1], aspect=ANCOVA_FIGURE_SIZE[0]/ANCOVA_FIGURE_SIZE[1])
    plt.title(f'“{metric_ancova}”与Prompt长度的关系 (ANCOVA可视化)', fontsize=16)
    plt.ylabel(f"值 (Value of {metric_ancova})")
    plt.xlabel("Prompt总长度 (Total Prompt Length)")
    plt.tight_layout()
    
    for fmt in PLOT_FORMATS:
        plot_filename = f"{OUTPUT_DIR}/ancova_visualization.{fmt}"
        plt.savefig(plot_filename)
        plot_files.append(plot_filename)
        print(f"[{datetime.now()}] Saved plot: {plot_filename}")
    plt.close()
    
    return plot_files