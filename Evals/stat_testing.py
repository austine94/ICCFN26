#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 06:19:59 2026

@author: localadmin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 18:00:00 2026

Statistical Significance Analysis Script
Compares SLM performance against Non-AI baselines across 3 experimental scenarios:
1. Experiment 1 (Clean Text, Temp = 0.0)
2. Experiment 1 Noise (Aggregated & per Noise Type)
3. Experiment 2 (Knowledge Graph Structured)
"""

import json
import pandas as pd
import numpy as np
from scipy import stats

def assign_significance_stars(p_val):
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    else:
        return "ns"  # Not Significant


def perform_pairwise_baseline_comparison(df, baseline_name, scenario_label, metric_col='accuracy', match_cols=['iteration']):
    """
    Performs paired statistical tests (Paired t-test & Wilcoxon signed-rank)
    between all candidate models and a designated baseline model.
    """
    models = [m for m in df['model'].unique() if m != baseline_name]
    results = []

    # Filter baseline data
    baseline_df = df[df['model'] == baseline_name]
    if baseline_df.empty:
        print(f"Warning: Baseline '{baseline_name}' not found in dataset for {scenario_label}.")
        return pd.DataFrame()

    for model in models:
        model_df = df[df['model'] == model]

        # Merge on iteration matching columns to ensure paired samples
        merged = pd.merge(
            model_df, 
            baseline_df, 
            on=match_cols, 
            suffixes=('_model', '_baseline')
        )

        if merged.empty:
            continue

        model_scores = merged[f'{metric_col}_model'].values
        baseline_scores = merged[f'{metric_col}_baseline'].values

        mean_model = np.mean(model_scores) * 100
        mean_baseline = np.mean(baseline_scores) * 100
        diff = mean_model - mean_baseline

        # Paired t-test
        try:
            t_stat, p_val_ttest = stats.ttest_rel(model_scores, baseline_scores)
        except Exception:
            t_stat, p_val_ttest = np.nan, 1.0

        # Wilcoxon Signed-Rank Test (Non-parametric alternative)
        try:
            # Drop zero differences for Wilcoxon test
            diffs = model_scores - baseline_scores
            if np.all(diffs == 0):
                w_stat, p_val_wilcox = np.nan, 1.0
            else:
                w_stat, p_val_wilcox = stats.wilcoxon(model_scores, baseline_scores)
        except Exception:
            w_stat, p_val_wilcox = np.nan, 1.0

        results.append({
            "Scenario": scenario_label,
            "Model": model,
            "Baseline": baseline_name,
            "Model Mean (%)": round(mean_model, 2),
            "Baseline Mean (%)": round(mean_baseline, 2),
            "Diff (%)": round(diff, 2),
            "t-stat": round(t_stat, 3) if not np.isnan(t_stat) else "N/A",
            "p-value (t-test)": f"{p_val_ttest:.4e}",
            "Wilcoxon p-val": f"{p_val_wilcox:.4e}",
            "Sig.": assign_significance_stars(p_val_ttest)
        })

    return pd.DataFrame(results)


def run_statistical_suite():
    print("=========================================================")
    print(" RUNNING STATISTICAL SIGNIFICANCE TESTS (SLMs vs BASELINES)")
    print("=========================================================\n")

    all_comparison_tables = []

    # =========================================================================
    # 1. EXPERIMENT 1: CLEAN TEXT (Temp = 0.0)
    # =========================================================================
    try:
        with open('experiment_1_results.json', 'r') as f:
            exp1_data = json.load(f)
        
        df_exp1 = pd.DataFrame(exp1_data)
        
        # Filter for Temperature = 0.0
        if 'temperature' in df_exp1.columns:
            df_exp1_t0 = df_exp1[df_exp1['temperature'] == 0.0]
        else:
            df_exp1_t0 = df_exp1

        exp1_stats = perform_pairwise_baseline_comparison(
            df=df_exp1_t0,
            baseline_name='tfidf-cosine-baseline',
            scenario_label='Exp 1 (Clean, Temp 0.0)',
            metric_col='accuracy',
            match_cols=['iteration']
        )
        all_comparison_tables.append(exp1_stats)

    except FileNotFoundError:
        print("Note: experiment_1_results.json not found. Skipping Exp 1 Clean.")
    except Exception as e:
        print(f"Error processing Exp 1 Clean: {e}")

    # =========================================================================
    # 2. EXPERIMENT 1 NOISE: ALL NOISE TYPES (Temp = 0.0)
    # =========================================================================
    try:
        with open('experiment_1_noise_results.json', 'r') as f:
            exp1_noise_data = json.load(f)

        df_exp1_noise = pd.DataFrame(exp1_noise_data)

        # Overall Noise Comparison (Aggregated across all noise types)
        exp1_noise_overall_stats = perform_pairwise_baseline_comparison(
            df=df_exp1_noise,
            baseline_name='tfidf-cosine-baseline',
            scenario_label='Exp 1 Noise (Aggregated)',
            metric_col='accuracy',
            match_cols=['noise_type', 'iteration']
        )
        all_comparison_tables.append(exp1_noise_overall_stats)

        # Per-Noise Type Analysis
        for noise_type in df_exp1_noise['noise_type'].unique():
            df_sub = df_exp1_noise[df_exp1_noise['noise_type'] == noise_type]
            exp1_noise_type_stats = perform_pairwise_baseline_comparison(
                df=df_sub,
                baseline_name='tfidf-cosine-baseline',
                scenario_label=f'Exp 1 Noise ({noise_type})',
                metric_col='accuracy',
                match_cols=['iteration']
            )
            all_comparison_tables.append(exp1_noise_type_stats)

    except FileNotFoundError:
        print("Note: experiment_1_noise_results.json not found. Skipping Exp 1 Noise.")
    except Exception as e:
        print(f"Error processing Exp 1 Noise: {e}")

    # =========================================================================
    # 3. EXPERIMENT 2: KNOWLEDGE GRAPH STRUCTURED
    # =========================================================================
    try:
        with open('experiment_2_results.json', 'r') as f:
            exp2_raw = json.load(f)

        rows_exp2 = []
        for k_str, iterations in exp2_raw.items():
            k_val = int(k_str)
            for entry in iterations:
                rows_exp2.append({
                    'k': k_val,
                    'iteration': entry.get('iteration', 1),
                    'model': entry['model'],
                    'accuracy': int(entry['is_correct'])  # Convert bool to 1.0 / 0.0
                })
        
        df_exp2 = pd.DataFrame(rows_exp2)

        exp2_stats = perform_pairwise_baseline_comparison(
            df=df_exp2,
            baseline_name='textrank-baseline',
            scenario_label='Exp 2 (KG Structured)',
            metric_col='accuracy',
            match_cols=['k', 'iteration']
        )
        all_comparison_tables.append(exp2_stats)

    except FileNotFoundError:
        print("Note: experiment_2_results.json not found. Skipping Exp 2.")
    except Exception as e:
        print(f"Error processing Exp 2: {e}")

    # =========================================================================
    # DISPLAY & SAVE SUMMARY TABLE
    # =========================================================================
    if all_comparison_tables:
        final_df = pd.concat(all_comparison_tables, ignore_index=True)

        print("\n### PAIRWISE STATISTICAL SIGNIFICANCE RESULTS ###\n")
        print(final_df.to_markdown(index=False))

        output_csv = "statistical_significance_summary.csv"
        final_df.to_csv(output_csv, index=False)
        print(f"\nSignificance summary saved to: {output_csv}")
        print("\nSignificance Legend:")
        print("  *** : p < 0.001 (Highly Significant)")
        print("  **  : p < 0.01  (Very Significant)")
        print("  *   : p < 0.05  (Statistically Significant)")
        print("  ns  : p >= 0.05 (Not Statistically Significant)")

if __name__ == "__main__":
    run_statistical_suite()