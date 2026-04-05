"""Core Skills Drill — Descriptive Analytics

Compute summary statistics, plot distributions, and create a correlation
heatmap for the sample sales dataset.

Usage:
    python drill_eda.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def compute_summary(df):
    numeric_df = df.select_dtypes(include='number')
    summary = numeric_df.agg(['count', 'mean', 'std', 'min', 'max'])
    summary.loc['median'] = numeric_df.median()
    summary.to_csv('output/summary.csv')
    return summary


def plot_distributions(df, columns, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, col in enumerate(columns):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(col)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_correlation(df, output_path):
    corr = df.select_dtypes(include='number').corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    os.makedirs("output", exist_ok=True)
    df = pd.read_csv('data/sample_sales.csv')
    compute_summary(df)
    plot_distributions(df, ['quantity', 'unit_price', 'quantity', 'unit_price'], 'output/distributions.png')
    plot_correlation(df, 'output/correlation.png')
    print("Done!")


if __name__ == "__main__":
    main()