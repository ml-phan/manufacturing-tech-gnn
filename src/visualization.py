import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_class_distribution(data, target_col='technology_name'):
    # Count the number of samples per class
    class_percentages = data[target_col].value_counts(
        normalize=True).sort_index().sort_values(ascending=False) * 100

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_percentages.index, y=class_percentages.values,
                hue=class_percentages.index, palette="viridis", legend=False)

    plt.title("Class Distribution in 'technology' (as percentage)")
    plt.ylabel("Percentage (%)")
    plt.xlabel("Technology Class")
    plt.xticks(rotation=90)
    plt.ylim(0, class_percentages.max() + 5)

    # Annotate bars with percentage values
    for index, value in enumerate(class_percentages.values):
        plt.text(index, value + 0.5, f"{value:.1f}%", ha='center')

    plt.tight_layout()
    plt.show()


def plot_topological_features_distribution(data, feature_cols):
    # Plot the distribution of topological features
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(feature_cols):
        plt.subplot(2, 2, i + 1)
        sns.histplot(data[feature], bins=50, kde=True, log_scale=(True, False))
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()