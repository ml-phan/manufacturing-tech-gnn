import numpy as np
import pandas as pd

from pprint import pprint
from sklearn.metrics import (
    accuracy_score, f1_score, top_k_accuracy_score,
    classification_report, confusion_matrix, roc_auc_score, average_precision_score
)


def evaluate_classification(y_true, y_pred, y_prob=None, top_k=3):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "top_{}_accuracy".format(top_k):
            top_k_accuracy_score(y_true, y_prob, k=top_k)
            if len(np.unique(y_true)) != 2
            else None
    }

    if y_prob is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
        metrics["average_precision"] = average_precision_score(y_true, y_prob[:, 1])

    return metrics


def compute_per_class_accuracy(y_true, y_pred, label_encoder):
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    labels = label_encoder.inverse_transform(np.unique(y_true))

    return pd.DataFrame({
        "Class": labels,
        "Accuracy": per_class_acc,
        "Support": cm.sum(axis=1)
    })


def print_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred)
    print(report)