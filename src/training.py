import pandas as pd
import numpy as np

from pprint import pprint
from scipy.stats import randint, uniform
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, top_k_accuracy_score,
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score
)


def train_xgboost(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        use_sample_weight=True) -> XGBClassifier:
    """
    Train an XGBoost classifier on the provided training data.
    Ags:
        X_train (pd.DataFrame): Feature matrix for training.
        y_train (pd.Series): Target variable for training.
        use_sample_weight (bool): Whether to use sample weights for training.
    Returns:
        XGBClassifier: Trained XGBoost classifier.
    """

    # Compute sample weights if needed
    sample_weight = None
    if use_sample_weight:
        sample_weight = compute_sample_weight(
            class_weight="balanced", y=y_train
        )

    # Initialize and train the XGBoost model
    model_xgbost = XGBClassifier(
        objective="multi:softprob",
        num_class=len(y_train.unique()),
        random_state=42,
        eval_metric="mlogloss",
        n_estimators=100,
    )
    model_xgbost.fit(X_train, y_train, sample_weight=sample_weight)
    return model_xgbost


def randomizedsearchcv_xgboost(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: dict,
        n_iter: int = 10,
        cv: int = 5,
        scoring: str = "f1_weighted",
        random_state: int = 42,
        use_sample_weight: bool = True,
    ) -> RandomizedSearchCV:
    """
    Perform Randomized Search Cross-Validation on the XGBoost model.
    Args:
        X_train (pd.DataFrame): Feature matrix for training.
        y_train (pd.Series): Target variable for training.
        params (dict): Dictionary hyperparameters to be tuned.
        n_iter (int): Number of parameter settings that are sampled.
        cv (int): Number of folds in cross-validation.
        scoring (str): Scoring method to use.
        random_state (int): Random state for reproducibility.
        use_sample_weight (bool): Whether to use sample weights for training.-
    Returns:
        RandomizedSearchCV: Fitted RandomizedSearchCV object.
    """
    sample_weight = None
    if use_sample_weight:
        sample_weight = compute_sample_weight(
            class_weight="balanced", y=y_train
        )

    # Initialize the XGBoost model based on multi or binary classification
    unique_labels = y_train.unique()
    if len(unique_labels) == 2:
        model_xgboost = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state
        )
    else:
        model_xgboost = XGBClassifier(
            objective="multi:softprob",
            num_class=len(unique_labels),
            eval_metric="mlogloss",
            random_state=random_state
        )
    # model_xgboost = XGBClassifier()

    search = RandomizedSearchCV(
        estimator=model_xgboost,
        param_distributions=params,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        verbose=0,
    )
    search.fit(X_train, y_train, sample_weight=sample_weight)
    return search


def training_pipeline_xgboost(X_train, y_binary_train, y_multi_train=None):
    """
    Run the training pipeline for XGBoost on both binary and multi-class datasets.
    Args:
        X_train (pd.DataFrame): Feature matrix for training.
        y_binary_train (pd.Series): Binary target variable for training.
        y_multi_train (pd.Series): Multi-class target variable for training.
    Returns:
        tuple: Best RandomizedSearchCV objects for binary and multi-class datasets.

    """
    best_search_binary = None
    best_search_multi = None
    params = {
        "n_estimators": randint(100, 300),
        "max_depth": randint(3, 10),
        "learning_rate": uniform(0.01, 0.3),
    }
    best_search_binary = randomizedsearchcv_xgboost(X_train, y_binary_train,
                                                    params)
    # best_search_multi = randomizedsearchcv_xgboost(X_train, y_multi_train,
    #                                                params)
    return best_search_binary, best_search_multi


