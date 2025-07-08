import pandas as pd

from src.training import *
from src.evaluation import *
from src.visualization import *


def training_one_fold(dataset:pd.DataFrame):
    results_binary = []
    results_multi = []
    for fold in range(dataset.binary_fold.nunique()):
        print(f"Training fold {fold}...")
        features = [
            "faces", "edges", "vertices", "quantity",
            "height", "width", "depth", "volume", "area",
            "bbox_height", "bbox_width", "bbox_depth", "bbox_volume",
            "bbox_area",
        ]
        train_data = dataset[dataset.binary_fold == fold].reset_index(drop=True)
        X = train_data[features]
        X_train, X_test, y_train_index, y_test_index = train_test_split(
            X,
            range(len(X)),
            test_size=0.2,
            random_state=42,
            stratify=train_data["is_cnc"],
        )
        # y_multi_train = train_data["multiclass_labels"].iloc[y_train_index]
        # y_multi_test = train_data["multiclass_labels"].iloc[y_test_index]

        y_binary_train = train_data["is_cnc"].iloc[y_train_index]
        y_binary_test = train_data["is_cnc"].iloc[y_test_index]
        best_search_binary, best_search_multi = training_pipeline_xgboost(
            X_train,
            y_binary_train,
            # y_multi_train,
        )
        # Multiclass predictions
        # y_multi_pred = best_search_multi.best_estimator_.predict(X_test)
        # y_multi_prob = best_search_multi.best_estimator_.predict_proba(X_test)

        # Binary predictions
        y_binary_pred = best_search_binary.best_estimator_.predict(X_test)
        y_binary_prob = best_search_binary.best_estimator_.predict_proba(
            X_test)
        # Evaluate binary classification
        metrics_binary = evaluate_classification(
            y_true=y_binary_test,
            y_pred=y_binary_pred,
            y_prob=y_binary_prob,
        )
        # metrics_multi = evaluate_classification(
        #     y_true=y_multi_test,
        #     y_pred=y_multi_pred,
        #     y_prob=y_multi_prob,
        #     # top_k=3
        # )
        results_binary.append(metrics_binary)
        # results_multi = pd.concat([results_multi, metrics_multi])
    result_df_binary = pd.DataFrame(results_binary).T
    result_df_binary.columns = [f"Fold {i}" for i in range(len(results_binary))]

    results_df_multi = pd.DataFrame()  # Placeholder for multi-class results

    return result_df_binary, results_df_multi

if __name__ == '__main__':
    print("Shit")