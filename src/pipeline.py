from src.training import *
from src.evaluation import *
from src.visualization import *

import pickle
import os
import time

import optuna
import threading
import torch
import torch.nn.functional as F
from sklearn.preprocessing import PowerTransformer, StandardScaler
from pathlib import Path
from tqdm import tqdm


class PyGDatasetScaler:
    def __init__(self):
        self.x_scaler = None
        self.edge_attr_scaler = None
        self.global_features_scaler = None
        self.surface_type_categories = 10  # 0-9
        self.curve_type_categories = 7  # 0-6
        self.shared_face_count_categories = 3  # 2-3-4

    def collect_statistics(self, data_files):
        """
        Collect statistics from all data files for fitting scalers.
        This loads all files to compute global statistics.
        """
        print("Collecting statistics from all files...")

        # Collect all features for fitting scalers
        all_x_features = []
        all_edge_attr_features = []
        all_global_features = []

        for file_path in tqdm(data_files, desc="Loading files for statistics"):
            data = torch.load(file_path, weights_only=False)

            # Extract x features (excluding surface_type at index 8)
            x_without_surface = data.x[:, 1:13]
            all_x_features.append(x_without_surface)

            # Extract edge_attr features (excluding curve_type at index 2)
            edge_attr_without_curve = data.edge_attr[:, 2:4]
            all_edge_attr_features.append(edge_attr_without_curve)

            # Extract global features
            all_global_features.append(data.global_features)

        # Concatenate all features
        all_x = torch.cat(all_x_features, dim=0).numpy()
        all_edge_attr = torch.cat(all_edge_attr_features, dim=0).numpy()
        all_global = torch.cat(all_global_features, dim=0).numpy()

        # Fit scalers
        print("Fitting scalers...")
        self.x_scaler = PowerTransformer()
        self.edge_attr_scaler = PowerTransformer()
        self.global_features_scaler = StandardScaler()

        self.x_scaler.fit(all_x)
        self.edge_attr_scaler.fit(all_edge_attr)
        self.global_features_scaler.fit(all_global)

        print(f"Statistics collected from {len(data_files)} files")
        print(f"X features shape (excluding surface_type): {all_x.shape}")
        print(f"Edge attr shape (excluding curve_type): {all_edge_attr.shape}")
        print(f"Global features shape: {all_global.shape}")

    def transform_data(self, data):
        """
        Transform a single PyG Data object with scaling and one-hot encoding.
        """
        # Handle x features
        surface_type = data.x[:, -1].long()  # Extract surface_type
        x_to_scale = data.x[:, 1:13]

        # Scale x features
        x_scaled = torch.tensor(
            self.x_scaler.transform(x_to_scale.numpy()),
            dtype=torch.float32)

        # One-hot encode surface_type
        surface_one_hot = F.one_hot(surface_type,
                                    num_classes=self.surface_type_categories).float()

        # Combine scaled features with one-hot encoded surface_type
        x_final = torch.cat(
            [x_scaled, data.x[:, 13:-1], surface_one_hot], dim=1)

        # Handle edge_attr features
        shared_face_count = data.edge_attr[:, 0].long()
        shared_face_count = shared_face_count.unsqueeze(1).to(torch.float32)
        curve_type = data.edge_attr[:, 1].long()
        edge_attr_to_scale = data.edge_attr[:, 2:4]

        # Scale edge_attr features
        edge_attr_scaled = torch.tensor(
            self.edge_attr_scaler.transform(edge_attr_to_scale.numpy()),
            dtype=torch.float32)

        curve_one_hot = F.one_hot(
            curve_type,
            num_classes=self.curve_type_categories
        ).float()

        # Combine scaled features with one-hot encoded curve_type
        edge_attr_final = torch.cat(
            [
                edge_attr_scaled, shared_face_count, curve_one_hot,
                data.edge_attr[:, 4:]
            ],
            dim=1)

        # Handle global_features
        global_features_scaled = torch.tensor(
            self.global_features_scaler.transform(
                data.global_features.numpy()), dtype=torch.float32)

        # Create new data object with transformed features
        transformed_data = data.clone()
        transformed_data.x = x_final
        transformed_data.edge_attr = edge_attr_final
        transformed_data.global_features = global_features_scaled

        return transformed_data

    def save_scalers(self, save_path):
        """Save fitted scalers for later use."""
        scalers = {
            'x_scaler': self.x_scaler,
            'edge_attr_scaler': self.edge_attr_scaler,
            'global_features_scaler': self.global_features_scaler,
            'surface_type_categories': self.surface_type_categories,
            'curve_type_categories': self.curve_type_categories
        }
        with open(save_path, 'wb') as f:
            pickle.dump(scalers, f)
        print(f"Scalers saved to {save_path}")

    def load_scalers(self, load_path):
        """Load previously fitted scalers."""
        with open(load_path, 'rb') as f:
            scalers = pickle.load(f)

        self.x_scaler = scalers['x_scaler']
        self.edge_attr_scaler = scalers['edge_attr_scaler']
        self.global_features_scaler = scalers['global_features_scaler']
        self.surface_type_categories = scalers['surface_type_categories']
        self.curve_type_categories = scalers['curve_type_categories']
        print(f"Scalers loaded from {load_path}")


def process_dataset(input_dir, output_dir, scaler_path=None, fit_scalers=True):
    """
    Complete pipeline to process PyG dataset.

    Args:
        input_dir: Directory containing .pt files
        output_dir: Directory to save processed files
        scaler_path: Path to save/load scaler parameters
        fit_scalers: Whether to fit scalers (True) or load existing ones (False)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all .pt files
    data_files = list(input_path.glob("*.pt"))
    print(f"Found {len(data_files)} .pt files")

    # Initialize scaler
    scaler = PyGDatasetScaler()

    if fit_scalers:
        # Collect statistics and fit scalers
        scaler.collect_statistics(data_files)

        # Save scalers if path provided
        if scaler_path:
            scaler.save_scalers(scaler_path)
    else:
        # Load existing scalers
        if scaler_path and os.path.exists(scaler_path):
            scaler.load_scalers(scaler_path)
        else:
            raise ValueError(
                "Scaler path not found. Set fit_scalers=True or provide valid scaler_path")

    # Transform and save all files
    print("Transforming and saving files...")
    for file_path in tqdm(data_files, desc="Scaling pt files"):
        # Load original data
        data = torch.load(file_path, weights_only=False)

        # Transform data
        transformed_data = scaler.transform_data(data)

        # Save transformed data
        output_file = output_path / file_path.name
        torch.save(transformed_data, output_file)

    print(f"Processing complete! Transformed files saved to {output_path}")

    # Print dimension information
    sample_data = torch.load(output_path / data_files[0].name,
                             weights_only=False)
    print(f"\nDimension changes:")
    print(f"Original x shape: {torch.load(data_files[0]).x.shape}")
    print(f"Transformed x shape: {sample_data.x.shape}")
    print(
        f"Original edge_attr shape: {torch.load(data_files[0]).edge_attr.shape}")
    print(f"Transformed edge_attr shape: {sample_data.edge_attr.shape}")
    print(f"Global features shape: {sample_data.global_features.shape}")



def xgboost_optuna(data, n_trials=2000, features=None):
    fold_results = {}
    for fold in sorted(data.binary_fold.unique()):
        print(f"Optimizing fold {fold}:")
        # Samples not in fold will be training data
        X_train = data[data.binary_fold != fold][features]
        y_train = data[data.binary_fold != fold]["is_cnc"]

        # Samples in fold will be test data
        X_test = data[data.binary_fold == fold][features]
        y_test = data[data.binary_fold == fold]["is_cnc"]

        def objective(trial):
            try:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate",
                                                         0.005,
                                                         0.3),
                    "min_child_weight": trial.suggest_int("min_child_weight",
                                                          1, 10),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree",
                                                            0.6, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
                }

                sample_weight = compute_sample_weight(
                    class_weight="balanced", y=y_train
                )

                model = XGBClassifier(**params, use_label_encoder=False,
                                      eval_metric='logloss')
                model.fit(X_train, y_train, sample_weight=sample_weight)

                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)

                train_metrics = evaluate_classification(
                    y_true=y_train,
                    y_pred=model.predict(X_train),
                    y_prob=model.predict_proba(X_train),
                )

                val_metrics = evaluate_classification(
                    y_true=y_test,
                    y_pred=y_pred,
                    y_prob=y_prob,
                )

                trial.set_user_attr("train_auroc", train_metrics["roc_auc"])
                return val_metrics['roc_auc']
            # Return low score if an error occurs
            except Exception as e:
                print(f"Error during trial number: {trial.number}: {e}")
                return 0.0

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.CmaEsSampler(
                                        seed=100))
        study.optimize(objective, n_trials=n_trials, n_jobs=-1,
                       show_progress_bar=True)
        # pbar_cb = MinimalTqdmCallback(
        #     total_trials=n_trials, desc=f"Fold {fold}",
        #     flush_secs=1.0, step=50, disable=False
        # )
        # try:
        #     study.optimize(objective, n_trials=n_trials, n_jobs=-1,
        #                    callbacks=[pbar_cb])
        # finally:
        #     pbar_cb.close()

        fold_results[f"fold_{fold}"] = {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'study': study
        }
    return fold_results


# Example usage
if __name__ == "__main__":
    # Define paths
    input_directory = r"E:\gnn_data\pyg_data_v2"  # Replace with your input directory
    output_directory = r"E:\gnn_data\pyg_data_v2_scaled"  # Replace with your output directory
    scaler_save_path = "scalers.pkl"  # Path to save scaler parameters

    # Process dataset (fit scalers and transform)
    process_dataset(
        input_dir=input_directory,
        output_dir=output_directory,
        scaler_path=scaler_save_path,
        fit_scalers=False
    )
