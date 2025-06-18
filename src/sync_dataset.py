import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold


# Alternative approach using iterdir() if glob is slow
def build_file_lookup(directory: Path, suffix: str):
    """
    Build a lookup dictionary mapping item_id to file path.
    """
    file_lookup = {}

    for file_path in directory.glob(f"*.{suffix}"):
        # Extract item_id using split - more robust for {item_id}_file_name.* format
        item_id_str = file_path.stem.split("_")[0]
        if item_id_str.isdigit():
            item_id = int(item_id_str)
            # Store only the first match (like your original code)
            if item_id not in file_lookup:
                file_lookup[item_id] = str(file_path)

    return file_lookup


def assign_folds_based_on_labels(dataframe, label_column, num_folds=5,
                                 random_state=42):
    """
    Assign folds based on the specified label column.
    Uses StratifiedKFold to ensure balanced distribution of labels across folds.
    """
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True,
                          random_state=random_state)
    fold_assignments = np.zeros(len(dataframe))

    for fold_idx, (_, val_idx) in enumerate(
            skf.split(dataframe, dataframe[label_column])):
        fold_assignments[val_idx] = fold_idx

    return fold_assignments


def fold_assignment(dataframe, num_folds=5, random_state=42):
    """
        Assigns fixed stratified K-fold assignments to a DataFrame based on specified label columns.
        """

    dataframe['binary_fold'] = assign_folds_based_on_labels(
        dataframe,
        'is_cnc',
        num_folds=num_folds,
        random_state=random_state,
    )
    dataframe['multiclass_fold'] = assign_folds_based_on_labels(
        dataframe,
        'multiclass_labels',
        num_folds=num_folds,
        random_state=random_state,
    )
    return dataframe


def sync_dataset(dataframe):
    # Define base directories
    graphml_dir = Path(r"E:/gnn_data/graphml_files")
    pointcloud_dir = Path(r"E:/gnn_data/pointcloud_files")

    # Build lookup dictionaries once
    print("Building file lookups...")
    graphml_lookup = build_file_lookup(graphml_dir, "graphml")
    pointcloud_lookup = build_file_lookup(pointcloud_dir, "txt")

    print(f"Found {len(graphml_lookup)} GraphML files")
    print(f"Found {len(pointcloud_lookup)} PointCloud files")

    # Fast lookup using map() - much faster than apply()
    dataframe["graphml_file"] = dataframe["item_id"].map(graphml_lookup)
    dataframe["pointcloud_file"] = dataframe["item_id"].map(pointcloud_lookup)

    # Check for missing files
    missing_graphml = dataframe["graphml_file"].isnull().sum()
    missing_pc = dataframe["pointcloud_file"].isnull().sum()
    print(f"Missing GraphML files: {missing_graphml}")
    print(f"Missing PointCloud files: {missing_pc}")

    # Optional: Show some statistics
    total_items = len(dataframe)
    print(f"Total items in DataFrame: {total_items}")
    print(
        f"GraphML match rate: {(total_items - missing_graphml) / total_items * 100:.1f}%")
    print(
        f"PointCloud match rate: {(total_items - missing_pc) / total_items * 100:.1f}%")
    dataframe = dataframe.dropna().reset_index(drop=True)
    dataframe.to_csv("../data/synced_dataset.csv", index=False)
    return dataframe


if __name__ == '__main__':
    df = pd.read_csv("../data/dataset.csv").drop("material_name", axis=1)
    df = sync_dataset(df)
    df = fold_assignment(df, num_folds=10, random_state=42)
    df.to_csv("../data/synced_dataset.csv", index=False)
