import json
import pandas as pd
import time
import torch

from src.pointnet2_models import get_model, get_loss, PointCloudDataset
    # PointNet2, get_pointnet2_loss
from src.pointnet2_utils_v2 import PointNet2Classification
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score, AUROC, AveragePrecision, \
    MetricCollection, MetricTracker
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_metrics_tracker():
    num_classes = 2
    metrics_tracker = {
        "train_tracker": MetricTracker(
            MetricCollection(
                {
                    "acc": Accuracy(
                        task="multiclass", num_classes=num_classes),
                    "f1": F1Score(
                        task="multiclass",
                        num_classes=num_classes,
                        average="macro"),
                    "avp": AveragePrecision(
                        task="multiclass",
                        num_classes=num_classes,
                        average="macro"),
                    "auroc": AUROC(
                        task="multiclass", num_classes=num_classes)
                }
            )).to(device),
        "val_tracker": MetricTracker(
            MetricCollection(
                {
                    "acc": Accuracy(
                        task="multiclass", num_classes=num_classes),
                    "f1": F1Score(
                        task="multiclass",
                        num_classes=num_classes,
                        average="macro"),
                    "avp": AveragePrecision(
                        task="multiclass",
                        num_classes=num_classes,
                        average="macro"),
                    "auroc": AUROC(
                        task="multiclass", num_classes=num_classes)
                }
            )).to(device)}
    return metrics_tracker


def build_dataset(validation_fold=0):
    train_datasets = []
    val_dataset = None
    for i in range(10):
        fold_dir = pd.read_csv(
            fr"E:\gnn_data\pointcloud_files_folds\fold_data_{i:02d}.csv")
        dataset = PointCloudDataset(fold_dir)
        if i == validation_fold:
            val_dataset = dataset
        else:
            train_datasets.append(dataset)
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    # Print dataset sizes
    print(f"Training dataset size: {len(train_dataset)}", end=". ")
    print(f"Validation dataset size: {len(val_dataset)}")
    return train_dataset, val_dataset


def build_dataset_v2(validation_fold=0):
    train_dfs = []
    val_dataset = None
    for i in range(10):
        fold_df = pd.read_csv(
            fr"E:\gnn_data\pointcloud_files_folds\fold_data_{i:02d}.csv")
        if i == validation_fold:
            val_dataset = PointCloudDataset(fold_df)
        else:
            train_dfs.append(fold_df)
    train_df = pd.concat(train_dfs, ignore_index=True)
    train_dataset = PointCloudDataset(train_df)
    # Print dataset sizes
    print(f"Training dataset size: {len(train_dataset)}", end=". ")
    print(f"Validation dataset size: {len(val_dataset)}")
    return train_dataset, val_dataset

def train_pointnet(
        validation_fold=0,
        num_epochs=50,
        pointnet_version=1,
):
    training_start = time.perf_counter()

    # Build datasets and dataloaders
    print("Building datasets and dataloaders...")
    train_dataset, val_dataset = build_dataset(validation_fold)
    train_loader = DataLoader(train_dataset, batch_size=32,
                          shuffle=True, drop_last=True,
                          num_workers=4, pin_memory=True,
                          persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=32,
                          shuffle=False, drop_last=False    ,
                          num_workers=4, pin_memory=True,
                          persistent_workers=True)

    # Initialize model, loss, optimizer, scheduler, and metrics tracker
    print("Initializing model, loss function, optimizer, scheduler, and metrics tracker...")

    if pointnet_version == 1:
        model = get_model().to(device)
        criterion = get_loss().to(device)
    else:
        # PointNet2 model version 2
        model = PointNet2Classification(num_classes=2, input_channels=3).to(device)
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
                                 betas=(0.9, 0.999), weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20,
                                                gamma=0.7)
    metrics_tracker = init_metrics_tracker()

    # Store loss, the best metrics and model state
    loss_history = {"train_loss": [], "val_loss": []}
    best_val_f1 = 0.0
    best_val_auroc = 0.0
    best_val_ap = 0.0
    best_model_state = None

    for epoch in range(num_epochs):

        # Training Phase
        model.train()
        total_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        metrics_tracker["train_tracker"].increment()
        metrics_tracker["val_tracker"].increment()

        for points, labels in train_pbar:
            points, labels = points.to(device), labels.to(device)
            optimizer.zero_grad()

            if pointnet_version == 1:
                # Forward PointNet2
                pred, trans_feat = model(points)
                loss = criterion(pred, labels, trans_feat)
            else:
                # Forward PointNet2
                pred = model(points)
                loss = criterion(pred, labels)

            # Backward
            loss.backward()
            optimizer.step()

            # Stats
            total_train_loss += loss.item()

            # Update metrics tracker
            metrics_tracker["train_tracker"].update(pred, labels)
            current_train_metrics = metrics_tracker["train_tracker"].compute()
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_train_metrics["acc"] * 100:.2f}%',
                'f1': f'{current_train_metrics["f1"]:.4f}',
                'auroc': f'{current_train_metrics["auroc"]:.4f}',
                'avp': f'{current_train_metrics["avp"]:.4f}',
            })
        avg_train_loss = total_train_loss / len(train_loader)
        train_results = metrics_tracker["train_tracker"].compute()

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader,
                            desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')
            for points, labels in val_pbar:
                points, labels = points.to(device), labels.to(device)
                if pointnet_version == 1:
                    # Forward PointNet1
                    pred, trans_feat = model(points)
                    loss = criterion(pred, labels, trans_feat)
                else:
                    # Forward PointNet2
                    pred = model(points)
                    loss = criterion(pred, labels)
                total_val_loss += loss.item()

                # Update metrics tracker
                metrics_tracker["val_tracker"].update(pred, labels)
                current_val_metrics = metrics_tracker["val_tracker"].compute()
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_val_metrics["acc"] * 100:.2f}%',
                    'f1': f'{current_val_metrics["f1"]:.4f}',
                    'auroc': f'{current_val_metrics["auroc"]:.4f}',
                    'avp': f'{current_val_metrics["avp"]:.4f}',
                })
        val_results = metrics_tracker["val_tracker"].compute()
        avg_val_loss = total_val_loss / len(val_loader)

        # Store loss history manually
        loss_history['train_loss'].append(avg_train_loss)
        loss_history['val_loss'].append(avg_val_loss)

        scheduler.step()


        if val_results['f1'] > best_val_f1:
            best_val_f1 = val_results['f1']
        # Save the best model based on AUROC
        if val_results['auroc'] > best_val_auroc:
            best_val_auroc = val_results['auroc']
            best_model_state = model.state_dict().copy()
        if val_results['avp'] > best_val_ap:
            best_val_ap = val_results['avp']

        # Print epoch statistics
        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {avg_train_loss:.4f} | Acc: {train_results['acc'].item():.2%}"
            f"Val Loss: {avg_val_loss:.4f}, Acc: {val_results['acc'].item():.2%} | "
            f"Val F1: {val_results['f1']:.4f} (Best: {best_val_f1:.4f}) | "
            f"Val AUC: {val_results['auroc']:.4f} (Best: {best_val_auroc:.4f}) | "
            f"Val AP: {val_results['avp']:.4f} (Best: {best_val_ap:.4f})"
        )
        # torch.cuda.empty_cache()

    # Load the best model state before returning
    model.load_state_dict(best_model_state)
    training_end = time.perf_counter()
    print(
        f"Training completed. Training time: {training_end - training_start:.2f}.",
        end=" ")
    print(f"Best Val F1: {best_val_f1:.4f}. Best Val AUROC: {best_val_auroc:.4f}. Best Val AP: {best_val_ap:.4f}.")
    best_metrics = {
        "train" : metrics_tracker["train_tracker"].best_metric(),
        "val" : metrics_tracker["val_tracker"].best_metric(),
    }
    return model, best_metrics

if __name__ == '__main__':
    pointnet_all_folds_metrics = {}
    for fold in range(10):
        model, best_metrics = train_pointnet(validation_fold=fold, num_epochs=50, pointnet_version=1)
        metrics_numpy = {
            "train": {k: v.cpu().numpy().item() for k, v in best_metrics["train"].items()},
            "val": {k: v.cpu().numpy().item() for k, v in best_metrics["val"].items()}
        }
        pointnet_all_folds_metrics[f"fold_{fold:2d}"] = metrics_numpy
        # Append each fold result to json file
        with open("pointnet1_all_folds_metrics.json", "w") as f:
            json.dump(pointnet_all_folds_metrics, f, indent=4)
