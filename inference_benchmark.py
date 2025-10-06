import time

from src.gnn_models import *
# from src.pointnet2_models import *
from training_pointnet import *
from xgboost import XGBClassifier


def measure_inference_time(model, data_loader, device="cuda", warmup=10,
                           runs=100):
    model.to(device)
    model.eval()

    # Warm-up to stabilize CUDA kernel timings
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            _ = model(data.to(device))
        if i >= warmup:
            break

    times = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            start = time.perf_counter()
            _ = model(data.to(device))
            torch.cuda.synchronize()  # important on GPU
            end = time.perf_counter()
            times.append(end - start)
            if i >= runs:
                break

    avg_time = sum(times) / len(times)
    print(f"Average inference time per batch: {avg_time * 1000:.2f} ms")
    print(f"Average per sample: {(avg_time / len(data.y)) * 1000:.2f} ms")


def benchmark_gine(gnn_type="gine"):
    if gnn_type.lower() == "gine":
        study_name = "optuna_gine_fold_0"
        storage = "sqlite:///gnn_optuna_database/optuna_gine_fold_0.db"
        study = optuna.load_study(study_name=study_name, storage=storage)
    else:
        study_name = "optuna_gatv2_fold_0"
        storage = "sqlite:///gnn_optuna_database/optuna_gatv2_fold_0.db"
        study = optuna.load_study(study_name=study_name, storage=storage)

    best_params = study.best_params
    print("Best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    # Extract parameters
    model_params = {
        "hidden_sizes": [best_params["hidden_size"]] * best_params[
            "num_layers"],
        "conv_dropout_rate": best_params["conv_dropout_rate"],
        "classifier_dropout_rate": best_params["classifier_dropout_rate"],
        "use_layer_norm": best_params["use_layer_norm"],
        "pool_hidden_size": best_params["pool_hidden_size"]
    }

    train_function_params = {
        "num_epochs": best_params["num_epochs"],
        "batch_size": best_params["batch_size"],
        "learning_rate": best_params["learning_rate"],
        "optimizer_scheduler": best_params["optimizer_scheduler"]
    }

    if gnn_type.lower() != "gine":
        model_params["num_heads"] = best_params["num_head"]

    dataset_dir = r"E:\gnn_data\pyg_data_v2_scaled_validation_fold_00\fold_00"
    train_dataset = PyGInMemoryDataset_v2(
        root=str(dataset_dir))
    if gnn_type.lower() == "gine":
        model = GINECombined_v2(
            input_features=train_dataset[0].x.shape[1],
            global_feature_dim=train_dataset[0].global_features.shape[1],
            edge_features=train_dataset[0].edge_attr.shape[1],
            **model_params
        ).to(device)
    else:
        model = GATCombined_v2(
            input_features=train_dataset[0].x.shape[1],
            global_feature_dim=train_dataset[0].global_features.shape[1],
            edge_features=train_dataset[0].edge_attr.shape[1],
            **model_params
        ).to(device)

    train_loader = DataLoader(train_dataset, batch_size=4,
                              shuffle=True)
    model.eval()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for batch in tqdm(train_loader):
            batch = batch.to(device)
            _ = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
                global_features=batch.global_features
            )
    torch.cuda.synchronize()
    end = time.perf_counter()
    avg_time = (end - start) / len(train_dataset)

    return avg_time

def benchmark_gcn():
    PROCESSED_DATA_DIR = r"E:\gnn_data\pyg_data_v1_simple"
    dataset = FastSTEPDataset(PROCESSED_DATA_DIR, start_index=0)
    get_dataset_stats(dataset)
    model = GCN_v1_Simple(
        input_features=dataset[0].x.shape[1],
        embedding_dim=16,
        hidden_sizes=[512],
        conv_dropout_rate=0.1,
        classifier_dropout_rate=0.1,
        use_layer_norm=True,
        pool_hidden_size=128
    ).to(device)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch = batch.to(device)
            _ = model(
                x=batch.x,
                edge_index=batch.edge_index,
                batch=batch.batch,
            )
    torch.cuda.synchronize()
    end = time.perf_counter()
    avg_time = (end - start) / len(dataset)
    return avg_time


def benchmark_pointnet(pointnet_version="1"):
    train_dataset, val_dataset = build_dataset(validation_fold=0)
    train_loader = DataLoader(train_dataset, batch_size=4,
                              shuffle=True, drop_last=True,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True)
    if pointnet_version == 1:
        model = get_model().to(device)
        print(model.__class__.__name__)
    else:
        # PointNet2 model version 2
        model = PointNet2Classification(num_classes=2, input_channels=3).to(device)
        print(model.__class__.__name__)

    train_pbar = tqdm(train_loader)
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        for points, _ in train_pbar:
            points = points.to(device)
            _ = model(points)

    torch.cuda.synchronize()
    end = time.perf_counter()
    avg_time = (end - start) / len(train_dataset)
    return avg_time


def benchmark_xgboost():
    best_params = {'n_estimators': 229,
                   'max_depth': 10,
                   'learning_rate': 0.23010627398694375,
                   'min_child_weight': 5,
                   'subsample': 0.8515134589080573,
                   'colsample_bytree': 0.7674160634634193,
                   'reg_alpha': 3.8657297966420128,
                   'reg_lambda': 0.6141160901516102}
    model = XGBClassifier(**best_params)
    features = [
        "faces", "edges", "vertices", "quantity",
        "height", "width", "depth", "volume", "area",
        "bbox_height", "bbox_width", "bbox_depth", "bbox_volume",
        "bbox_area",
    ]
    data = pd.read_csv("./data/synced_dataset_final.csv")
    X = data[features]
    y = data["is_cnc"]
    model.fit(X, y)
    length = len(data)
    start = time.perf_counter_ns()
    for i in range(100):
        _ = model.predict(X)
    end = time.perf_counter_ns()
    avg_time = (end - start) / length / 100
    return avg_time


if __name__ == '__main__':

    # infer_time = benchmark_gine(gnn_type="gine")
    # infer_time = benchmark_gcn()
    infer_time = benchmark_pointnet(pointnet_version="2")
    # infer_time = benchmark_xgboost()
    print(f"Average inference time per batch: {infer_time :.2f} nanoseconds")
