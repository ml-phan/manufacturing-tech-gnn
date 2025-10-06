from src.gnn_models import *
import datetime


def training_pipeline():
    PROCESSED_DATA_DIR = r"E:\gnn_data\pyg_data_v2_scaled"

    dataset = PyGInMemoryDataset(
        root=PROCESSED_DATA_DIR,
        pattern="*.pt",
        transform=None,
        pre_transform=None,
        pre_filter=None
    )

    model = GINECombined_v2(
        input_features=dataset[0].x.shape[1],
        global_feature_dim=dataset[0].global_features.shape[1],
        edge_features=dataset[0].edge_attr.shape[1],
        hidden_sizes=[512, 256],
        conv_dropout_rate=0.1,
        classifier_dropout_rate=0.1,
        use_layer_norm=True,
        pool_hidden_size=256
    )

    torch.cuda.empty_cache()
    model_name = model.__class__.__name__
    model_save_path = f"{model_name}.pth"
    tracker_save_path = f"{model_name}_metrics_tracker.pkl"
    if Path(model_save_path).exists():
        print(f"Loading model from {model_save_path}")
        model.load_state_dict(torch.load(model_save_path))
    else:
        print(
            f"Model file {model_save_path} does not exist. Initializing a new model.")
    # Load metrics tracker if it exists
    tracker = None
    if Path(tracker_save_path).exists():
        with open(tracker_save_path, "rb") as f:
            tracker = joblib.load(f)
    else:
        print(
            f"Metrics tracker file {tracker_save_path} does not exist. Initializing a new tracker.")
    trained_model, new_tracker = simple_train_model_v4(
        PROCESSED_DATA_DIR,
        validation_fold=0,
        model_params=None,
        gnn_model=model,
        num_epochs=100,
        batch_size=128,
        learning_rate=0.001,
        optimizer_scheduler="OneCycleLR",
        start_index=0,
        num_graphs_to_use=63000,
        metrics_tracker=tracker,
        random_state=100,
    )
    # save model and metrics tracker
    torch.save(trained_model.state_dict(), model_save_path)
    with open(tracker_save_path, "wb") as f:
        joblib.dump(new_tracker, f)


def train_gcn_simple_v1(num_epochs=50):
    PROCESSED_DATA_DIR = r"E:\gnn_data\pyg_data_v1_simple"
    dataset = FastSTEPDataset(PROCESSED_DATA_DIR, start_index=0)
    model_state_save_path = "gcn_v1_simple_model_state.pth"
    model_save_path = "gcn_v1_simple_model.pkl"
    # Load model if exists
    if Path(model_save_path).exists():
        with open(model_save_path, "rb") as f:
            model = joblib.load(f)
    else:
        model = GCN_v1_Simple(
            input_features=dataset[0].x.shape[1],
            embedding_dim=16,
            hidden_sizes=[512],
            conv_dropout_rate=0.1,
            classifier_dropout_rate=0.1,
            use_layer_norm=True,
            pool_hidden_size=128
        )
    trained_model, history = simple_train_model_v1(
        dataset_dir=PROCESSED_DATA_DIR,
        validation_fold=None,
        gnn_model=model,
        num_epochs=num_epochs,
        batch_size=4,
        learning_rate=0.001,
        optimizer_scheduler="OneCycleLR",
        start_index=0,
        num_graphs_to_use=63000,
    )
    torch.save(trained_model.state_dict(), model_state_save_path)
    with open(model_save_path, "wb") as f:
        joblib.dump(trained_model, f)


if __name__ == '__main__':
    dataset_path = Path(r"E:\gnn_data\pyg_data_v2_scaled_validation_fold_00")
    validation_fold = 0
    results_dir = Path(r"gnn_optuna_database")
    results_dir.mkdir(exist_ok=True)
    db_path = results_dir / f"optuna_gatv2_fold_{validation_fold}.db"
    fold_results, database_path = optuna_gnn(
        dataset_path, validation_fold=validation_fold,
        n_trials=5, db_path=db_path
    )
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"optuna_gatv2_results_{now}.pkl", "wb") as f:
        joblib.dump(fold_results, f)
    # study_name = "optuna_gine_fold_0"
    # storage = "sqlite:///gnn_optuna_database/optuna_gine_fold_0.db"
    #
    # study = optuna.load_study(study_name=study_name, storage=storage)
    # run_with_best_params(study)
    # train_gcn_simple_v1(50)