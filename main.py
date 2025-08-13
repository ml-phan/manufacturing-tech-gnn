from src.gnn_models import *

if __name__ == '__main__':
    PROCESSED_DATA_DIR = r"E:\gnn_data\pyg_data_v2_scaled"

    dataset2 = PyGInMemoryDataset(
        root=PROCESSED_DATA_DIR,
        pattern="*.pt",
        transform=None,
        pre_transform=None,
        pre_filter=None
    )

    model = GINECombined_v2(
        input_features=dataset2[0].x.shape[1],
        global_feature_dim=dataset2[0].global_features.shape[1],
        edge_features=dataset2[0].edge_attr.shape[1],
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
        dataset2,
        gnn_model=model,
        num_epochs=50,
        batch_size=128,
        learning_rate=0.001,
        start_index=0,
        num_graphs_to_use=63000,
        metrics_tracker=tracker,
        random_state=100,
    )
    # save model and metrics tracker
    torch.save(trained_model.state_dict(), model_save_path)
    with open(tracker_save_path, "wb") as f:
        joblib.dump(new_tracker, f)
