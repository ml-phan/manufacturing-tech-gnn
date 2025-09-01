import torch
from torch_geometric.data import InMemoryDataset
import numpy as np
from pathlib import Path
import joblib


class PyGInMemoryDataset_v2(InMemoryDataset):
    """
    InMemoryDataset that loads PyG Data objects from .pt files located in `root/raw`.
    It collates them into a single processed file at `root/processed/data.pt`.
    """

    def __init__(self, root, transform=None):
        super().__init__(root, transform)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        file_list = list(Path(self.root).glob('*.pt'))
        data_list = []
        for file in file_list:
            # Load the PyG Data object
            data = torch.load(file, weights_only=False)

            # Apply transform if provided
            if self.transform is not None:
                data = self.transform(data)

            # Append to the list
            data_list.append(data)
        self.save(data_list, self.processed_paths[0])

    def get_labels(self):
        """
        Retrieves all 'y' attributes from the data objects in the dataset.

        Returns:
            list: A list of all 'y' attributes.
        """
        all_labels = []
        for data in self:
            if hasattr(data, 'y'):
                all_labels.append(data.y.item())
        return all_labels


class GraphCustomTransform:
    def __init__(self,
                 node_power_transformer, node_minmax_scaler,
                 node_onehot_encoder,
                 edge_onehot_encoder_0, edge_onehot_encoder_1,
                 edge_minmax_scaler, global_minmax_scaler):
        self.node_power_transformer = node_power_transformer
        self.node_minmax_scaler = node_minmax_scaler
        self.node_onehot_encoder = node_onehot_encoder

        self.edge_onehot_encoder_0 = edge_onehot_encoder_0
        self.edge_onehot_encoder_1 = edge_onehot_encoder_1
        self.edge_minmax_scaler = edge_minmax_scaler

        self.global_minmax_scaler = global_minmax_scaler

    def __call__(self, data):
        data = data.clone()

        # Process node features (Data.x)
        if data.x is not None:
            x = data.x.numpy()

            # Discard x[:, 0]
            # Apply PowerTransformer + MinMaxScaler on x[:, 1:13]
            x_power_part = x[:, 1:13]
            x_power_transformed = self.node_power_transformer.transform(
                x_power_part)
            x_power_scaled = self.node_minmax_scaler.transform(
                x_power_transformed)

            # Keep x[:, 13:17] the same (assuming you meant 13:17 based on your indexing)
            x_unchanged = x[:, 13:18]

            # One-hot encode x[:, 17] (assuming 0-based indexing, so column 17 is what you called 18)
            x_onehot_part = x[:, 18:19]  # Keep as 2D
            x_onehot_encoded = self.node_onehot_encoder.transform(
                x_onehot_part)

            # Concatenate all parts
            new_x = np.concatenate([
                x_power_scaled,  # columns 1:13 -> power + minmax scaled
                x_unchanged,  # columns 13:17 -> unchanged
                x_onehot_encoded
                # column 17 -> one-hot encoded (10 categories)
            ], axis=1)

            data.x = torch.tensor(new_x, dtype=torch.float32)

        # Process edge features (Data.edge_attr)
        if data.edge_attr is not None:
            edge_attr = data.edge_attr.numpy()

            # One-hot encode edge_attr[:, 0] with 3 categories (2, 3, 4)
            edge_onehot_0 = self.edge_onehot_encoder_0.transform(
                edge_attr[:, 0:1])

            # One-hot encode edge_attr[:, 1] with 7 categories
            edge_onehot_1 = self.edge_onehot_encoder_1.transform(
                edge_attr[:, 1:2])

            # Apply PowerTransformer + MinMaxScaler on edge_attr[:, 2:4]
            edge_power_part = edge_attr[:, 2:4]
            edge_log_transformed = np.log1p(edge_power_part)
            edge_scaled = self.edge_minmax_scaler.transform(
                edge_log_transformed)

            # Keep edge_attr[:, 4:6] the same
            edge_unchanged = edge_attr[:, 4:6]

            # Concatenate all parts
            new_edge_attr = np.concatenate([
                edge_onehot_0, # column 0 -> one-hot encoded (3 categories)
                edge_onehot_1, # column 1 -> one-hot encoded (7 categories)
                edge_scaled,  # columns 2:4 -> power + minmax scaled
                edge_unchanged  # columns 4:6 -> unchanged
            ], axis=1)

            data.edge_attr = torch.tensor(new_edge_attr,
                                          dtype=torch.float32)

        # Process global features (Data.global_features)
        if data.global_features is not None:
            global_features = data.global_features.numpy()

            # Apply PowerTransformer + MinMaxScaler on all global features
            global_log_transformed = np.log1p(global_features)
            global_scaled = self.global_minmax_scaler.transform(
                global_log_transformed)

            data.global_features = torch.tensor(global_scaled,
                                                dtype=torch.float32)

        return data

# Exampl useage
if __name__ == '__main__':

    with open(Path("path_to_saved_scalers.pkl"), "rb") as f:
        scalers = joblib.load(f)
    transform = GraphCustomTransform(**scalers)

    dataset_path = Path("path_to_dataset")
    dataset = PyGInMemoryDataset_v2(
        root=str(dataset_path),
        transform=transform,
    )