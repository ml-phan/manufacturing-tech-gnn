import datetime
import json

import dill
import gc
import joblib
import numpy as np
import os
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from torch_geometric.data import InMemoryDataset


from collections import Counter
import optuna
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, GINConv, GlobalAttention, \
    GINEConv
from torch_geometric.nn.aggr import AttentionalAggregation
from torch.utils.data import Subset
from torchmetrics import Accuracy, F1Score, AUROC, \
    Recall, AveragePrecision, MetricCollection, MetricTracker
from tqdm import tqdm
from sklearn.preprocessing import PowerTransformer, \
        MinMaxScaler, OneHotEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_dataset_stats(dataset):
    labels = dataset.get_labels()
    label_counts = Counter(dataset.get_labels())
    print("Label counts in dataset:")
    for label, count in label_counts.items():
        print(f"Label {label}: {count} instances")
    # Calculate the percentage of each label
    total_count = sum(label_counts.values())
    for label, count in label_counts.items():
        percentage = (count / total_count) * 100
        print(f"Label {label}: {percentage:.2f}% of total instances")
    class_weights = [len(labels) / count for label, count in
                     label_counts.items()]
    print("Class weights for loss function:", class_weights)


class FastSTEPDataset:
    """
    Fast dataset that loads pre-processed PyTorch Geometric data
    """

    def __init__(self, processed_data_dir, start_index=None, end_index=None):
        """
        processed_data_dir: directory containing processed .pt files and mapping.pkl
        """
        self.processed_data_dir = processed_data_dir
        self.start_index = start_index
        self.end_index = end_index

        # Load the mapping file
        mapping_path = os.path.join(processed_data_dir, 'dataset_mapping.pkl')
        with open(mapping_path, 'rb') as f:
            self.mapping = pickle.load(f)

        self.processed_files = self.mapping['processed_files']

        print(f"Fast dataset loaded:")
        print(f"- Total samples: {len(self.processed_files)}")
        print(f"- Processed successfully: {self.mapping['total_processed']}")
        print(f"- Failed processing: {self.mapping['total_failed']}")

    def __len__(self):
        return len(self.processed_files)

    def __getitem__(self, index):
        """Load pre-processed data - much faster than converting from GraphML"""
        file_info = self.processed_files[index]
        processed_path = file_info['processed_path']

        # Load the pre-processed PyTorch Geometric data
        data = torch.load(processed_path, weights_only=False)
        if self.start_index is not None and hasattr(data,
                                                    'x') and data.x is not None:
            data = data.clone()
            data.x = data.x[:, self.start_index:self.end_index]
        return data

    def get_labels(self):
        """Get all labels for stratified splitting"""
        return [pf['label'] for pf in self.processed_files]


class PyGInMemoryDataset(InMemoryDataset):
    """
    Custom InMemoryDataset that loads PyTorch Geometric Data objects from .pt files.
    """

    def __init__(
            self,
            root,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            pattern: str = "*.pt"
    ):
        self.pattern = pattern
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # raw_dir = os.path.join(self.root, 'raw')
        """Returns list of raw file names in the raw directory"""
        return [f for f in os.listdir(self.root) if f.endswith('.pt')]

    @property
    def processed_file_names(self):
        """Returns list of processed file names"""
        return ['data.pt']

    def download(self):
        """Download raw data (not needed if files already exist)"""
        pass

    def process(self):
        """Process raw data and save to processed directory"""
        # Look for .pt files in the root directory
        data_list = [f for f in os.listdir(self.root) if f.endswith('.pt')]
        if self.pre_filter is not None:
            data_list = [data for data in self.data_list if
                         self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in
                         self.data_list]

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


class DynamicGNN(nn.Module):
    """
    Dynamic Graph Neural Network for binary classification with configurable hidden layers
    """

    def __init__(self, input_features, hidden_sizes=None,
                 conv_dropout_rate=0.3, classifier_dropout_rate=0.4,
                 use_layer_norm=True, pool_hidden_size=128):
        """
        Args:
            input_features: Number of input node features
            hidden_sizes: List of hidden layer sizes. Length determines number of hidden layers.
                         e.g., [256, 128, 64] creates 3 hidden layers with these sizes
            conv_dropout_rate: Dropout rate for convolution layers
            classifier_dropout_rate: Dropout rate for classifier
            use_layer_norm: Whether to use layer normalization
            pool_hidden_size: Hidden size for the attention pooling gate network
        """
        super(DynamicGNN, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 256]
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one element")

        self.num_layers = len(hidden_sizes)
        self.hidden_sizes = hidden_sizes
        self.use_layer_norm = use_layer_norm

        # Create convolution layers dynamically
        self.convs = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None

        # First layer: input_features -> hidden_sizes[0]
        self.convs.append(GCNConv(input_features, hidden_sizes[0]))
        self.dropouts.append(nn.Dropout(p=conv_dropout_rate))
        if use_layer_norm:
            self.layer_norms.append(nn.LayerNorm(hidden_sizes[0]))

        # Additional layers: hidden_sizes[i-1] -> hidden_sizes[i]
        for i in range(1, self.num_layers):
            self.convs.append(GCNConv(hidden_sizes[i - 1], hidden_sizes[i]))
            self.dropouts.append(nn.Dropout(p=conv_dropout_rate))
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_sizes[i]))

        # Global attention pooling
        self.pool = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(hidden_sizes[-1], pool_hidden_size),
            nn.ReLU(),
            nn.Linear(pool_hidden_size, 1)
        ))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_sizes[-1], pool_hidden_size),
            nn.ReLU(),
            nn.Dropout(p=classifier_dropout_rate),
            nn.Linear(pool_hidden_size, 2)
        )

        self._print_model_info(input_features, conv_dropout_rate,
                               classifier_dropout_rate)

    def _print_model_info(self, input_features, conv_dropout_rate,
                          classifier_dropout_rate):
        """Print model configuration"""
        print(f"Created Dynamic GNN model:")
        print(f"- Input features: {input_features}")
        print(f"- Number of hidden layers: {self.num_layers}")
        print(f"- Hidden layer sizes: {self.hidden_sizes}")
        print(f"- Output classes: 2")
        print(f"- Convolution dropout rate: {conv_dropout_rate}")
        print(f"- Classifier dropout rate: {classifier_dropout_rate}")
        print(f"- Layer normalization: {self.use_layer_norm}")

    def forward(self, x, edge_index, batch):
        """
        Forward pass through the network
        x: node features
        edge_index: graph edges
        batch: which graph each node belongs to (for batching multiple graphs)
        """

        # Pass through all convolution layers
        for i, (conv, dropout) in enumerate(zip(self.convs, self.dropouts)):
            x = conv(x, edge_index)

            # Apply layer normalization if enabled
            if self.use_layer_norm:
                x = self.layer_norms[i](x)

            x = F.relu(x)
            x = dropout(x)

        # Pool node features to get one vector per graph
        x = self.pool(x, batch)

        # Final classification
        x = self.classifier(x)

        return x


class DynamicGNN_Embedding_GlobalFeatures(nn.Module):
    """
    Dynamic Graph Neural Network for binary classification with configurable hidden layers
    """

    def __init__(self, input_features, global_feature_dim, embedding_dim=16,
                 hidden_sizes=None, conv_dropout_rate=0.1,
                 classifier_dropout_rate=0.1, use_layer_norm=True,
                 pool_hidden_size=128):
        """
        Args:
            input_features: Number of input node features
            hidden_sizes: List of hidden layer sizes. Length determines number of hidden layers.
                         e.g., [256, 128, 64] creates 3 hidden layers with these sizes
            conv_dropout_rate: Dropout rate for convolution layers
            classifier_dropout_rate: Dropout rate for classifier
            use_layer_norm: Whether to use layer normalization
            pool_hidden_size: Hidden size for the attention pooling gate network
        """
        super(DynamicGNN_Embedding_GlobalFeatures, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 256]
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one element")

        self.num_layers = len(hidden_sizes)
        self.hidden_sizes = hidden_sizes
        self.embedding = nn.Embedding(400, embedding_dim)
        self.use_layer_norm = use_layer_norm
        self.total_input_features = (input_features - 1) + embedding_dim

        # Create convolution layers dynamically
        self.convs = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None

        # First layer: input_features -> hidden_sizes[0]
        self.convs.append(GCNConv(self.total_input_features, hidden_sizes[0]))
        self.dropouts.append(nn.Dropout(p=conv_dropout_rate))
        if use_layer_norm:
            self.layer_norms.append(nn.LayerNorm(hidden_sizes[0]))

        # Additional layers: hidden_sizes[i-1] -> hidden_sizes[i]
        for i in range(1, self.num_layers):
            self.convs.append(GCNConv(hidden_sizes[i - 1], hidden_sizes[i]))
            self.dropouts.append(nn.Dropout(p=conv_dropout_rate))
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_sizes[i]))

        # Global attention pooling:
        self.pool = AttentionalAggregation(gate_nn=nn.Sequential(
            nn.Linear(hidden_sizes[-1], pool_hidden_size),
            nn.ReLU(),
            nn.Linear(pool_hidden_size, 1)
        ))

        # Classifier: Concatenates node features with global features
        self.classifier = nn.Sequential(
            nn.Linear(hidden_sizes[-1] + global_feature_dim, pool_hidden_size),
            nn.ReLU(),
            nn.Dropout(p=classifier_dropout_rate),
            nn.Linear(pool_hidden_size, 2)
        )

        self._print_model_info(input_features, conv_dropout_rate,
                               classifier_dropout_rate)

    def _print_model_info(self, input_features, conv_dropout_rate,
                          classifier_dropout_rate):
        """Print model configuration"""
        print(f"Created Dynamic GNN model:")
        print(f"- Input features: {input_features}")
        print(f"- Embedding dimension: {self.embedding.embedding_dim}")
        print(f"- Number of hidden layers: {self.num_layers}")
        print(f"- Hidden layer sizes: {self.hidden_sizes}")
        print(f"- Output classes: 2")
        print(f"- Convolution dropout rate: {conv_dropout_rate}")
        print(f"- Classifier dropout rate: {classifier_dropout_rate}")
        print(f"- Layer normalization: {self.use_layer_norm}")

    def forward(self, x, edge_index, batch, global_features):
        """
        Forward pass through the network
        x: node features
        edge_index: graph edges
        batch: which graph each node belongs to (for batching multiple graphs)
        """
        # Split node type and node features
        node_types = x[:, 0].long()
        node_features = x[:, 1:]

        # Get embeddings for node types
        node_type_embeddings = self.embedding(node_types)
        # Concatenate embeddings with original features
        x = torch.cat((node_features, node_type_embeddings), dim=1)

        # Pass through all convolution layers
        for i, (conv, dropout) in enumerate(zip(self.convs, self.dropouts)):
            x = conv(x, edge_index)

            # Apply layer normalization if enabled
            if self.use_layer_norm:
                x = self.layer_norms[i](x)

            x = F.relu(x)
            x = dropout(x)

        # Pool node features to get one vector per graph
        pooled_x = self.pool(x, batch)

        # Concatenate classifier input with global features
        combined_features = torch.cat((pooled_x, global_features), dim=1)
        # Final classification
        output = self.classifier(combined_features)

        return output


class GATConvCombined(nn.Module):
    """
    Dynamic Graph Neural Network for binary classification with configurable hidden layers
    """

    def __init__(self, input_features, hidden_sizes=None, heads=4,
                 conv_dropout_rate=0.2, classifier_dropout_rate=0.2,
                 use_layer_norm=True, pool_hidden_size=128):
        """
        Args:
            input_features: Number of input node features
            hidden_sizes: List of hidden layer sizes. Length determines number of hidden layers.
                         e.g., [256, 128, 64] creates 3 hidden layers with these sizes
            conv_dropout_rate: Dropout rate for convolution layers
            classifier_dropout_rate: Dropout rate for classifier
            use_layer_norm: Whether to use layer normalization
            pool_hidden_size: Hidden size for the attention pooling gate network
        """
        super(GATConvCombined, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 256]
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one element")

        self.num_layers = len(hidden_sizes)
        self.hidden_sizes = hidden_sizes
        self.use_layer_norm = use_layer_norm
        self.heads = heads
        # Create convolution layers dynamically
        self.convs = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None

        # First layer: input_features -> hidden_sizes[0]
        self.convs.append(
            GATConv(
                input_features,
                hidden_sizes[0],
                heads=self.heads,
                dropout=conv_dropout_rate
            ))
        if use_layer_norm:
            self.layer_norms.append(nn.LayerNorm(hidden_sizes[0] * self.heads))

        # Additional layers: hidden_sizes[i-1] -> hidden_sizes[i]
        for i in range(1, self.num_layers):
            self.convs.append(
                GATConv(
                    hidden_sizes[i - 1] * self.heads,
                    hidden_sizes[i],
                    heads=self.heads,
                    dropout=conv_dropout_rate
                ))
            if use_layer_norm:
                self.layer_norms.append(
                    nn.LayerNorm(hidden_sizes[i] * self.heads))

        # Global attention pooling
        self.pool = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(hidden_sizes[-1] * self.heads, pool_hidden_size),
            nn.ReLU(),
            nn.Linear(pool_hidden_size, 1)
        ))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_sizes[-1] * self.heads, pool_hidden_size),
            nn.ReLU(),
            nn.Dropout(p=classifier_dropout_rate),
            nn.Linear(pool_hidden_size, 2)
        )

        self._print_model_info(input_features, conv_dropout_rate,
                               classifier_dropout_rate)

    def _print_model_info(self, input_features, conv_dropout_rate,
                          classifier_dropout_rate):
        """Print model configuration"""
        print(f"Created Dynamic GAT model:")
        print(f"- Input features: {input_features}")
        print(f"- Number of hidden layers: {self.num_layers}")
        print(f"- Hidden layer sizes: {self.hidden_sizes}")
        print(f"- Attention heads: {self.heads}")
        print(f"- Output classes: 2")
        print(f"- Convolution dropout rate: {conv_dropout_rate}")
        print(f"- Classifier dropout rate: {classifier_dropout_rate}")
        print(f"- Layer normalization: {self.use_layer_norm}")

    def forward(self, x, edge_index, batch):
        """
        Forward pass through the network
        x: node features
        edge_index: graph edges
        batch: which graph each node belongs to (for batching multiple graphs)
        """

        # Pass through all convolution layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            # Apply layer normalization if enabled
            if self.use_layer_norm:
                x = self.layer_norms[i](x)

            x = F.relu(x)

        # Pool node features to get one vector per graph
        x = self.pool(x, batch)

        # Final classification
        x = self.classifier(x)

        return x


class GINCombined(nn.Module):
    """
    Dynamic Graph Neural Network for binary classification with configurable hidden layers
    """

    def __init__(self, input_features, global_feature_dim,
                 embedding_dim=16, hidden_sizes=None,
                 conv_dropout_rate=0.2, classifier_dropout_rate=0.2,
                 use_layer_norm=True, pool_hidden_size=128):
        """
        Args:
            input_features: Number of input node features
            hidden_sizes: List of hidden layer sizes. Length determines number of hidden layers.
                         e.g., [256, 128, 64] creates 3 hidden layers with these sizes
            conv_dropout_rate: Dropout rate for convolution layers
            classifier_dropout_rate: Dropout rate for classifier
            use_layer_norm: Whether to use layer normalization
            pool_hidden_size: Hidden size for the attention pooling gate network
        """
        super(GINCombined, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 256]
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one element")

        self.num_layers = len(hidden_sizes)
        self.hidden_sizes = hidden_sizes
        self.embedding = nn.Embedding(400, embedding_dim)
        self.use_layer_norm = use_layer_norm
        self.total_input_features = (input_features - 1) + embedding_dim

        # Create convolution layers dynamically
        self.convs = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None

        # First layer: input_features -> hidden_sizes[0]
        mlp = nn.Sequential(
            nn.Linear(self.total_input_features, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[0])
        )
        self.convs.append(GINConv(mlp))
        self.dropouts.append(nn.Dropout(p=conv_dropout_rate))
        if use_layer_norm:
            self.layer_norms.append(nn.LayerNorm(hidden_sizes[0]))

        # Additional layers: hidden_sizes[i-1] -> hidden_sizes[i]
        for i in range(1, self.num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[i], hidden_sizes[i])
            )
            self.convs.append(GINConv(mlp))
            self.dropouts.append(nn.Dropout(p=conv_dropout_rate))
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_sizes[i]))

        # Global attention pooling
        self.pool = AttentionalAggregation(gate_nn=nn.Sequential(
            nn.Linear(hidden_sizes[-1], pool_hidden_size),
            nn.ReLU(),
            nn.Linear(pool_hidden_size, 1)
        ))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_sizes[-1] + global_feature_dim, pool_hidden_size),
            nn.ReLU(),
            nn.Dropout(p=classifier_dropout_rate),
            nn.Linear(pool_hidden_size, 2)
        )

        self._print_model_info(input_features, conv_dropout_rate,
                               classifier_dropout_rate)

    def _print_model_info(self, input_features, conv_dropout_rate,
                          classifier_dropout_rate):
        """Print model configuration"""
        print(f"Created Dynamic GIN model:")
        print(f"- Input features: {input_features}")
        print(f"- Number of hidden layers: {self.num_layers}")
        print(f"- Hidden layer sizes: {self.hidden_sizes}")
        print(f"- Output classes: 2")
        print(f"- Convolution dropout rate: {conv_dropout_rate}")
        print(f"- Classifier dropout rate: {classifier_dropout_rate}")
        print(f"- Layer normalization: {self.use_layer_norm}")

    def forward(self, x, edge_index, batch, global_features):
        """
        Forward pass through the network
        x: node features
        edge_index: graph edges
        batch: which graph each node belongs to (for batching multiple graphs)
        """
        node_types = x[:, 0].long()
        node_features = x[:, 1:]

        # Get embeddings for node types
        node_type_embeddings = self.embedding(node_types)
        # Concatenate embeddings with original features
        x = torch.cat((node_features, node_type_embeddings), dim=1)

        # Pass through all convolution layers
        for i, (conv, dropout) in enumerate(zip(self.convs, self.dropouts)):

            #
            x_ = conv(x, edge_index)

            # Apply layer normalization if enabled
            if self.use_layer_norm:
                x_ = self.layer_norms[i](x_)

            # TODO: Also try with Leaky ReLU
            x_ = F.relu(x_)
            x_ = dropout(x_)

            # Add residual connection
            # TODO: Need to test this
            x = x + x_

        # Pool node features to get one vector per graph
        pooled_x = self.pool(x, batch)

        # Concatenate classifier input with global features
        combined_features = torch.cat((pooled_x, global_features), dim=1)
        # Final classification
        output = self.classifier(combined_features)

        return output


class GINCombinedv2(nn.Module):
    """
    Dynamic Graph Neural Network for binary classification with configurable hidden layers
    """

    def __init__(self, input_features, global_feature_dim,
                 hidden_sizes=None, conv_dropout_rate=0.2,
                 classifier_dropout_rate=0.2, use_layer_norm=True,
                 pool_hidden_size=128):
        """
        Args:
            input_features: Number of input node features
            hidden_sizes: List of hidden layer sizes. Length determines number of hidden layers.
                         e.g., [256, 128, 64] creates 3 hidden layers with these sizes
            conv_dropout_rate: Dropout rate for convolution layers
            classifier_dropout_rate: Dropout rate for classifier
            use_layer_norm: Whether to use layer normalization
            pool_hidden_size: Hidden size for the attention pooling gate network
        """
        super(GINCombinedv2, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 256]
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one element")

        self.num_layers = len(hidden_sizes)
        self.hidden_sizes = hidden_sizes
        self.use_layer_norm = use_layer_norm
        self.input_features = input_features
        self.conv_dropout_rate = conv_dropout_rate
        self.classifier_dropout_rate = classifier_dropout_rate
        # Create convolution layers dynamically
        self.convs = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None

        # First layer: input_features -> hidden_sizes[0]
        mlp = nn.Sequential(
            nn.Linear(input_features, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[0])
        )
        self.convs.append(GINConv(mlp))
        self.dropouts.append(nn.Dropout(p=conv_dropout_rate))
        if use_layer_norm:
            self.layer_norms.append(nn.LayerNorm(hidden_sizes[0]))

        # Additional layers: hidden_sizes[i-1] -> hidden_sizes[i]
        for i in range(1, self.num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[i], hidden_sizes[i])
            )
            self.convs.append(GINConv(mlp))
            self.dropouts.append(nn.Dropout(p=conv_dropout_rate))
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_sizes[i]))

        # Global attention pooling
        self.pool = AttentionalAggregation(gate_nn=nn.Sequential(
            nn.Linear(hidden_sizes[-1], pool_hidden_size),
            nn.ReLU(),
            nn.Linear(pool_hidden_size, 1)
        ))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_sizes[-1] + global_feature_dim, pool_hidden_size),
            nn.ReLU(),
            nn.Dropout(p=classifier_dropout_rate),
            nn.Linear(pool_hidden_size, 2)
        )

        self._print_model_info(input_features, conv_dropout_rate,
                               classifier_dropout_rate)

    def _print_model_info(self, input_features, conv_dropout_rate,
                          classifier_dropout_rate):
        """Print model configuration"""
        print(f"Created Dynamic GIN model:")
        print(f"- Input features: {input_features}")
        print(f"- Number of hidden layers: {self.num_layers}")
        print(f"- Hidden layer sizes: {self.hidden_sizes}")
        print(f"- Output classes: 2")
        print(f"- Convolution dropout rate: {conv_dropout_rate}")
        print(f"- Classifier dropout rate: {classifier_dropout_rate}")
        print(f"- Layer normalization: {self.use_layer_norm}")

    def forward(self, x, edge_index, edge_attr, batch, global_features):

        """
        Forward pass through the network
        x: node features
        edge_index: graph edges
        batch: which graph each node belongs to (for batching multiple graphs)
        """

        # Pass through all convolution layers
        for i, (conv, dropout) in enumerate(zip(self.convs, self.dropouts)):
            x = conv(x, edge_index)

            # Apply layer normalization if enabled
            if self.use_layer_norm:
                x = self.layer_norms[i](x)

            x = F.relu(x)
            x = dropout(x)

        # Pool node features to get one vector per graph
        pooled_x = self.pool(x, batch)

        # Concatenate classifier input with global features
        combined_features = torch.cat((pooled_x, global_features), dim=1)
        # Final classification
        output = self.classifier(combined_features)

        return output


class GINECombined_v2(nn.Module):
    """
    Dynamic Graph Neural Network for binary classification with configurable hidden layers
    """

    def __init__(self, input_features, edge_features, global_feature_dim,
                 hidden_sizes=None, conv_dropout_rate=0.2,
                 classifier_dropout_rate=0.2,
                 use_layer_norm=True, pool_hidden_size=128):
        """
        Args:
            input_features: Number of input node features
            hidden_sizes: List of hidden layer sizes. Length determines number of hidden layers.
                         e.g., [256, 128, 64] creates 3 hidden layers with these sizes
            conv_dropout_rate: Dropout rate for convolution layers
            classifier_dropout_rate: Dropout rate for classifier
            use_layer_norm: Whether to use layer normalization
            pool_hidden_size: Hidden size for the attention pooling gate network
        """
        super(GINECombined_v2, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 256]
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one element")

        self.num_layers = len(hidden_sizes)
        self.hidden_sizes = hidden_sizes
        self.use_layer_norm = use_layer_norm
        self.edge_features = edge_features

        # Create convolution layers dynamically
        self.convs = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None

        # First layer: input_features -> hidden_sizes[0]
        mlp = nn.Sequential(
            nn.Linear(input_features, hidden_sizes[0]),
            nn.GELU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[0])
        )

        self.convs.append(
            GINEConv(mlp, edge_dim=edge_features, train_eps=True)
        )
        self.dropouts.append(nn.Dropout(p=conv_dropout_rate))
        if use_layer_norm:
            self.layer_norms.append(nn.LayerNorm(hidden_sizes[0]))

        # Additional layers: hidden_sizes[i-1] -> hidden_sizes[i]
        for i in range(1, self.num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]),
                nn.GELU(),
                nn.Linear(hidden_sizes[i], hidden_sizes[i])
            )
            self.convs.append(
                GINEConv(mlp, edge_dim=edge_features, train_eps=True)
            )
            self.dropouts.append(nn.Dropout(p=conv_dropout_rate))
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_sizes[i]))

        # Global attention pooling
        self.pool = AttentionalAggregation(gate_nn=nn.Sequential(
            nn.Linear(hidden_sizes[-1], pool_hidden_size),
            nn.GELU(),
            nn.Linear(pool_hidden_size, 1)
        ))

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_sizes[-1] + global_feature_dim),
            nn.Linear(hidden_sizes[-1] + global_feature_dim, 512),
            nn.GELU(),
            nn.Dropout(p=classifier_dropout_rate),
            nn.Linear(512, pool_hidden_size),
            nn.GELU(),
            nn.Dropout(p=classifier_dropout_rate),
            nn.Linear(pool_hidden_size, 2)
        )

        self._print_model_info(
            input_features, edge_features, conv_dropout_rate,
            classifier_dropout_rate, pool_hidden_size
        )

    def _print_model_info(self, input_features, edge_features,
                          conv_dropout_rate, classifier_dropout_rate,
                          pool_hidden_size):
        """Print model configuration"""
        print(f"Created GINE model:")
        print(f"- Input features: {input_features}")
        print(f"- Input edge features: {edge_features}")
        print(f"- Number of hidden layers: {self.num_layers}")
        print(f"- Hidden layer sizes: {self.hidden_sizes}")
        print(f"- Output classes: 2")
        print(f"- Convolution dropout rate: {conv_dropout_rate}")
        print(f"- Classifier dropout rate: {classifier_dropout_rate}")
        print(f"- Pooling hidden layer size: {pool_hidden_size}")
        print(f"- Layer normalization: {self.use_layer_norm}")

    def forward(self, x, edge_index, edge_attr, batch, global_features):
        """
        Forward pass through the network
        x: node features
        edge_index: graph edges
        batch: which graph each node belongs to (for batching multiple graphs)
        """
        x_ = None
        # Pass through all convolution layers
        for i, (conv, dropout) in enumerate(zip(self.convs, self.dropouts)):
            x = conv(x, edge_index, edge_attr)

            # Apply layer normalization if enabled
            if self.use_layer_norm:
                x = self.layer_norms[i](x)

            x = F.gelu(x)
            x = dropout(x)

            # # Residual connection
            # x = x + x_
        # Pool node features to get one vector per graph
        pooled_x = self.pool(x, batch)

        # Concatenate classifier input with global features
        combined_features = torch.cat((pooled_x, global_features), dim=1)

        # Final classification
        output = self.classifier(combined_features)

        return output


def scaler_creation(dataset_dir, validation_fold):

    # First, collect all training data to fit scalers
    all_train_data = []
    for fold_idx in range(10):
        if fold_idx != validation_fold:
            dataset_path = Path(dataset_dir) / f"fold_{str(fold_idx).zfill(2)}"
            if dataset_path.exists():
                dataset = PyGInMemoryDataset_v2(root=str(dataset_path),
                                                transform=None)
                all_train_data.extend([data for data in dataset])

    # Initialize all transformers
    # For node features (Data.x)
    node_power_transformer = PowerTransformer()
    node_minmax_scaler = MinMaxScaler()
    node_onehot_encoder = OneHotEncoder(categories=[list(range(10))],
                                        sparse_output=False)

    # For edge features (Data.edge_attr)
    edge_onehot_encoder_0 = OneHotEncoder(categories=[[2, 3, 4]],
                                          sparse_output=False)
    edge_onehot_encoder_1 = OneHotEncoder(categories=[list(range(7))],
                                          sparse_output=False)
    edge_minmax_scaler = MinMaxScaler()

    # For global features
    global_minmax_scaler = MinMaxScaler()

    # Collect features for fitting from training data only
    print("Collecting training features for fitting transformers...")

    # Node features
    train_node_power_features = []  # x[:, 1:13]
    train_node_onehot_features = []  # x[:, 18]

    # Edge features
    train_edge_onehot_0_features = []  # edge_attr[:, 0]
    train_edge_onehot_1_features = []  # edge_attr[:, 1]
    train_edge_power_features = []  # edge_attr[:, 2:4]

    # Global features
    train_global_features = []

    for data in all_train_data:
        # Node features
        train_node_power_features.append(data.x[:, 1:13])
        train_node_onehot_features.append(data.x[:, 18:19])  # Keep as 2D

        # Edge features
        train_edge_onehot_0_features.append(
            data.edge_attr[:, 0:1])  # Keep as 2D
        train_edge_onehot_1_features.append(
            data.edge_attr[:, 1:2])  # Keep as 2D
        train_edge_power_features.append(data.edge_attr[:, 2:4])

        # Global features
        train_global_features.append(data.global_features)

    # Concatenate all training features
    train_node_power_concat = torch.cat(train_node_power_features,
                                        dim=0).numpy()
    train_node_onehot_concat = torch.cat(train_node_onehot_features,
                                         dim=0).numpy()

    train_edge_onehot_0_concat = torch.cat(train_edge_onehot_0_features,
                                           dim=0).numpy()
    train_edge_onehot_1_concat = torch.cat(train_edge_onehot_1_features,
                                           dim=0).numpy()
    train_edge_power_concat = torch.cat(train_edge_power_features,
                                        dim=0).numpy()

    train_global_concat = torch.cat(train_global_features, dim=0).numpy()

    print("Fitting transformers on training data...")

    # Fit node transformers
    node_power_transformed = node_power_transformer.fit_transform(
        train_node_power_concat)
    node_minmax_scaler.fit(node_power_transformed)
    node_onehot_encoder.fit(train_node_onehot_concat)

    # Fit edge transformers
    edge_onehot_encoder_0.fit(train_edge_onehot_0_concat)
    edge_onehot_encoder_1.fit(train_edge_onehot_1_concat)
    edge_power_transformed = np.log1p(train_edge_power_concat)
    edge_minmax_scaler.fit(edge_power_transformed)

    # Fit global transformers
    global_power_transformed = np.log1p(train_global_concat)
    global_minmax_scaler.fit(global_power_transformed)

    # Return all scalers as dictionary
    return {
        'node_power_transformer': node_power_transformer,
        'node_minmax_scaler': node_minmax_scaler,
        'node_onehot_encoder': node_onehot_encoder,
        'edge_onehot_encoder_0': edge_onehot_encoder_0,
        'edge_onehot_encoder_1': edge_onehot_encoder_1,
        'edge_minmax_scaler': edge_minmax_scaler,
        'global_minmax_scaler': global_minmax_scaler
    }


def dataset_scaling(dataset_dir, validation_fold):
    # Create the transform instance
    with open(Path(dataset_dir) / f"scalers_fold_{str(validation_fold).zfill(2)}.pkl",
            "rb") as f:
        scalers = joblib.load(f)
    transform = GraphCustomTransform(**scalers)

    print("Loading datasets with custom transform...")

    # Now load datasets with custom scaling
    all_data_set_folds = []
    for fold in tqdm(range(10), desc="Generating folds"):
        dataset_path = Path(dataset_dir) / f"fold_{str(fold).zfill(2)}"
        if dataset_path.exists():
            dataset = PyGInMemoryDataset_v2(
                root=str(dataset_path),
                transform=transform,
            )
            all_data_set_folds.append(dataset)

    # Create train/val split as before
    train_folds = all_data_set_folds[:validation_fold] + all_data_set_folds[
                                                         validation_fold + 1:]
    # train_dataset = ConcatDataset(train_folds)
    val_dataset = all_data_set_folds[validation_fold]

    print("Dataset preprocessing complete!")
    return train_folds, val_dataset, transform


def simple_train_model_v2(
        dataset,
        gnn_model,
        num_epochs=10,
        batch_size=2,
        learning_rate=0.001,
        start_index=None,
        num_graphs_to_use=None,
):
    """
    Simple training function with progress tracking
    """
    num_graphs_to_use = min(num_graphs_to_use, len(dataset) - start_index)
    # Split dataset into train/validation
    training_start = time.perf_counter()
    all_labels = dataset.get_labels()
    labels = all_labels[start_index:start_index + num_graphs_to_use]
    indices = list(range(start_index, start_index + num_graphs_to_use))
    label_counts = Counter(labels)
    for label, count in label_counts.items():
        print(f"Label {label}: {count} instances")
    total_count = sum(label_counts.values())
    for label, count in label_counts.items():
        percentage = (count / total_count) * 100
        print(f"Label {label}: {percentage:.2f}% of total instances")
    # Calculating class weights
    class_weights = [len(labels) / count for label, count in
                     label_counts.items()]
    class_weights.reverse()
    class_weights = torch.tensor(class_weights,
                                 dtype=torch.float).to(device)
    print(f"Class weights: {class_weights}")

    print("Splitting dataset into train and validation sets")
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    print(f"Train samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")

    # Create train and validation using Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=8)

    # Create model
    model = gnn_model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        # weight_decay=1e-5
    )

    print(f"\nStarting training for {num_epochs} epochs...")

    # Progress tracking
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch': []
    }

    best_val_acc = 0.0
    best_model_state = None

    # Training loop
    for epoch in range(num_epochs):
        # print("Training epoch:", epoch + 1)
        # Training phase
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0

        # Progress bar for training batches
        train_pbar = tqdm(train_loader,
                          desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')

        for batch in train_pbar:
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            batch = batch.to(device)
            # print(f"x shape: {batch.x.shape}")
            # print(f"edge_index shape: {batch.edge_index.shape}")
            # print(f"batch shape: {batch.batch.shape}")
            # print(f"global_features shape: {batch.global_features.shape}")
            outputs = model(
                batch.x, batch.edge_index,
                batch.batch, batch.global_features
            )
            loss = criterion(outputs.squeeze(), batch.y.squeeze())
            # Dimension check for outputs batch

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            total_train += batch.y.size(0)
            correct_train += (predicted == batch.y).sum().item()

            # Update progress bar
            current_acc = 100 * correct_train / total_train
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.2f}%'
            })

        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader,
                            desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')

            for batch in val_pbar:
                batch = batch.to(device)
                outputs = model(
                    batch.x, batch.edge_index,
                    batch.batch, batch.global_features
                )
                loss = criterion(outputs, batch.y)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += batch.y.size(0)
                correct_val += (predicted == batch.y).sum().item()

                # Update progress bar
                current_val_acc = 100 * correct_val / total_val
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_val_acc:.2f}%'
                })

        # Calculate epoch statistics
        train_acc = 100 * correct_train / total_train
        val_acc = 100 * correct_val / total_val
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Store progress
        training_history['epoch'].append(epoch + 1)
        training_history['train_loss'].append(avg_train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(avg_val_loss)
        training_history['val_acc'].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        # Print epoch summary
        print(
            f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}'
            f', Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f}, '
            f'Acc: {val_acc:.2f}% (Best Val: {best_val_acc:.2f}%)')
        # print(f'Epoch {epoch + 1}/{num_epochs} Summary:')
        # print(
        #     f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        # print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        # print(f'  Best Val Acc: {best_val_acc:.2f}%')
        # print('-' * 50)

        # Clear CUDA cache to free memory incase of leaks
        torch.cuda.empty_cache()

    # Load best model
    model.load_state_dict(best_model_state)
    training_end = time.perf_counter()
    print("Training completed!")
    print(f"Training time: {training_end - training_start}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    return model, training_history


def simple_train_model_v4(
        dataset_dir,
        validation_fold,
        model_params=None,
        gnn_model=None,
        num_epochs=10,
        batch_size=2,
        learning_rate=0.001,
        optimizer_scheduler="ReduceLROnPlateau",
        start_index=0,
        num_graphs_to_use=None,
        metrics_tracker=None,
        random_state=None,
        trial_progress=r"1/1",
):
    """
    Simple training function with progress tracking
    """
    # Split dataset into train/validation
    training_start = time.perf_counter()

    if validation_fold is None:
        dataset = PyGInMemoryDataset_v2(
            root=str(dataset_dir))
        all_labels = dataset.get_labels()
        num_graphs_to_use = min(num_graphs_to_use, len(dataset) - start_index)
        labels = all_labels[start_index:start_index + num_graphs_to_use]
        indices = list(range(start_index, start_index + num_graphs_to_use))
        label_counts = Counter(labels)
        num_classes = len(label_counts)
        total_count = sum(label_counts.values())
        for label, count in label_counts.items():
            percentage = (count / total_count) * 100
            print(f"Label {label}: {count} instances ({percentage:.2f}%)")
        # Calculating class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )

        # Convert to dictionary format
        class_weights = torch.FloatTensor(
            [class_weights[0], class_weights[1]]).to(device)

        train_indices, val_indices = train_test_split(
            indices,
            test_size=0.2,
            stratify=labels,
            random_state=random_state,
        )

        print(f"Train samples: {len(train_indices)}")
        print(f"Validation samples: {len(val_indices)}")
        print(f"Random state: {random_state}")

        # Create train and validation using Subset
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)


    else:
        all_data_set_folds = []
        for fold in tqdm(range(10), desc="Generating folds"):
            dataset_path = Path(
                dataset_dir) / f"fold_{str(fold).zfill(2)}"
            if dataset_path.exists():
                dataset = PyGInMemoryDataset_v2(
                    root=str(dataset_path),
                    transform=None,
                )
                all_data_set_folds.append(dataset)
        train_folds = all_data_set_folds[
                      :validation_fold] + all_data_set_folds[
                                          validation_fold + 1:]
        # train_folds, val_dataset, transform = dataset_scaling(
        #     dataset_dir, validation_fold,
        #     scalers=None
        # )
        train_dataset = ConcatDataset(train_folds)
        # print(f"Using fold {validation_fold} for validation")
        val_dataset = all_data_set_folds[validation_fold]
        train_labels = [
            inner_item
            for item in train_folds
            for inner_item in item.get_labels()]
        val_labels = val_dataset.get_labels()
        train_label_counts = Counter(train_labels)
        val_label_counts = Counter(val_labels)
        total_count = sum(train_label_counts.values())
        for label, count in train_label_counts.items():
            percentage = (count / total_count) * 100
            print(f"Label {label}: {count} instances ({percentage:.2f}%)")
        num_classes = len(train_label_counts)

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )

        # Convert to dictionary format
        class_weights = torch.FloatTensor(
            [class_weights[0], class_weights[1]]).to(device)

    print(f"Class weights: {class_weights}")
    # Scale the dataset
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    if model_params:
        model = GINECombined_v2(
            input_features=train_dataset[0].x.shape[1],
            global_feature_dim=train_dataset[0].global_features.shape[1],
            edge_features=train_dataset[0].edge_attr.shape[1],
            **model_params
        ).to(device)
    elif gnn_model:
        model = gnn_model.to(device)
    else:
        print("Either model_params or gnn_model must be provided")
        return None, None
    # Loss function and optimizer
    # TODO: Focal Loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )
    print(f"Optimizer: {optimizer.__class__.__name__}", end=". ")
    print(f"Learning rate: {learning_rate}", end=". ")
    # Create MetricCollection for all metrics
    # metrics = MetricCollection({
    #     'accuracy': Accuracy(task="multiclass", num_classes=num_classes),
    #     'f1': F1Score(task="multiclass", num_classes=num_classes,
    #                   average="weighted"),
    #     'precision': Precision(task="multiclass", num_classes=num_classes,
    #                            average="weighted"),
    #     'recall': Recall(task="multiclass", num_classes=num_classes,
    #                      average="weighted"),
    #     'auroc': AUROC(task="multiclass", num_classes=num_classes)
    # }).to(device)
    # train_metrics = metrics.clone(prefix='train_')
    # val_metrics = metrics.clone(prefix='val_')

    if metrics_tracker is None:
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
    else:
        metrics_tracker["train_tracker"].to(device)
        metrics_tracker["val_tracker"].to(device)

    if optimizer_scheduler == "OneCycleLR":
        # OneCycleLR scheduler
        total_steps = num_epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate * 10,  # Peak learning rate (10x base rate)
            total_steps=total_steps,  # Total number of batch steps
            pct_start=0.3,  # 30% of training spent increasing LR
        )
    else:
        # ReduceLROnPlateau scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # Maximize auroc
            factor=0.7,
            patience=10,
            threshold=0.001,
            min_lr=1e-6,
        )

    print(f"LR scheduler: {scheduler.__class__.__name__}")

    loss_history = {"train_loss": [], "val_loss": []}

    best_val_f1 = 0.0
    best_auroc = 0.0
    best_model_state = None

    print(f"\nStarting training for {num_epochs} epochs...")

    # Training loop
    for epoch in range(num_epochs):

        # Training phase
        model.train()
        total_train_loss = 0
        metrics_tracker["train_tracker"].increment()
        metrics_tracker["val_tracker"].increment()
        # Progress bar for training batches
        train_pbar = tqdm(train_loader,
                          desc=f'Epoch {epoch + 1}/{num_epochs} Trial {trial_progress} [Train]')

        # Train loop
        for batch in train_pbar:
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            # print(f"x shape: {batch.x.shape}")
            # print(f"edge_index shape: {batch.edge_index.shape}")
            # print(f"batch shape: {batch.batch.shape}")
            # print(f"global_features shape: {batch.global_features.shape}")
            batch = batch.to(device)
            outputs = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
                global_features=batch.global_features
            )
            loss = criterion(outputs.squeeze(), batch.y.squeeze())

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            # Statistics
            total_train_loss += loss.item()

            # Update all metrics
            # train_metrics.update(outputs, batch.y)
            metrics_tracker["train_tracker"].update(outputs, batch.y)

            # Update progress bar
            # current_metrics = train_metrics.compute()
            current_metrics2 = metrics_tracker["train_tracker"].compute()
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                # 'acc': f'{current_metrics["train_accuracy"] * 100:.2f}%',
                'acc2': f'{current_metrics2["acc"] * 100:.2f}%',
                # 'f1': f'{current_metrics["train_f1"]:.4f}',
                'f1_2': f'{current_metrics2["f1"]:.4f}',
                # 'auroc': f'{current_metrics["train_auroc"]:.4f}',
                'auroc2': f'{current_metrics2["auroc"]:.4f}',
            })
        # Compute final accuracy for the epoch and then reset
        train_results = metrics_tracker["train_tracker"].compute()
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        toal_val_loss = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader,
                            desc=f'Epoch {epoch + 1}/{num_epochs} Trial {trial_progress} [Val]')

            for batch in val_pbar:
                batch = batch.to(device)
                outputs = model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch,
                    global_features=batch.global_features
                )
                loss = criterion(outputs, batch.y)

                toal_val_loss += loss.item()
                # val_metrics.update(outputs, batch.y)
                metrics_tracker["val_tracker"].update(outputs, batch.y)

                # Update progress bar
                # current_val_metrics = val_metrics.compute()
                current_val_metrics2 = metrics_tracker["val_tracker"].compute()

                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    # 'acc': f'{current_val_metrics["val_accuracy"] * 100:.2f}%',
                    'acc2': f'{current_val_metrics2["acc"] * 100:.2f}%',
                    # 'f1': f'{current_val_metrics["val_f1"]:.4f}',
                    'f1_2': f'{current_val_metrics2["f1"]:.4f}',
                    # 'auroc': f'{current_val_metrics["val_auroc"]:.4f}',
                    'auroc2': f'{current_val_metrics2["auroc"]:.4f}',
                })

        # Compute final accuracy for the epoch and then reset
        val_results = metrics_tracker["val_tracker"].compute()
        avg_val_loss = toal_val_loss / len(val_loader)

        # Store loss history manually
        loss_history['train_loss'].append(avg_train_loss)
        loss_history['val_loss'].append(avg_val_loss)

        # Update the scheduler with validation auroc
        if optimizer_scheduler == "ReduceLROnPlateau":
            scheduler.step(val_results['auroc'])
        else:
            scheduler.step()

        # Update metric tracker
        # epoch_metrics = {**train_results, **val_results}
        # metric_tracker.log(epoch_metrics)

        # Save best model
        if val_results["auroc"] > best_auroc:
            best_auroc = val_results["auroc"]
            best_model_state = model.state_dict().copy()

        # Save best F1 score
        if val_results["f1"] > best_val_f1:
            best_val_f1 = val_results["f1"]

        # Print epoch summary
        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, Acc: {train_results['acc'].item():.2%} | "
            f"Val Loss: {avg_val_loss:.4f}, Acc: {val_results['acc'].item():.2%} | "
            f"Val F1: {val_results['f1']:.4f} (Best: {best_val_f1:.4f}) | "
            f"Val AUC: {val_results['auroc']:.4f} (Best: {best_auroc:.4f})"
        )

        # Reset all metrics Clear CUDA cache to free memory in case of leaks
        # No reset when using MetricTracker
        # metrics_tracker["train_tracker"].reset()
        # metrics_tracker["val_tracker"].reset()
        torch.cuda.empty_cache()

    # Load best model
    model.load_state_dict(best_model_state)
    training_end = time.perf_counter()
    print(f"Training completed. Training time: {training_end - training_start:.2f}.", end=" ")
    print(f"Best Val F1: {best_val_f1:.4f}. Best Val AUROC: {best_auroc:.4f}")

    return model, metrics_tracker


def optuna_gnn(dataset_path, validation_fold=0, n_trials=1, db_path=None):
    start_time = time.perf_counter()
    tracker_path = Path(r"E:\gnn_data\optuna_tracker_v2")

    if tracker_path.exists():
        with open(tracker_path, 'rb') as f:
            tracker = pickle.load(f)
    else:
        tracker = None

    fold_results = {}

    results_dir = Path(r"gnn_optuna_database")
    results_dir.mkdir(exist_ok=True)
    if db_path is None:
        db_path = results_dir / f"optuna_gine_fold_{validation_fold}.db"

    study = optuna.create_study(
        study_name=f"optuna_gine_fold_{validation_fold}",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=100)
    )
    existing_trials = len(study.get_trials())
    def objective(trial):
        try:
            num_layers = trial.suggest_int("num_layers", 1, 3)
            hidden_size = trial.suggest_int('hidden_size', 128, 512)

            model_params = {
                "hidden_sizes": [hidden_size] * num_layers,
                "conv_dropout_rate": trial.suggest_float("conv_dropout_rate", 0.01,
                                                         0.5),
                "classifier_dropout_rate": trial.suggest_float(
                    "classifier_dropout_rate", 0.01, 0.5),
                "use_layer_norm": trial.suggest_categorical(
                    "use_layer_norm", [True, False]),
                "pool_hidden_size": trial.suggest_int("pool_hidden_size", 128, 1024)
            }
            print(model_params)
            train_function_params = {
                "num_epochs": trial.suggest_int("num_epochs", 100, 300),
                "batch_size": trial.suggest_int("batch_size", 32, 128),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-5, 1e-2, log=True),
                "optimizer_scheduler": trial.suggest_categorical(
                    "optimizer_scheduler", ["OneCycleLR", "ReduceLROnPlateau"])
            }
            print(train_function_params)
            total_trials = n_trials + existing_trials
            trial_progress = rf"{trial.number}/{total_trials}"
            trained_model, new_tracker = simple_train_model_v4(
                dataset_dir=dataset_path,
                validation_fold=validation_fold,
                model_params=model_params,
                gnn_model=None,
                **train_function_params,
                metrics_tracker=tracker,
                trial_progress=trial_progress,
            )
            best_val_metric = new_tracker["val_tracker"].best_metric()
            best_auroc = best_val_metric["auroc"]
            # Save relevant metrics into file
            fold_results[f"fold_{validation_fold}_trial_{trial.number}"] = {
                'model_params': model_params,
                'train_function_params': train_function_params,
                'best_metric': best_val_metric,
            }
            # Append results to json file
            results_path = Path(
                fr"optuna_gine_fold_{validation_fold}_results.json")
            if results_path.exists():
                with open(results_path, 'r') as f:
                    all_results = json.load(f)
            else:
                all_results = {}
            all_results[f"trial_{trial.number}"] = fold_results[f"fold_{validation_fold}_trial_{trial.number}"]
            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=4)


            gc.collect()
            return best_auroc
        except Exception as e:
            print(f"Error during trial: {e}")
            # Check if trial has attribute number then print it out
            if hasattr(trial, 'number'):
                print(f"Trial number: {trial.number}")
            return 0.0

    study.optimize(objective, n_trials=n_trials, n_jobs=1,
                   show_progress_bar=False)

    fold_results[f"fold_{validation_fold}"] = {
        'best_params': study.best_params,
        'best_score': study.best_value,
        'study': study
    }
    end_time = time.perf_counter()
    print(f"Study time: {end_time - start_time:.2f}")
    return fold_results, db_path


# Run with new data using best parameters
def run_with_best_params(study):

    # Get best parameters
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

    fold_results = {}

    for fold in range(10):

        # Train with new data
        print(f"Training fold {fold} with best parameters...")
        dataset_path = Path(
            rf"E:\gnn_data\pyg_data_v2_scaled_validation_fold_{str(fold).zfill(2)}")
        # model = GINECombined_v2(
        #     input_features=dataset[0].x.shape[1],
        #     global_feature_dim=dataset[0].global_features.shape[1],
        #     edge_features=dataset[0].edge_attr.shape[1],
        #     **model_params
        # )

        # trained_model, tracker = simple_train_model_v4(
        #     dataset_dir=dataset_path,
        #     validation_fold=fold,
        #     gnn_model=model,
        #     **train_function_params,
        #     metrics_tracker=None
        # )
        trained_model, new_tracker = simple_train_model_v4(
            dataset_dir=dataset_path,
            validation_fold=fold,
            model_params=model_params,
            gnn_model=None,
            **train_function_params,
            metrics_tracker=None,
        )
        # Save model to file
        model_name = fr"model_states\optuna_gine_modelstate_fold_{fold}.pth"
        torch.save(trained_model.state_dict(), model_name)

        # Save model object to pickle
        model_obj_name = fr"model_states\optuna_gine_model_fold_{fold}.pkl"
        with open(model_obj_name, 'wb') as f:
            joblib.dump(trained_model, f)

        print(f"Model saved to {model_name}")
        # Save fold results
        results = {
            'train_tracker': new_tracker["train_tracker"].best_metric(),
            'val_tracker': new_tracker["val_tracker"].best_metric(),
        }
        # save results to json
        results_path = Path(
            fr"optuna_gine_all_fold_results.json")
        if results_path.exists():
            with open(results_path, 'r') as f:
                fold_results = json.load(f)
        else:
            fold_results = {}
        fold_results[f"fold_{fold}"] = results
        with open(results_path, 'w') as f:
            json.dump(fold_results, f, indent=4)

    return fold_results