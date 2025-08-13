import joblib
import networkx as nx
import os
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset
import glob

from collections import Counter
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, GINConv, GlobalAttention, \
    GINEConv
from torch_geometric.data import Data
from torch_geometric.nn.aggr import AttentionalAggregation
from torch.utils.data import Subset
from torchmetrics import Accuracy, F1Score, AUROC, Precision, \
    Recall, AveragePrecision, MetricCollection, MetricTracker
from tqdm import tqdm

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

    Args:
        root (str): Root directory containing the .pt files
        transform (callable, optional): A function/transform that takes in a Data object
            and returns a transformed version
        pre_transform (callable, optional): A function/transform that takes in a Data object
            and returns a transformed version (applied before caching)
        pre_filter (callable, optional): A function that takes in a Data object and
            returns a boolean indicating whether the data should be included
        pattern (str): File pattern to match (default: "*.pt")
    """

    def __init__(
            self,
            root: str,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            pattern: str = "*.pt"
    ):
        self.pattern = pattern
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        self.data_list = []

    @property
    def raw_file_names(self):
        """Returns list of raw file names in the raw directory"""
        # raw_dir = os.path.join(self.root, 'raw')
        # if not os.path.exists(raw_dir):
        #     return []
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

        if self.pre_filter is not None:
            self.data_list = [data for data in self.data_list if
                              self.pre_filter(data)]

        if self.pre_transform is not None:
            self.data_list = [self.pre_transform(data) for data in
                              self.data_list]

        self.save(self.data_list, self.processed_paths[0])

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
            input_features, conv_dropout_rate, edge_features,
            classifier_dropout_rate
        )

    def _print_model_info(self, input_features, edge_features,
                          conv_dropout_rate, classifier_dropout_rate):
        """Print model configuration"""
        print(f"Created GINE model:")
        print(f"- Input features: {input_features}")
        print(f"- Input edge features: {edge_features}")
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


def simple_train_model_v3(
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
    num_classes = len(label_counts)
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
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = gnn_model.to(device)

    # Loss function and optimizer
    # TODO: Focal Loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        # weight_decay=1e-5
    )

    # Create MetricCollection for all metrics
    metrics = MetricCollection({
        'accuracy': Accuracy(task="multiclass", num_classes=num_classes),
        'f1': F1Score(task="multiclass", num_classes=num_classes,
                      average="weighted"),
        'precision': Precision(task="multiclass", num_classes=num_classes,
                               average="weighted"),
        'recall': Recall(task="multiclass", num_classes=num_classes,
                         average="weighted"),
        'auroc': AUROC(task="multiclass", num_classes=num_classes)
    }).to(device)
    train_metrics = metrics.clone(prefix='train_')
    val_metrics = metrics.clone(prefix='val_')

    metric_tracker = MetricTracker(
        MetricCollection(
            {
                **train_metrics,
                **val_metrics,
                "train_loss": nn.Module(),
                "val_loss": nn.Module(),
            }
        )
    )
    # OneCycleLR scheduler
    # total_steps = num_epochs * len(train_loader)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=0.005,  # Peak learning rate (10x base rate)
    #     total_steps=total_steps,  # Total number of batch steps
    #     pct_start=0.3,  # 30% of training spent increasing LR
    # )

    # ReduceLROnPlateau scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Maximize F1 score
        factor=0.7,
        patience=7,
        threshold=0.001,
        min_lr=1e-6,
    )

    # Initialize all the metrics
    train_accuracy_metric = Accuracy(task="multiclass",
                                     num_classes=2).to(device)
    val_accuracy_metric = Accuracy(task="multiclass",
                                   num_classes=2).to(device)

    train_f1_metric = F1Score(task="multiclass", num_classes=num_classes,
                              average="weighted").to(device)
    val_f1_metric = F1Score(task="multiclass", num_classes=num_classes,
                            average="weighted").to(device)

    train_precision_metric = Precision(task="multiclass",
                                       num_classes=num_classes,
                                       average="weighted").to(device)
    val_precision_metric = Precision(task="multiclass",
                                     num_classes=num_classes,
                                     average="weighted").to(device)

    train_recall_metric = Recall(task="multiclass",
                                 num_classes=num_classes,
                                 average="weighted").to(device)
    val_recall_metric = Recall(task="multiclass", num_classes=num_classes,
                               average="weighted").to(device)

    train_auroc_metric = AUROC(task="multiclass",
                               num_classes=num_classes).to(device)
    val_auroc_metric = AUROC(task="multiclass",
                             num_classes=num_classes).to(device)
    # Progress tracking
    training_history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'train_precision': [], 'train_recall': [], 'train_auroc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_precision': [],
        'val_recall': [], 'val_auroc': [],
        'epoch': []
    }

    loss_history = {"train_loss": [], "val_loss": []}

    best_val_f1 = 0.0
    best_auroc = 0.0
    best_model_state = None

    print(f"\nStarting training for {num_epochs} epochs...")

    # Training loop
    for epoch in range(num_epochs):
        # print("Training epoch:", epoch + 1)
        # Training phase
        model.train()
        total_loss = 0

        # Progress bar for training batches
        train_pbar = tqdm(train_loader,
                          desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')

        # Store all metrics in a list for easier management
        train_metrics = [
            train_accuracy_metric,
            train_f1_metric,
            train_precision_metric,
            train_recall_metric,
            train_auroc_metric
        ]
        val_metrics = [
            val_accuracy_metric,
            val_f1_metric,
            val_precision_metric,
            val_recall_metric,
            val_auroc_metric
        ]
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
                batch.x, batch.edge_index,
                batch.batch, batch.global_features
            )
            loss = criterion(outputs.squeeze(), batch.y.squeeze())

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            # Update the scheduler

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            # Update all metrics
            for metric in train_metrics:
                metric.update(outputs, batch.y)

            # Update progress bar
            current_acc = train_accuracy_metric.compute().item() * 100
            current_f1 = train_f1_metric.compute().item()
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.2f}%',
                'f1': f'{current_f1:.4f}',
            })
        # Compute final accuracy for the epoch and then reset
        train_metrics_results = {
            'acc': train_accuracy_metric.compute().item(),
            'f1': train_f1_metric.compute().item(),
            'precision': train_precision_metric.compute().item(),
            'recall': train_recall_metric.compute().item(),
            'auroc': train_auroc_metric.compute().item(),
        }
        for metric in train_metrics:
            metric.reset()

        # Validation phase
        model.eval()
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
                for metric in val_metrics:
                    metric.update(outputs, batch.y)
                scheduler.step(val_loss)
                # Update progress bar
                current_val_acc = val_accuracy_metric.compute().item() * 100
                current_val_f1 = val_f1_metric.compute().item()
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_val_acc:.2f}%',
                    'f1': f'{current_val_f1:.4f}',
                })

        # Compute final accuracy for the epoch and then reset
        val_metrics_results = {
            'acc': val_accuracy_metric.compute().item(),
            'f1': val_f1_metric.compute().item(),
            'precision': val_precision_metric.compute().item(),
            'recall': val_recall_metric.compute().item(),
            'auroc': val_auroc_metric.compute().item(),
        }
        for metric in val_metrics:
            metric.reset()

        # Calculate epoch statistics
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Store progress
        training_history['epoch'].append(epoch + 1)
        training_history['train_loss'].append(avg_train_loss)
        training_history['train_acc'].append(
            train_metrics_results['acc'] * 100)
        training_history['train_f1'].append(train_metrics_results['f1'])
        training_history['train_precision'].append(
            train_metrics_results['precision'])
        training_history['train_recall'].append(
            train_metrics_results['recall'])
        training_history['train_auroc'].append(
            train_metrics_results['auroc'])

        training_history['val_loss'].append(avg_val_loss)
        training_history['val_acc'].append(val_metrics_results['acc'] * 100)
        training_history['val_f1'].append(val_metrics_results['f1'])
        training_history['val_precision'].append(
            val_metrics_results['precision'])
        training_history['val_recall'].append(
            val_metrics_results['recall'])
        training_history['val_auroc'].append(
            val_metrics_results['auroc'])

        # Save best model
        if val_metrics_results["f1"] > best_val_f1:
            best_val_f1 = val_metrics_results["f1"]
            best_model_state = model.state_dict().copy()

        # Print epoch summary
        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, Acc: {train_metrics_results['acc']:.2%} | "
            f"Val Loss: {avg_val_loss:.4f} | Acc: {val_metrics_results['acc']:.2%} | "
            f"Val F1: {val_metrics_results['f1']:.4f} (Best: {best_val_f1:.4f}) | "
            f"Val AUC: {val_metrics_results['auroc']:.4f}"
        )
        # print(f'Epoch {epoch + 1}/{num_epochs} Summary:')
        # print(
        #     f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        # print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        # print(f'  Best Val Acc: {best_val_acc:.2f}%')
        # print('-' * 50)

        # Clear CUDA cache to free memory in case of leaks
        torch.cuda.empty_cache()

    # Load best model
    model.load_state_dict(best_model_state)
    training_end = time.perf_counter()
    print("Training completed!")
    print(f"Training time: {training_end - training_start:.2f}")
    print(f"Best validation f1: {best_val_f1:.4f}")

    return model, training_history


def simple_train_model_v4(
        dataset,
        gnn_model,
        num_epochs=10,
        batch_size=2,
        learning_rate=0.001,
        start_index=None,
        num_graphs_to_use=None,
        metrics_tracker=None,
        random_state=None,
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
    num_classes = len(label_counts)
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
        random_state=random_state,
    )

    print(f"Train samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    print(f"Random state: {random_state}")

    # Create train and validation using Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = gnn_model.to(device)

    # Loss function and optimizer
    # TODO: Focal Loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        # weight_decay=1e-5
    )
    print(f"Using optimizer: {optimizer.__class__.__name__}")
    # Create MetricCollection for all metrics
    metrics = MetricCollection({
        'accuracy': Accuracy(task="multiclass", num_classes=num_classes),
        'f1': F1Score(task="multiclass", num_classes=num_classes,
                      average="weighted"),
        'precision': Precision(task="multiclass", num_classes=num_classes,
                               average="weighted"),
        'recall': Recall(task="multiclass", num_classes=num_classes,
                         average="weighted"),
        'auroc': AUROC(task="multiclass", num_classes=num_classes)
    }).to(device)
    train_metrics = metrics.clone(prefix='train_')
    val_metrics = metrics.clone(prefix='val_')

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
                            average="weighted"),
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
                            average="weighted"),
                        "auroc": AUROC(
                            task="multiclass", num_classes=num_classes)
                    }
                )).to(device)}
    else:
        metrics_tracker["train_tracker"].to(device)
        metrics_tracker["val_tracker"].to(device)

    # OneCycleLR scheduler
    # total_steps = num_epochs * len(train_loader)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=0.005,  # Peak learning rate (10x base rate)
    #     total_steps=total_steps,  # Total number of batch steps
    #     pct_start=0.3,  # 30% of training spent increasing LR
    # )

    # ReduceLROnPlateau scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Maximize F1 score
        factor=0.7,
        patience=7,
        threshold=0.001,
        min_lr=1e-6,
    )
    print(f"Using LR scheduler: {scheduler.__class__.__name__}")

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
        # metric_tracker.increment()
        # train_metrics_tracker.increment()
        metrics_tracker["train_tracker"].increment()
        metrics_tracker["val_tracker"].increment()
        # Progress bar for training batches
        train_pbar = tqdm(train_loader,
                          desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')

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
            train_metrics.update(outputs, batch.y)
            metrics_tracker["train_tracker"].update(outputs, batch.y)

            # Update progress bar
            current_metrics = train_metrics.compute()
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_metrics["train_accuracy"] * 100:.2f}%',
                'f1': f'{current_metrics["train_f1"]:.4f}',
                'auroc': f'{current_metrics["train_auroc"]:.4f}',
            })
        # Compute final accuracy for the epoch and then reset
        train_results = train_metrics.compute()
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        toal_val_loss = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader,
                            desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')

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
                val_metrics.update(outputs, batch.y)
                metrics_tracker["val_tracker"].update(outputs, batch.y)

                # Update progress bar
                current_val_metrics = val_metrics.compute()

                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_val_metrics["val_accuracy"] * 100:.2f}%',
                    'f1': f'{current_val_metrics["val_f1"]:.4f}',
                    'auroc': f'{current_val_metrics["val_auroc"]:.4f}',
                })

        # Compute final accuracy for the epoch and then reset
        val_results = val_metrics.compute()
        avg_val_loss = toal_val_loss / len(val_loader)

        # Store loss history manually
        loss_history['train_loss'].append(avg_train_loss)
        loss_history['val_loss'].append(avg_val_loss)

        # Update the scheduler with validation auroc
        scheduler.step(val_results['val_auroc'])

        # Update metric tracker
        epoch_metrics = {**train_results, **val_results}
        # metric_tracker.log(epoch_metrics)

        # Save best model
        if val_results["val_auroc"] > best_auroc:
            best_auroc = val_results["val_auroc"]
            best_model_state = model.state_dict().copy()

        # Save best F1 score
        if val_results["val_f1"] > best_val_f1:
            best_val_f1 = val_results["val_f1"]

        # Print epoch summary
        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, Acc: {train_results['train_accuracy'].item():.2%} | "
            f"Val Loss: {avg_val_loss:.4f}, Acc: {val_results['val_accuracy'].item():.2%} | "
            f"Val F1: {val_results['val_f1']:.4f} (Best: {best_val_f1:.4f}) | "
            f"Val AUC: {val_results['val_auroc']:.4f} (Best: {best_auroc:.4f})"
        )

        # Reset all metrics Clear CUDA cache to free memory in case of leaks
        train_metrics.reset()
        val_metrics.reset()
        torch.cuda.empty_cache()

    # Load best model
    model.load_state_dict(best_model_state)
    training_end = time.perf_counter()
    print("Training completed!")
    print(f"Training time: {training_end - training_start:.2f}")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Best validation AUROC: {best_auroc:.4f}")

    # Get complete training history from tracker
    training_history = None
    # training_history = metric_tracker.compute_all()

    # Add loss history
    # for key, values in loss_history.items():
    #     training_history[key] = torch.tensor(values)
    #
    # # Add epoch numbers
    # training_history['epoch'] = torch.arange(1, num_epochs + 1)

    return model, metrics_tracker
