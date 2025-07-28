import joblib
import networkx as nx
import os
import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import Counter
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, GINConv, GlobalAttention
from torch_geometric.data import Data
from torch_geometric.nn.aggr import AttentionalAggregation
from torch.utils.data import Subset
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


def simple_convert_graph(networkx_graph, label):
    """
    Convert a NetworkX graph to PyTorch Geometric Data object
    This is a simple version - we'll improve it later
    """

    # Step 3a: Create a mapping from node names to numbers
    # PyTorch Geometric needs nodes to be numbered 0, 1, 2, ...
    nodes = list(networkx_graph.nodes())
    node_to_index = {node: i for i, node in enumerate(nodes)}

    print(f"Graph has {len(nodes)} nodes")

    # Step 3b: Create simple node features
    # For now, let's just use the node degree (how many connections each node has)
    node_features = []
    for node in nodes:
        degree = networkx_graph.degree(node)  # Number of connections
        # For now, our feature is just [degree]. Later we'll add more features.
        node_features.append([float(degree)])

    # Convert to PyTorch tensor
    x = torch.tensor(node_features, dtype=torch.float)
    print(f"Node features shape: {x.shape}")

    # Step 3c: Create edge list
    edge_list = []
    for edge in networkx_graph.edges():
        source, target = edge
        # Convert node names to indices
        source_idx = node_to_index[source]
        target_idx = node_to_index[target]

        # Add both directions (undirected graph)
        edge_list.append([source_idx, target_idx])
        edge_list.append([target_idx, source_idx])

    # Convert to PyTorch tensor and transpose
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    print(f"Edge index shape: {edge_index.shape}")

    # Step 3d: Create label tensor
    y = torch.tensor([label], dtype=torch.long)

    # Step 3e: Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, y=y)

    return data


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


class SimpleSTEPDataset:
    """
    Simple dataset class that loads GraphML files and converts them
    """

    def __init__(self, dataframe):
        """
        dataframe: your pandas dataframe with 'graphml_file' and 'is_cnc' columns
        """
        self.df = dataframe.copy()
        print(f"Dataset created with {len(self.df)} samples")

    def __len__(self):
        """Return number of samples"""
        return len(self.df)

    def __getitem__(self, index):
        """Get one sample by index"""
        # Get file path and label
        file_path = self.df.iloc[index]['graphml_file']
        label = self.df.iloc[index]['is_cnc']

        # Load the GraphML file
        G = nx.read_graphml(file_path)

        # Convert to PyTorch Geometric format
        data = simple_convert_graph(G, label)

        return data


class SimpleGNN(nn.Module):
    """
    Simple Graph Neural Network for binary classification
    """

    def __init__(self, input_features, hidden_size=256,
                 conv_dropout_rate=0.6, classifier_dropout_rate=0.4,
                 norm=None

                 ):
        super(SimpleGNN, self).__init__()

        # Two graph convolution layers
        self.conv1 = GCNConv(input_features, hidden_size)
        # self.bn1 = nn.BatchNorm1d(hidden_size_1)
        self.dropout_conv1 = nn.Dropout(p=conv_dropout_rate)
        self.norm = nn.LayerNorm(hidden_size)

        self.conv2 = GCNConv(hidden_size, hidden_size)
        # self.bn2 = nn.BatchNorm1d(hidden_size_2)
        self.dropout_conv2 = nn.Dropout(p=conv_dropout_rate)

        # Higher hidden units
        self.pool = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ))

        # self.pool = aggr.MeanAggregation()

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(p=classifier_dropout_rate),
            nn.Linear(128, 2)
        )

        print(f"Created GNN model:")
        print(f"- Input features: {input_features}")
        print(f"- Hidden sizes: {hidden_size}")
        print(f"- Output classes: 2")
        print(f"- Convolution dropout rate: {conv_dropout_rate}")
        print(f"- Classifier dropout rate: {classifier_dropout_rate}")

    def forward(self, x, edge_index, batch):
        """
        Forward pass through the network
        x: node features
        edge_index: graph edges
        batch: which graph each node belongs to (for batching multiple graphs)
        """

        # First graph convolution + activation
        x = self.conv1(x, edge_index)
        # x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout_conv1(x)

        # Second graph convolution + activation
        x = self.conv2(x, edge_index)
        # x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout_conv2(x)

        # Pool node features to get one vector per graph
        # This averages all node features in each graph
        x = self.pool(x, batch)

        # Final classification
        x = self.classifier(x)

        return x


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


class DynamicGNN_Embedding(nn.Module):
    """
    Dynamic Graph Neural Network for binary classification with configurable hidden layers
    """

    def __init__(self, input_features, embedding_dim=16, hidden_sizes=None,
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
        super(DynamicGNN_Embedding, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 256]
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one element")

        self.num_layers = len(hidden_sizes)
        self.hidden_sizes = hidden_sizes
        self.embedding = nn.Embedding(400, embedding_dim)
        self.use_layer_norm = use_layer_norm
        self.total_input_features = input_features + embedding_dim - 1

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
        self.pool = AttentionalAggregation(gate_nn=nn.Sequential(
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
        print(f"- Embedding dimension: {self.embedding.embedding_dim}")
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
        x = self.pool(x, batch)

        # Final classification
        x = self.classifier(x)

        return x


class DynamicGAT(nn.Module):
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
        super(DynamicGAT, self).__init__()

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


class DynamicGIN(nn.Module):
    """
    Dynamic Graph Neural Network for binary classification with configurable hidden layers
    """

    def __init__(self, input_features, hidden_sizes=None,
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
        super(DynamicGIN, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 256]
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one element")

        self.num_layers = len(hidden_sizes)
        self.hidden_sizes = hidden_sizes
        # Create convolution layers dynamically
        self.convs = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        self.use_layer_norm = use_layer_norm
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
        print(f"Created Dynamic GIN model:")
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


class DynamicGIN_Embedding(nn.Module):
    """
    Dynamic Graph Neural Network for binary classification with configurable hidden layers
    """

    def __init__(self, input_features, embedding_dim=16, hidden_sizes=None,
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
        super(DynamicGIN_Embedding, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 256]
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one element")

        self.num_layers = len(hidden_sizes)
        self.hidden_sizes = hidden_sizes
        self.embedding = nn.Embedding(400, embedding_dim)
        # Create convolution layers dynamically
        self.convs = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        self.use_layer_norm = use_layer_norm
        self.total_input_features = input_features + embedding_dim - 1
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
        print(f"Created Dynamic GIN model:")
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

        node_types = x[:, 0].long()
        node_features = x[:, 1:]

        # Get embeddings for node types
        node_type_embeddings = self.embedding(node_types)
        # Concatenate embeddings with original features
        x = torch.cat((node_features, node_type_embeddings), dim=1)

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


class Multi_GNN(nn.Module):
    """
    Dynamic Graph Neural Network for binary classification with configurable hidden layers
    and convolution types.
    """

    def __init__(self, input_features, hidden_sizes=None, conv_type='GCN',
                 heads=4, conv_dropout_rate=0.3, classifier_dropout_rate=0.4,
                 use_layer_norm=True, pool_hidden_size=128):
        """
        Args:
            input_features: Number of input node features
            hidden_sizes: List of hidden layer sizes. Length determines number of hidden layers.
                         e.g., [256, 128] creates 2 hidden layers with these sizes.
            conv_type: Type of convolution layer ('GCN', 'GAT', or 'GIN').
            heads: Number of attention heads for GATConv.
            conv_dropout_rate: Dropout rate for convolution layers.
            classifier_dropout_rate: Dropout rate for classifier.
            use_layer_norm: Whether to use layer normalization.
            pool_hidden_size: Hidden size for the attention pooling gate network.
        """
        super(Multi_GNN, self).__init__()

        # --- Configuration Validation ---
        valid_conv_types = ['GCN', 'GAT', 'GIN']
        if conv_type not in valid_conv_types:
            raise ValueError(
                f"conv_type must be one of {valid_conv_types}, but got {conv_type}")

        if hidden_sizes is None:
            hidden_sizes = [256, 256]
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one element")

        self.num_layers = len(hidden_sizes)
        self.hidden_sizes = hidden_sizes
        self.conv_type = conv_type
        self.heads = heads
        self.use_layer_norm = use_layer_norm

        # --- Dynamic Layer Creation ---
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None

        current_input_size = input_features

        for i in range(self.num_layers):
            current_output_size = hidden_sizes[i]

            # GATConv needs special handling for input/output sizes due to attention heads
            if self.conv_type == 'GAT':
                # GATConv concatenates heads, so output size is hidden_size * heads
                if i < self.num_layers - 1:
                    layer_output_size = current_output_size
                else:
                    # Final GATConv layer usually averages heads for a single output vector
                    # However, to keep it consistent with the pooling layer, let's keep concatenating
                    # and adjust the final classifier and pooling layers accordingly.
                    layer_output_size = current_output_size

                self.convs.append(
                    GATConv(current_input_size, layer_output_size,
                            heads=self.heads, dropout=conv_dropout_rate))
                current_input_size = layer_output_size * self.heads

            # GCNConv and GINConv have a straightforward input/output size relationship
            elif self.conv_type == 'GIN':
                mlp = nn.Sequential(
                    nn.Linear(current_input_size, current_output_size),
                    nn.ReLU(),
                    nn.Linear(current_output_size, current_output_size)
                )
                self.convs.append(GINConv(mlp))
                current_input_size = current_output_size

            else:  # Default to GCNConv
                self.convs.append(
                    GCNConv(current_input_size, current_output_size))
                current_input_size = current_output_size

            if use_layer_norm:
                # The size of the layer norm needs to match the output size of the conv layer
                if self.conv_type == 'GAT':
                    self.layer_norms.append(
                        nn.LayerNorm(current_output_size * self.heads))
                else:
                    self.layer_norms.append(nn.LayerNorm(current_output_size))

        # --- Pooling and Classifier Layers ---
        # The input size for these layers depends on the final convolution layer's output
        final_conv_output_size = current_input_size

        self.pool = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(final_conv_output_size, pool_hidden_size),
            nn.ReLU(),
            nn.Linear(pool_hidden_size, 1)
        ))

        self.classifier = nn.Sequential(
            nn.Linear(final_conv_output_size, pool_hidden_size),
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
        print(f"- Convolution Type: {self.conv_type}")
        if self.conv_type == 'GAT':
            print(f"- GAT Attention Heads: {self.heads}")
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
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            if self.use_layer_norm:
                x = self.layer_norms[i](x)

            x = F.relu(x)
            # GATConv and GINConv have built-in dropout, so an extra dropout here is a choice.
            # We will keep it for consistency with the GCN implementation.
            x = F.dropout(x, p=F.dropout(x, p=self.conv_dropout_rate,
                                         training=self.training))

        x = self.pool(x, batch)
        x = self.classifier(x)
        return x


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

    # if num_graphs_to_use is not None:
    #     if num_graphs_to_use > len(dataset) - start_index - 1:
    #         print(
    #             f"Warning: num_graphs_to_use ({num_graphs_to_use}) is greater than dataset size ({len(dataset)}). Using full dataset.")
    #         actual_indices = list(range(len(dataset)))
    #     else:
    #         print(
    #             f"Limiting training to the first {num_graphs_to_use} graphs.")
    #         actual_indices = list(range(num_graphs_to_use))
    # else:
    #     actual_indices = list(
    #         range(len(dataset)))  # Use all graphs if not specified
    num_graphs_to_use = min(num_graphs_to_use, len(dataset) - start_index)
    # Split dataset into train/validation
    training_start = time.perf_counter()
    all_labels = dataset.get_labels()
    # labels = [all_labels[i] for i in actual_indices]
    # indices = list(range(len(actual_indices)))
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

    # Create train and validation datasets
    # train_dataset = [dataset[i] for i in train_indices]
    # val_dataset = [dataset[i] for i in val_indices]

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
            outputs = model(batch.x, batch.edge_index,
                            batch.batch)
            loss = criterion(outputs.squeeze(), batch.y.squeeze())

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
                outputs = model(batch.x,
                                batch.edge_index,
                                batch.batch)
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
