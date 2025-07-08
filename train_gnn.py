import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data

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

test_data = simple_convert_graph(G, first_label)
print(f"\nPyTorch Geometric Data object:")
print(f"- Node features (x): {test_data.x.shape}")
print(f"- Edge index: {test_data.edge_index.shape}")
print(f"- Label (y): {test_data.y}")
print(f"- Number of edges: {test_data.edge_index.shape[1]}")