import networkx as nx
import os
import pandas as pd
import pickle
import torch
import torch.nn as nn

from torch_geometric.data import Data
from tqdm import tqdm


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


def enhanced_convert_graph(networkx_graph, label, node_type_to_index):
    nodes = list(networkx_graph.nodes())
    node_to_index = {node: i for i, node in enumerate(nodes)}
    print(f"Graph has {len(nodes)} nodes")
    print("Start calculating raw features...")
    # Calculate centrality measures once
    degree_centrality = nx.degree_centrality(networkx_graph)
    betweenness_centrality = nx.betweenness_centrality(networkx_graph, k=100)
    closeness_centrality = nx.closeness_centrality(networkx_graph)
    page_rank = nx.pagerank(networkx_graph)
    average_neighbor_degree = nx.average_neighbor_degree(networkx_graph)
    print("Finished calculating raw features")
    # clustering_coeff = nx.clustering(networkx_graph)

    num_types = len(node_type_to_index)
    embedding_layer = nn.Embedding(num_types, 64)  # Example embedding size

    node_features = []
    for node in tqdm(nodes):
        # Get node type (assuming it's stored as an attribute)
        node_type = networkx_graph.nodes[node].get('type', '')
        type_idx = node_type_to_index.get(node_type,
                                          0)  # Default to 0 if unknown
        with torch.no_grad():
            type_embedding = embedding_layer(torch.tensor(type_idx)).numpy()
        features = [
            float(networkx_graph.degree(node)),  # Degree
            degree_centrality[node],  # Degree centrality
            betweenness_centrality[node],  # Betweenness centrality
            closeness_centrality[node],  # Closeness centrality
            page_rank[node],  # PageRank
            average_neighbor_degree[node],  # Average neighbor degree
            # clustering_coeff[node],  # Clustering coefficient
        ]
        features.extend(type_embedding.tolist())
        node_features.append(features)

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

def preprocess_and_save_dataset(dataframe, save_dir):
    """
    Convert all GraphML files to PyTorch Geometric format and save them
    """

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    print(f"Pre-processing {len(dataframe)} samples...")
    print(f"Saving processed data to: {save_dir}")

    processed_files = []
    failed_files = []

    for idx in tqdm(range(len(dataframe)), desc="Processing files"):
        file_path = None
        try:
            # Get file info
            file_path = dataframe.iloc[idx]['graphml_file']
            label = dataframe.iloc[idx]['is_cnc']

            with open(r"../all_attribute_type.pkl", "rb") as f:
                all_attribute_type = sorted(list(pickle.load(f)))
            node_type_to_index = {node_type: idx for idx, node_type in
                                  enumerate(all_attribute_type)}

            # Create a unique filename for the processed data
            base_name = os.path.basename(file_path).replace('.graphml', '')
            processed_filename = f"{base_name}.pt"
            processed_path = os.path.join(save_dir, processed_filename)

            # Skip if already processed
            if os.path.exists(processed_path):
                processed_files.append({
                    'original_path': file_path,
                    'processed_path': processed_path,
                    'label': label,
                    'index': idx
                })
                continue

            # Load and convert GraphML
            G = nx.read_graphml(file_path)
            data = enhanced_convert_graph(G, label, node_type_to_index)

            # Save using PyTorch's save function
            torch.save(data, processed_path)

            # Keep track of processed files
            processed_files.append({
                'original_path': file_path,
                'processed_path': processed_path,
                'label': label,
                'index': idx
            })

        except Exception as e:
            print(f"Failed to process {file_path}: {str(e)}")
            failed_files.append(
                {'path': file_path, 'error': str(e), 'index': idx})

    # Save the mapping file
    mapping = {
        'processed_files': processed_files,
        'failed_files': failed_files,
        'total_original': len(dataframe),
        'total_processed': len(processed_files),
        'total_failed': len(failed_files)
    }

    mapping_path = os.path.join(save_dir, 'dataset_mapping.pkl')
    with open(mapping_path, 'wb') as file:
        pickle.dump(mapping, file)

    print(f"\nProcessing complete!")
    print(f"- Successfully processed: {len(processed_files)}")
    print(f"- Failed: {len(failed_files)}")
    print(f"- Mapping saved to: {mapping_path}")

    return processed_files, failed_files

if __name__ == '__main__':
    df = pd.read_csv(r"../data/synced_dataset.csv")
    # Example usage (adjust the save directory path)
    PROCESSED_DATA_DIR = r"E:\gnn_data\processed_step_data_full_node_features"
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # For testing, let's use a small subset first
    # Change this to df for your full dataset
    test_df = df.head(10)  # Just 10 samples for testing

    print("Starting pre-processing...")
    processed_files, failed_files = preprocess_and_save_dataset(df,
                                                                PROCESSED_DATA_DIR)

    # Check what we created
    print(f"\nChecking processed files:")
    for i, pf in enumerate(processed_files[:3]):  # Show first 3
        print(f"File {i}: {pf['processed_path']} (label: {pf['label']})")

        # Load one to verify
        loaded_data = torch.load(pf['processed_path'], weights_only=False)
        print(
            f"  - Nodes: {loaded_data.x.shape[0]}, Edges: {loaded_data.edge_index.shape[1]}")
