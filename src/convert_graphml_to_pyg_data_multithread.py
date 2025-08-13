import joblib
import networkx as nx
import numpy as np
import os
import pandas as pd
import pickle
import torch
import torch.nn as nn

from pathlib import Path
from multiprocessing import Pool, Manager
from torch_geometric.data import Data
from tqdm import tqdm


def enhanced_convert_graph(
        networkx_graph, label, node_type_index
):
    nodes = list(networkx_graph.nodes())
    node_to_index = {node: i for i, node in enumerate(nodes)}
    # print(f"Graph has {len(nodes)} nodes")
    # print("Start calculating raw features...")
    # Calculate centrality measures
    degree_centrality = nx.degree_centrality(networkx_graph)
    betweenness_centrality = nx.betweenness_centrality(networkx_graph, k=100)
    closeness_centrality = nx.closeness_centrality(networkx_graph)
    page_rank = nx.pagerank(networkx_graph)
    average_neighbor_degree = nx.average_neighbor_degree(networkx_graph)
    # print("Finished calculating raw features")

    node_features = []
    for node in nodes:
        # Get node type (assuming it's stored as an attribute)
        node_type = networkx_graph.nodes[node].get('type', '')
        type_idx = node_type_index.get(node_type,
                                       0)  # Default to 0 if unknown
        features = [
            type_idx,
            float(networkx_graph.degree(node)),  # Degree
            degree_centrality[node],  # Degree centrality
            betweenness_centrality[node],  # Betweenness centrality
            closeness_centrality[node],  # Closeness centrality
            page_rank[node],  # PageRank
            average_neighbor_degree[node],  # Average neighbor degree
            # clustering_coeff[node],  # Clustering coefficient

        ]
        node_features.append(features)

    # Convert to PyTorch tensor
    x = torch.tensor(node_features, dtype=torch.float)
    # print(f"Node features shape: {x.shape}")

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

    # print(f"Edge index shape: {edge_index.shape}")

    # Step 3d: Create label tensor
    y = torch.tensor([label], dtype=torch.long)

    # Step 3e: Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, y=y)

    return data


def enhanced_convert_graph_global_features(
        networkx_graph, label, node_type_index, global_features=None
):
    nodes = list(networkx_graph.nodes())
    node_to_index = {node: i for i, node in enumerate(nodes)}
    # Calculate centrality measures once
    degree_centrality = nx.degree_centrality(networkx_graph)
    betweenness_centrality = nx.betweenness_centrality(networkx_graph, k=100)
    closeness_centrality = nx.closeness_centrality(networkx_graph)
    average_neighbor_degree = nx.average_neighbor_degree(networkx_graph)
    page_rank = nx.pagerank(networkx_graph)

    node_features = []
    for node in nodes:
        # Get node type
        node_type = networkx_graph.nodes[node].get('type', '')
        type_idx = node_type_index.get(node_type,
                                       0)  # Default to 0 if unknown
        node_data = 0
        features = [
            type_idx,
            float(networkx_graph.degree(node)),  # Degree
            degree_centrality[node],  # Degree centrality
            betweenness_centrality[node],  # Betweenness centrality
            closeness_centrality[node],  # Closeness centrality
            page_rank[node],  # PageRank
            average_neighbor_degree[node],  # Average neighbor degree

        ]
        node_features.append(features)

    # Convert to PyTorch tensor
    x = torch.tensor(node_features, dtype=torch.float)
    # print(f"Node features shape: {x.shape}")

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

    # print(f"Edge index shape: {edge_index.shape}")

    # Step 3d: Create label tensor
    y = torch.tensor([label], dtype=torch.long)

    # Step 3e: Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, y=y,
                global_features=global_features)

    return data


def enhanced_convert_graph_global_features_v2(
        networkx_graph, label, label_multi, node_type_index,
        global_features=None, item_id=None,
):
    nodes = list(networkx_graph.nodes())
    node_to_index = {node: i for i, node in enumerate(nodes)}
    # Calculate centrality measures once
    node_degree = networkx_graph.degree()
    degree_centrality = nx.degree_centrality(networkx_graph)
    betweenness_centrality = nx.betweenness_centrality(networkx_graph)
    closeness_centrality = nx.closeness_centrality(networkx_graph)
    page_rank = nx.pagerank(networkx_graph)
    average_neighbor_degree = nx.average_neighbor_degree(networkx_graph)
    clustering_coefficient = nx.clustering(networkx_graph)
    triangles = nx.triangles(networkx_graph)

    node_features = []
    for node in networkx_graph.nodes(data=True):
        # Calculate architectural features
        # Number of features: 8
        node_id = node[0]
        features = [
            item_id,
            node_degree[node_id],  # Node Degree
            degree_centrality[node_id],  # Degree centrality
            average_neighbor_degree[node_id],  # Average neighbor degree
            triangles[node_id],  # Number of triangles
            page_rank[node_id],  # PageRank
            betweenness_centrality[node_id],  # Betweenness centrality
            closeness_centrality[node_id],  # Closeness centrality
            clustering_coefficient[node_id],  # Clustering coefficient
        ]
        # Extract node data and extend features
        # Number of node features: 11
        # Surface_type, Area, Perimeter, Edge Count, Vertex Count, Compactness,
        # U Span, V Span, Mean Curvature, Gaussian Curvature, Orientation
        node_data = list(node[-1].values())
        features.extend(node_data)

        node_features.append(features)

    # Convert to PyTorch tensor
    x = torch.tensor(node_features, dtype=torch.float)
    # print(f"Node features shape: {x.shape}")

    # Step 3c: Create edge list
    edge_list = []
    edge_atrribute = []
    for edge in networkx_graph.edges(data=True):
        source, target, attrs = edge
        # Convert node names to indices
        source_idx = node_to_index[source]
        target_idx = node_to_index[target]

        # Add both directions (undirected graph)
        edge_list.append([source_idx, target_idx])
        edge_list.append([target_idx, source_idx])
        # Add edge attributes
        # Number of attributes: 6
        # Shared Face Count, Length, Curve Type, Is_Closed, Chord Length,
        # Orientation
        edge_attr = list(attrs.values())
        edge_atrribute.append(edge_attr)
        edge_atrribute.append(edge_attr)

    # Convert to PyTorch tensor and transpose
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    if edge_atrribute:
        edge_atrribute = torch.tensor(edge_atrribute, dtype=torch.float)
    else:
        edge_atrribute = torch.empty((0, 0), dtype=torch.float)
    # print(f"Edge index shape: {edge_index.shape}")

    # Step 3d: Create label tensor
    y = torch.tensor([label], dtype=torch.long)
    y_multi = torch.tensor([label_multi], dtype=torch.long)

    # Step 3e: Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_atrribute,
                y=y, y_multi=y_multi, global_features=global_features)

    return data


def init_worker(attribute_path):
    with open(attribute_path, "rb") as f:
        all_attribute_type = sorted(list(pickle.load(f)))
    return {node_type: idx for idx, node_type in
            enumerate(all_attribute_type)}


def process_single_file(args):
    idx, row, save_dir, node_type_to_index = args
    file_path = row['graphml_file']
    item_id = row['item_id']
    label = row['is_cnc']
    label_multi = row["multiclass_labels"]
    global_features_cols = [
        "faces", "edges", "vertices", "quantity",
        "height", "width", "depth", "volume", "area",
        "bbox_height", "bbox_width", "bbox_depth", "bbox_volume",
        "bbox_area",
    ]
    global_features = np.array(row[global_features_cols], dtype=np.float32)
    global_features_tensor = torch.tensor(global_features,
                                          dtype=torch.float).unsqueeze(0)
    try:
        base_name = os.path.basename(file_path).replace('.graphml', '')
        processed_filename = f"{base_name}.pt"
        processed_path = os.path.join(save_dir, processed_filename)

        if os.path.exists(processed_path):
            return 'success', {
                'original_path': file_path,
                'processed_path': processed_path,
                'label': label,
                'index': idx
            }

        G = nx.read_graphml(file_path)
        data = enhanced_convert_graph_global_features_v2(
            networkx_graph=G,
            label=label,
            label_multi=label_multi,
            node_type_index=node_type_to_index,
            global_features=global_features_tensor,
            item_id=item_id
        )
        for key, value in data:
            if 0 in value.shape:
                return "fail", {
                    'path': file_path,
                    'error': f'Empty graph data in attribute {key}',
                    'index': idx
                }
            if torch.isinf(value).any():
                return "fail", {
                    'path': file_path,
                    'error': f'Infinite values in attribute {key}',
                    'index': idx
                }
        if data.x is not None:
            data.x = torch.cat(
                (
                    data.x[:, :5],
                    data.x[:, 9:17],
                    data.x[:, 5:9],
                    data.x[:, 17:],
                ),
                dim=1

            )
        if (data.x > 1e8).any():
            print(f"Warning: Large values found in node features for {file_path}. ")
        if (data.x[:, 5:6] < 0).any():
            print(f"Warning: Negative values found in node features for {file_path}. ")
            with open(r"E:\gnn_data\step_negative_area.txt", "a+") as f:
                f.write(f"{file_path} - {data.x[:, 5:6].min().item()}\n")

        # if data.edge_attr is not None:
        #     data.edge_attr = torch.cat(
        #         (data.edge_attr[:, :2], data.edge_attr[:, 3:4],
        #          data.edge_attr[:, 2:3], data.edge_attr[:, 4:]), dim=1)
        torch.save(data, processed_path)

        return 'success', {
            'original_path': file_path,
            'processed_path': processed_path,
            'label': label,
            'index': idx
        }

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 'fail', {
            'path': file_path,
            'error': str(e),
            'index': idx
        }


def preprocess_and_save_dataset_parallel(
        dataframe, save_dir,
        node_type_to_index,
        num_workers=None,
        number_of_files=None
):
    os.makedirs(save_dir, exist_ok=True)

    print(f"Pre-processing {len(dataframe)} samples...")
    print(f"Saving processed data to: {save_dir}")
    tasks = [(idx, dataframe.iloc[idx], save_dir, node_type_to_index)
             for idx in range(len(dataframe[:number_of_files]))]

    processed_files = []
    failed_files = []

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_single_file, tasks),
                            total=len(tasks)))

    for status, result in results:
        if status == 'success':
            processed_files.append(result)
        else:
            failed_files.append(result)

    # Save mapping
    mapping = {
        'processed_files': processed_files,
        'failed_files': failed_files,
        'total_original': len(dataframe),
        'total_processed': len(processed_files),
        'total_failed': len(failed_files)
    }

    mapping_path = os.path.join(save_dir, 'dataset_mapping.pkl')
    with open(mapping_path, 'wb') as f:
        pickle.dump(mapping, f)

    print(f"\nProcessing complete!")
    print(f"- Successfully processed: {len(processed_files)}")
    print(f"- Failed: {len(failed_files)}")
    print(f"- Mapping saved to: {mapping_path}")

    return processed_files, failed_files


if __name__ == '__main__':
    df = pd.read_csv(r"../data/synced_dataset_final.csv")
    # Example usage (adjust the save directory path)
    PROCESSED_DATA_DIR = r"E:\gnn_data\pyg_data_v2"
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    node_type_to_index = init_worker(r"..\all_attribute_type.pkl")
    # For testing, let's use a small subset first
    # Change this to df for your full dataset
    # test_df = df.head(10)  # Just 10 samples for testing
    print("Starting pre-processing...")

    processed_files, failed_files = preprocess_and_save_dataset_parallel(
        dataframe=df,
        save_dir=PROCESSED_DATA_DIR,
        node_type_to_index=node_type_to_index,
        num_workers=14,
        number_of_files=None,
    )

    # Check what we created
    print(f"\nChecking processed files:")
    for i, pf in enumerate(processed_files[:3]):  # Show first 3
        print(f"File {i}: {pf['processed_path']} (label: {pf['label']})")

        # Load one to verify
        loaded_data = torch.load(pf['processed_path'], weights_only=False)
        print(
            f"  - Nodes: {loaded_data.x.shape[0]}, Edges: {loaded_data.edge_index.shape[1]}")
