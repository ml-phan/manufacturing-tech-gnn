import networkx as nx
import pickle

from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
import torch

def get_all_node_types_from_dataset():
    all_attribute_type = set()
    failed_files = Path("./failed_graphml_files.txt")
    for file in tqdm(
            list(Path(r"E:\gnn_data\graphml_files").glob("*.graphml"))):
        try:
            G = nx.read_graphml(file)
            for node, attrs in (G.nodes(data=True)):
                all_attribute_type.add(attrs.get("type"))
        except Exception as e:
            print(f"Error processing {file}: {e}")
            with failed_files.open("a") as failed_file:
                failed_file.write(file.name + " ." + str(e) + "\n")

    len(all_attribute_type)
    with open("all_attribute_type.pkl", "wb") as pickle_file:
        pickle.dump(all_attribute_type, pickle_file)


def parallel_execute(func: callable, iterable, num_processes=14):
    with Pool(processes=num_processes) as pool:
        results = list(
            tqdm(pool.imap_unordered(func, iterable), total=len(iterable)))
    return results

def arrange_tensor(tensor_file):
    try:
        data = torch.load(tensor_file, weights_only=False)
        # if data.x is not None:
        #     data.x = torch.cat((data.x[:, :2], data.x[:, 5:6], data.x[:, 7:16],
        #                         data.x[:, 2:5], data.x[:, 6:7],
        #                         data.x[:, 16:]), dim=1)
        if data.edge_attr is not None:
            data.edge_attr = torch.cat(
                (data.edge_attr[:, :2], data.edge_attr[:, 3:4],
                 data.edge_attr[:, 2:3], data.edge_attr[:, 4:]), dim=1)
        torch.save(data, tensor_file)
    except Exception as e:
        print(f"Error processing {tensor_file}: {e}")