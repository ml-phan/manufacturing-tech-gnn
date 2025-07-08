import networkx as nx
import pickle

from tqdm import tqdm
from pathlib import Path


def get_all_node_types_from_dataset():
    all_attribute_type = set()
    failed_files = Path("./failed_graphml_files.txt")
    for file in tqdm(list(Path(r"E:\gnn_data\graphml_files").glob("*.graphml"))):
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