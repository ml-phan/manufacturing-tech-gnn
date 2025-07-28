import networkx as nx
import pickle
import time

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

def process_single_file(file_path):
    """Process a single GraphML file and return results"""
    try:
        G = nx.read_graphml(file_path)
        number_of_nodes = G.number_of_nodes()
        number_of_edges = G.number_of_edges()
        file_id = file_path.stem.split("_")[0]
        return file_id, {
            "number_of_nodes": number_of_nodes,
            "number_of_edges": number_of_edges
        }, None
    except Exception as e:
        return None, None, (file_path.name, str(e))

def process_graphml_parallel(folder_path, max_workers=None, number_of_files=None, use_threads=False):
    """Process GraphML files in parallel using ProcessPoolExecutor or ThreadPoolExecutor"""

    files = list(Path(folder_path).glob("*.*"))[:number_of_files]
    old_graphml = {}
    failed_graphml_old = []
    ExecutorClass = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    with ExecutorClass(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = [executor.submit(process_single_file, file) for file in files]

        # Process results as they complete
        with tqdm(future_to_file, desc="Processing files") as pbar:
            for future in as_completed(future_to_file):
                # test_list.append(future.result())
                file_id_, result_, error_ = future.result()

                if error_:
                    print(f"Failed to process {error_[0]}: {error_[1]}")
                    failed_graphml_old.append(error_[0])
                else:
                    old_graphml[file_id_] = result_

                pbar.update(1)

    return old_graphml, failed_graphml_old

if __name__ == '__main__':
    start = time.perf_counter()
    graph_dir = Path(r"E:\gnn_data\graphml_files_old")
    old_graphml, failed_graphml_old = process_graphml_parallel(
        graph_dir,
        max_workers=14,
        number_of_files=1000,
        use_threads=False
    )
    end = time.perf_counter()
    print(f"Processed {len(old_graphml)} files in {end - start:.2f} seconds.")
    with open(f"{graph_dir.name}.pkl", "wb") as p_file:
        pickle.dump(old_graphml, p_file)
    # with open("old_graphml.pkl", "rb") as f:
    #     old_graphml = pickle.load(f)