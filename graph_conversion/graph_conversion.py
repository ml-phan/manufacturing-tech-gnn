from concurrent.futures import ProcessPoolExecutor
from graph_conversion_utils import *
from pathlib import Path
from tqdm import tqdm

def process_single_file(step_file_path, graphml_dir):

    file = Path(step_file_path)
    graphml_file_name = f"{file.stem}.graphml"
    graphml_file_path = Path(graphml_dir) / graphml_file_name
    try:
        # Parse the STEP file
        _, step_data = parse_file(file)
        all_flat_nodes, fast_dict_search = get_nodes_from_datas(step_data)
        replace_nodes(all_flat_nodes, fast_dict_search)

        # Create a directed graph and add nodes and edges
        G = nx.DiGraph()
        all_nodes_to_graph(G, all_flat_nodes, graphml_file_name, graphml_file_path)
        return file.name, "Success"
    except Exception as e:
        print(f"Failed to process {file.name}: {e}")
        return file.name, "Failed"


if __name__ == '__main__':
    step_file_dir = r"E:\gnn_data\square_pocket"
    graphml_dir = r"E:\gnn_data\square_pocket"

    all_files = list(Path(step_file_dir).glob("*.*"))
    failed_files = Path("failed_conversion.txt")
    with ProcessPoolExecutor(max_workers=14) as executor:
        futures = [
            executor.submit(process_single_file, file, graphml_dir)
            for file in all_files
        ]
        with tqdm(futures) as pbar:
            for future in pbar:
                pbar.set_description(f"Processing: {future.result()[0]}")
                file_name, result = future.result()  # Wait for the future to complete
                if result == "Failed":
                    print(f"Failed to process {file_name}")
                    with open(failed_files, "a") as failed_file:
                        failed_file.write(f"{file_name}\n")

