import pandas as pd
import sys

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone, IFSelect_ItemsByEntity
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder
from OCC.Core.TopoDS import topods
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Display.SimpleGui import init_display
from OCC.Extend.TopologyUtils import TopologyExplorer
from pathlib import Path

from tqdm import tqdm


def read_step_file(filename):
    """read the STEP file and returns a compound"""
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(filename)

    if status == IFSelect_RetDone:  # check status
        failsonly = False
        step_reader.PrintCheckLoad(failsonly, IFSelect_ItemsByEntity)
        step_reader.PrintCheckTransfer(failsonly, IFSelect_ItemsByEntity)
        step_reader.TransferRoot(1)
        a_shape = step_reader.Shape(1)
    else:
        print("Error: can't read file.")
        sys.exit(0)
    return a_shape


def extract_step_file_features(folder_path, start_index=0, end_index=None):
    """Extracts features from STEP files in the given folder."""
    step_file_features = {
        "item_id": [],
        "faces": [],
        "edges": [],
        "vertices": [],
        "file_name": [],
    }
    files = list(Path(folder_path).glob("*.*"))
    if end_index:
        files = files[start_index:end_index]
    else:
        files = files[start_index:]
    failed_files = []
    with tqdm(files) as pbar:
        for file in pbar:
            pbar.set_postfix_str(file.name)
            try:
                item_id = file.stem.split("_")[0]
                stp = read_step_file(str(file))
                te = TopologyExplorer(stp)
                number_of_faces = len(list(te.faces()))
                number_of_edges = len(list(te.edges()))
                number_of_vertices = len(list(te.vertices()))
                step_file_features["item_id"].append(item_id)
                step_file_features["faces"].append(number_of_faces)
                step_file_features["edges"].append(number_of_edges)
                step_file_features["vertices"].append(number_of_vertices)
                step_file_features["file_name"].append(file.name)
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                step_file_features["item_id"].append(file.stem)
                step_file_features["faces"].append(None)
                step_file_features["edges"].append(None)
                step_file_features["vertices"].append(None)
                step_file_features["file_name"].append(file.name)
                failed_files.append(file.name)
    if failed_files:
        with open("failed_files.txt", "w") as f:
            for failed_file in failed_files:
                f.write(f"{failed_file}\n")

    return pd.DataFrame(step_file_features)

if __name__ == '__main__':
    folder_path = "E:/step_files"
    start_index = 40000
    end_index = None  # Set to None to process all files
    features_df = extract_step_file_features(folder_path, start_index, end_index)
    # Save the features DataFrame to a CSV file
    features_df.to_csv(f"step_file_features_{start_index}_{end_index}.csv", index=False)
    print(features_df)
    # List the files in a natural order
    # for file in list(Path(folder_path).glob("*.stp"))[:5]:
