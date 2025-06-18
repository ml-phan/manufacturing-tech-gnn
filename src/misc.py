import pandas as pd
from pathlib import Path
from collections import defaultdict
import re

# Load your aligned DataFrame
df = pd.read_csv("../data/dataset.csv")

# Define base directories
graphml_dir = Path(r"E:\gnn_data\graphml_files")
pointcloud_dir = Path(r"E:\gnn_data\pointcloud_files")


