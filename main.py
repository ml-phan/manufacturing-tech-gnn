import os
import time

from pathlib import Path
from src.data_loader import *
from src.training import *
from src.evaluation import *
from src.visualization import *

if __name__ == '__main__':
    total = 0
    for file in os.listdir(r"C:\step_files"):
        if file.endswith(".STP"):
            total += 1
            print(file)
    print("Total files: ", total)


