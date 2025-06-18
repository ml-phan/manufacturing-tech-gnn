from OCC.Extend.DataExchange import read_step_file, write_stl_file

import os
import os.path
import sys


from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone, IFSelect_ItemsByEntity
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder
from OCC.Core.TopoDS import topods
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Display.SimpleGui import init_display

import OCC.Core.BRepGProp as brp
from OCC.Core.GProp import GProp_GProps
from OCC.Extend.TopologyUtils import TopologyExplorer

class VertexNode:
    def __init__(self, id):
        self.id = id  # Unique ID
        self.edges = set()
        self.neighbor_vertices = set()
        self.features = {}

class EdgeNode:
    def __init__(self, id, v1_id, v2_id, length=None):
        self.id = id  # Unique ID
        self.v1 = v1_id
        self.v2 = v2_id
        self.faces = set()
        self.length = length
        self.features = {}

class FaceNode:
    def __init__(self, id):
        self.id = id  # Unique ID
        self.edges = set()
        self.area = None
        self.features = {}

step_file_path = r"E:\gnn_data\step_files_test\119997_Thumbpick_Smaller_Right_Handed.step"
st_file_path = step_file_path.replace("step", "st")
shp = read_step_file(
        step_file_path
        # r"E:\Downloads\square_pocket.step",
    )
t1 = TopologyExplorer(shp)
for face in t1.faces():
    # get face surface area
    props = GProp_GProps()
    brp.brepgprop.SurfaceProperties(face, props)
    print("Area:", props.Mass())

for edge in t1.edges():
    # get edge length
    props = GProp_GProps()
    brp.brepgprop.LinearProperties(edge, props)
    print("Length:", props.Mass())

# write_stl_file(shp, st_file_path)
