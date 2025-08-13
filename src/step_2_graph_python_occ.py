import os
import sys
import networkx as nx
import pickle

from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone, IFSelect_ItemsByEntity
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
import OCC.Core.BRepGProp as brp
from OCC.Core.GProp import GProp_GProps
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.TopExp import TopExp_Explorer


def read_step_file(filename):
    """read the STEP file and returns a compound"""
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(str(filename))

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


def get_face_properties(face):
    face_properties = {}

    # Surface Type
    surface = BRepAdaptor_Surface(face)

    # Surface Area
    surface_props = GProp_GProps()
    brp.brepgprop.SurfaceProperties(face, surface_props)
    face_properties["area"] = surface_props.Mass()

    # Centroid
    # centroid = surface_props.CentreOfMass()
    # face_properties["centroid"] = (centroid.X(), centroid.Y(), centroid.Z())

    # Edge Count, Vertex Count, and Perimeter
    topo_explorer = TopologyExplorer(face)
    perimeter = 0.0
    for edge in topo_explorer.edges():
        edge_props = GProp_GProps()
        brp.brepgprop.LinearProperties(edge, edge_props)
        perimeter += edge_props.Mass()
    face_properties['perimeter'] = perimeter
    face_properties['edge_count'] = len(list(topo_explorer.edges()))
    face_properties['vertex_count'] = len(list(topo_explorer.vertices()))

    # Compactness (how close to a circle a face is)
    if face_properties['perimeter'] > 0 and face_properties['area'] > 0:
        face_properties["compactness"] = (4 * 3.14159 * face_properties[
            'area']) / (face_properties['perimeter'] ** 2)
    else:
        face_properties["compactness"] = 0.0

    # U and V Parameters
    u_min, u_max, v_min, v_max = surface.FirstUParameter(), surface.LastUParameter(), \
        surface.FirstVParameter(), surface.LastVParameter()
    u_span = u_max - u_min
    v_span = v_max - v_min
    face_properties["u_span"] = u_span
    face_properties["v_span"] = v_span

    # Mean and Gaussian Curvature at the center of the face
    u_mid = (u_min + u_max) / 2.0
    v_mid = (v_min + v_max) / 2.0
    geom_props = BRepLProp_SLProps(surface, 2, 1e-6)
    geom_props.SetParameters(u_mid, v_mid)
    if geom_props.IsCurvatureDefined():
        face_properties["mean_curvature"] = geom_props.MeanCurvature()
        # face_properties["gaussian_curvature"] = geom_props.GaussianCurvature()

    face_properties["orientation"] = face.Orientation()
    face_properties["surface_type"] = surface.GetType()

    return face_properties


def get_edge_properties(edge):
    edge_properties = {}

    # Length
    edge_props = GProp_GProps()
    brp.brepgprop.LinearProperties(edge, edge_props)
    curve_adapter = BRepAdaptor_Curve(edge)

    edge_properties["curve_type"] = curve_adapter.GetType()
    edge_properties["length"] = edge_props.Mass()

    # Curve Type

    # Start and End Points
    # edge_properties["start_point"] = curve_adapter.FirstParameter()
    # edge_properties["end_point"] = curve_adapter.LastParameter()

    # edge_properties["edge_type"] = {
    #     GeomAbs_Line: 0,
    #     GeomAbs_Circle: 1,
    #     GeomAbs_Ellipse: 2,
    #     GeomAbs_BSplineCurve: 3
    # }.get(curve_type, -1)
    # edge_properties["is_straight"] = int(curve_type == GeomAbs_Line)

    umin = curve_adapter.FirstParameter()
    umax = curve_adapter.LastParameter()
    # umid = (umin + umax) / 2
    # edge_properties["pnt"] = curve_adapter.Value(umid)
    # edge_properties["param_range"] = umax - umin
    start_pnt = curve_adapter.Value(umin)
    end_pnt = curve_adapter.Value(umax)
    # edge_properties['start_point'] = [start_pnt.X(), start_pnt.Y(), start_pnt.Z()]
    # edge_properties['end_point'] = [end_pnt.X(), end_pnt.Y(), end_pnt.Z()]

    edge_properties["chord_length"] = start_pnt.Distance(end_pnt)
    edge_properties["orientation"] = edge.Orientation()
    edge_properties["is_closed"] = int(curve_adapter.IsClosed())

    return edge_properties


def parse_step_with_occ(step_file_path: Path or str):
    shape = read_step_file(step_file_path)

    # Parse all faces
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    edge_to_faces = defaultdict(list)
    face_list = []
    face_index = 0
    face_to_index = {}
    face_properties = []
    while face_explorer.More():
        face = face_explorer.Current()
        if face:
            face_list.append(face)
            face_to_index[face] = face_index
            face_index += 1
            face_properties.append(get_face_properties(face))
        face_explorer.Next()

    edge_hash_to_edge = {}
    edge_properties_dict = {}

    # Parse all edges for each face
    for face_idx, face in enumerate(face_list):
        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while edge_explorer.More():
            edge = edge_explorer.Current()
            edge_hash = edge.__hash__()
            if edge_hash not in edge_hash_to_edge:
                edge_hash_to_edge[edge_hash] = edge
                # Extract edge properties once per unique edge
                edge_properties_dict[edge_hash] = get_edge_properties(edge)

            edge_to_faces[edge_hash].append((face_idx, edge))
            edge_explorer.Next()
    G = nx.Graph()
    for idx, (face, props) in enumerate(zip(face_list, face_properties)):
        G.add_node(idx, **props)

    # shared_edges_info = []
    # processed_edges_pair = []

    for edge_hash, face_list in edge_to_faces.items():
        # if len(face_list) == 2:
        #     face1_idx, face1 = face_list[0]
        #     face2_idx, face2 = face_list[1]
        #     shared_edges_info.append((edge_hash, face1_idx, face2_idx))
        #     G.add_edge(face1_idx, face2_idx, edge_hash=edge_hash,
        #                shared_edge_count=len(face_list))
        if len(face_list) >= 2:
            edge_props = edge_properties_dict[edge_hash]
            for i in range(len(face_list)):
                for j in range(i + 1, len(face_list)):
                    face1_idx, face1 = face_list[i]
                    face2_idx, face2 = face_list[j]
                    # shared_edges_info.append((edge_hash, face1_idx, face2_idx))
                    edge_attributes = {
                        # 'edge_hash': edge_hash,
                        'shared_face_count': len(face_list),
                        **edge_props,  # Unpack all edge properties

                    }
                    G.add_edge(face1_idx, face2_idx, **edge_attributes)
    return shape, G


def convert_step_to_graph(step_file_path: Path or str):
    """
    Convert a STEP file to a NetworkX graph.
    Each face is a node, and edges represent shared edges between faces.
    """
    step_path = Path(step_file_path)
    try:
        shape, G = parse_step_with_occ(step_path)
        graph_dir = Path("E:\gnn_data\graphml_files_v2")
        graph_path = graph_dir / step_path.with_suffix(".graphml").name
        nx.write_graphml_lxml(G, graph_path)
        return "success", {
            "path": str(graph_path),
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
        }
    except Exception as e:
        print(f"Error processing {step_path}: {e}")
        return "fail", {
            "path": str(step_path),
            "error": str(e)
        }


if __name__ == "__main__":

    processed_files = []
    failed_files = []
    graph_dir = Path("E:\gnn_data\graphml_files_v2")
    graph_dir.mkdir(parents=True, exist_ok=True)

    step_files = list(Path("E:\gnn_data\step_files").glob("*.*"))
    with Pool(processes=14) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(convert_step_to_graph, step_files),
                total=len(step_files)
            )
        )

    for status, result in results:
        if status == "fail":
            failed_files.append(result)

    # Save mapping

    failed_files_path = graph_dir / 'failed_files.pkl'
    with open(failed_files_path, 'wb') as f:
        pickle.dump(failed_files, f)

    print(f"\nProcessing complete!")
    print(f"- Successfully processed: {len(processed_files)}")
    print(f"- Failed: {len(failed_files)}")

