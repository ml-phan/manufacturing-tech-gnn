import open3d as o3d
import os
import numpy as np
import matplotlib.pyplot as plt
from OCC.Core.AIS import AIS_PointCloud
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.IFSelect import IFSelect_RetDone, IFSelect_ItemsByEntity
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Display.SimpleGui import init_display
from OCC.Extend.DataExchange import read_step_file, write_stl_file

from pathlib import Path
from tqdm import tqdm

def make_pointnet_dataset(step_dir, stl_dir, pointcloud_dir, num_points=2048):
    failed_conversion_file = Path("pointcloud_failed_conversion.txt")
    failed_conversion_file.write_text("Point Cloud failed conversions:\n")

    Path(step_dir).mkdir(parents=True, exist_ok=True)
    Path(stl_dir).mkdir(parents=True, exist_ok=True)
    valid_extensions = ['.stp', '.step']
    with tqdm(list(Path(step_dir).glob("*"))) as pbar:
        for step_file in pbar:
            if step_file.suffix.lower() in valid_extensions:
                pbar.set_description("Processing: {}".format(step_file.name))
                stl_path = stl_dir + "\\" + step_file.with_suffix(".stl").name
                txt_path = pointcloud_dir + "\\" + step_file.with_suffix(".txt").name
                try:
                    shape = read_step_file(str(step_file))
                    write_stl_file(shape, stl_path)

                    # point_cloud = gettarget(stl_dir, step_file.with_suffix(".stl").name, num_points)
                    point_cloud = sample_mesh_to_pointcloud(stl_path, num_points)
                    if point_cloud:
                        points = point_cloud.points
                        normals = point_cloud.normals
                        with open(txt_path, 'w') as f:
                            for i in range(num_points):
                                point = points[i]
                                normal = normals[i]
                                line = '{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n'.format(
                                    point[0], point[1], point[2],
                                    normal[0], normal[1], normal[2]
                                )
                                f.write(line)
                    else:
                        print(f"Failed to generate point cloud for {step_file}")

                except Exception as e:
                    with failed_conversion_file.open("a") as f:
                        f.write(f"{step_file.name}. Error : {e}\n")
                    print(f"Error processing {step_file}: {e}")


def gettarget(stl_file, num_points):
    try:
        mesh=o3d.io.read_triangle_mesh(stl_file)
        if mesh.has_triangles() is True:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
            area=sum(cluster_area)
            x=num_points#int(area//4) #means points on 2mmx2mm grid #was 4
            target = mesh.sample_points_uniformly(number_of_points=x, use_triangle_normal=True)
        else:
            raise ValueError
    except:
        print('Failed to read Mesh')
        try:
            target=o3d.io.read_point_cloud(stl_file)
        except:
            print('Failed to read Point Cloud Data')
            target=False
    return target


def sample_mesh_to_pointcloud(stl_file_path, num_points=2048):
    """
    Convert a 3D mesh file to a point cloud with specified number of points.

    Args:
        stl_file_path (Path): Path to the mesh file (.stl)
        num_points (int): Number of points to sample from the mesh surface

    Returns:
        o3d.geometry.PointCloud or None: Point cloud with points and normals,
                                        or None if conversion fails
    """
    try:
        # Read mesh file
        mesh = o3d.io.read_triangle_mesh(stl_file_path)

        # Check if mesh has valid triangle data
        if not mesh.has_triangles():
            print(f"Warning: Mesh {stl_file_path} has no triangles")

        # Check if mesh is empty
        if len(mesh.triangles) == 0:
            print(f"Warning: Mesh {stl_file_path} is empty")
            return None

        # Sample points uniformly from mesh surface with normals
        point_cloud = mesh.sample_points_uniformly(
            number_of_points=num_points,
            use_triangle_normal=True
        )

        return point_cloud

    except Exception as e:
        print(f"Failed to read mesh from {stl_file_path}: {e}")


def visualize_pointcloud_open3d(filepath):
    """
    Visualize point cloud from txt file using Open3D
    Assumes format: x y z nx ny nz (coordinates + normals)
    """
    # Load data from txt file
    data = np.loadtxt(filepath, delimiter=",")

    # Extract coordinates and normals
    points = data[:, :3]  # x, y, z coordinates
    normals = data[:, 3:6] if data.shape[
                                  1] >= 6 else None  # nx, ny, nz normals

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Add normals if available
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)

    # Optional: Color points based on normals or coordinates
    # if normals is not None:
    #     # Color based on normal directions
    #     colors = np.abs(normals)  # Use absolute values for RGB
    #     pcd.colors = o3d.utility.Vector3dVector(colors)
    # else:
    #     # Color based on height (z-coordinate)
    #     z_coords = points[:, 2]
    #     z_normalized = (z_coords - z_coords.min()) / (
    #                 z_coords.max() - z_coords.min())
    #     colors = plt.cm.viridis(z_normalized)[:, :3]  # Use viridis colormap
    #     pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize
    o3d.visualization.draw_geometries([pcd],
                                      window_name="Point Cloud Visualization",
                                      width=800, height=600,
                                      point_show_normal=True,
                                      left=50, top=50)

def visualize_pointcloud_with_normals(filepath):
    """
    Visualize point cloud with normals as lines.
    Assumes format: x y z nx ny nz
    """
    # Load data
    data = np.loadtxt(filepath, delimiter=",")
    points = data[:, :3]
    normals = data[:, 3:6]

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    # Add color for visualization (optional)
    pcd.paint_uniform_color([0.7, 0.7, 0.7])  # light gray

    # Visualize with normals
    o3d.visualization.draw_geometries(
        [pcd],
        point_show_normal=True,  # <<=== show normals as lines
        window_name="Point Cloud with Normals",
        width=900, height=700
    )

# Example
# visualize_pointcloud_with_normals("example_pointcloud.txt")


if __name__ == '__main__':
    # step_dir = r"E:\gnn_data\step_files_square"
    # Path(step_dir).mkdir(parents=True, exist_ok=True)
    # stl_dir = r"E:\gnn_data\stl_files_square"
    # Path(stl_dir).mkdir(parents=True, exist_ok=True)
    # pointcloud_dir = r"E:\gnn_data\pointcloud_files_square"
    # Path(pointcloud_dir).mkdir(parents=True, exist_ok=True)
    # make_pointnet_dataset(step_dir, stl_dir, pointcloud_dir, num_points=2048)
    stl_file = r"E:\gnn_data\pointcloud_files_square\square_pocket.txt"
    visualize_pointcloud_open3d(stl_file)
    # visualize_pointcloud_with_normals(stl_file)