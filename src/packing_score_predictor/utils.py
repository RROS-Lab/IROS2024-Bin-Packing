"""
    General utilities for normalizing, transforming data
"""
import numpy as np
import open3d as o3d


def min_max_normalization(data_arr:np.array, min_val:float = None, max_val: float = None)->np.array:
     
    if(min_val == None or max_val == None):
        print("Computing default")
        new_data_array = (data_arr-np.min(data_arr))/(np.max(data_arr) - np.min(data_arr))
    else:
        new_data_array = (data_arr-min_val)/(max_val-min_val)

    return new_data_array

def max_normalization(data_arr:np.array)->np.array:
    return data_arr/np.max(data_arr)


def z_normalization(data_arr:np.array)->np.array:

    mean = np.mean(data_arr)
    std_dev = np.std(data_arr)

    return (data_arr-mean)/std_dev


def visualize_bbox(bbox_1, bbox_2, pcd_1 = None, pcd_2 = None):

    axes_points = np.array([
    [0, 0, 0],
    [0.1, 0, 0],
    [0, 0.1, 0],
    [0, 0, 0.1]
        ])
    axes_colors = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0]
    ])
    axes_lines = [
        [0, 1],
        [0, 2],
        [0, 3]
    ]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(axes_points),
        lines=o3d.utility.Vector2iVector(axes_lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(axes_colors)

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(axes_points),
        lines=o3d.utility.Vector2iVector(axes_lines)
    )
    # Rotate and translate the LineSet to match the bounding box orientation and position
    line_set.rotate(bbox_1.R, bbox_1.center)
    line_set.translate(bbox_1.center)

    line_set_2 = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(axes_points),
        lines=o3d.utility.Vector2iVector(axes_lines)
    )
    line_set_2.colors = o3d.utility.Vector3dVector(axes_colors)

    line_set_2 = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(axes_points),
        lines=o3d.utility.Vector2iVector(axes_lines)
    )
    line_set_2.colors = o3d.utility.Vector3dVector(axes_colors)

    line_set_2 = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(axes_points),
        lines=o3d.utility.Vector2iVector(axes_lines)
    )
    line_set_2.colors = o3d.utility.Vector3dVector(axes_colors)
    
    # Rotate and translate the LineSet to match the bounding box orientation and position
    line_set_2.rotate(bbox_2.R, bbox_2.center)
    line_set_2.translate(bbox_2.center)
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(bbox_1)
    visualizer.add_geometry(bbox_2)
    visualizer.add_geometry(line_set)
    visualizer.add_geometry(line_set_2)
    if(pcd_1 is not None):
        visualizer.add_geometry(pcd_1)
    if(pcd_2 is not None):
        visualizer.add_geometry(pcd_2)
    visualizer.run()
    visualizer.destroy_window()

    return