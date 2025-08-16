import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import time
import concurrent.futures

def cuboid_mask(matrix, base_z, base_y, base_x, cuboid_depth, cuboid_height, cuboid_width,
                yaw=0.0, pitch=0.0, roll=0.0, taper_width=0.0, taper_height=0.0):
    """
    Create a 3D boolean mask with a tapered, rotated cuboid (rectangular prism).

    Tapering values are normalized: 1.0 means taper to a point at the end of the cuboid.

    Parameters:
        matrix (np.ndarray): 3D array to match shape.
        base_z/y/x (float): Starting corner (base) of the cuboid.
        cuboid_depth (float): Length of the cuboid along its local z-axis.
        cuboid_height (float): Initial height at base (local y direction).
        cuboid_width (float): Initial width at base (local x direction).
        taper_width (float): Fraction of width to taper along z (0 = no taper, 1 = taper to point).
        taper_height (float): Same for height.
        yaw, pitch, roll (float): Rotation angles in degrees (ZYX order).

    Returns:
        np.ndarray: 3D boolean mask.
    """
    depth, height, width = matrix.shape

    zz, yy, xx = np.meshgrid(
        np.arange(depth), np.arange(height), np.arange(width), indexing='ij'
    )

    z = zz - base_z
    y = yy - base_y
    x = xx - base_x

    coords = np.stack([z, y, x], axis=-1)

    yaw_rad, pitch_rad, roll_rad = np.deg2rad([yaw, pitch, roll])
    Rz = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                   [np.sin(yaw_rad),  np.cos(yaw_rad), 0],
                   [0, 0, 1]])
    Ry = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                   [0, 1, 0],
                   [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll_rad), -np.sin(roll_rad)],
                   [0, np.sin(roll_rad),  np.cos(roll_rad)]])

    R = Rz @ Ry @ Rx
    R_inv = R.T  

    coords_rot = np.tensordot(coords, R_inv, axes=([3], [1]))
    z_rot, y_rot, x_rot = coords_rot[..., 0], coords_rot[..., 1], coords_rot[..., 2]

    in_depth = (z_rot >= 0) & (z_rot < cuboid_depth)

    depth_frac = np.clip(z_rot / cuboid_depth, 0.0, 1.0)

    half_w = (cuboid_width * (1.0 - taper_width * depth_frac)) / 2.0
    half_h = (cuboid_height * (1.0 - taper_height * depth_frac)) / 2.0

    half_w = np.clip(half_w, a_min=0.0, a_max=None)
    half_h = np.clip(half_h, a_min=0.0, a_max=None)

    in_width = np.abs(x_rot) <= half_w
    in_height = np.abs(y_rot) <= half_h

    return in_depth & in_width & in_height

def elliptical_cylinder_mask(matrix, base_z, base_y, base_x, cylinder_length, 
                            radius_y, radius_x, yaw=0.0, pitch=0.0, roll=0.0, taper_z=0.0):
    """
    Create a 3D boolean mask with a rotated, tapered elliptical cylinder.

    Parameters:
        matrix (np.ndarray): 3D volume to match shape.
        base_z/y/x (float): Starting point of the cylinder.
        cylinder_length (float): Length of the cylinder along local z-axis.
        radius_y (float): Semi-axis length in y direction.
        radius_x (float): Semi-axis length in x direction.
        yaw/pitch/roll (float): Euler angles in degrees.
        taper_z (float): Linear tapering along z-axis (0 = cylinder, 1 = cone to point).

    Returns:
        np.ndarray: Boolean mask.
    """
    depth, height, width = matrix.shape

    zz, yy, xx = np.meshgrid(
        np.arange(depth), np.arange(height), np.arange(width), indexing='ij'
    )

    z = zz - base_z
    y = yy - base_y
    x = xx - base_x
    coords = np.stack([z, y, x], axis=-1)

    yaw_rad, pitch_rad, roll_rad = np.deg2rad([yaw, pitch, roll])
    Rz = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                   [np.sin(yaw_rad),  np.cos(yaw_rad), 0],
                   [0, 0, 1]])
    Ry = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                   [0, 1, 0],
                   [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll_rad), -np.sin(roll_rad)],
                   [0, np.sin(roll_rad),  np.cos(roll_rad)]])
    R = Rz @ Ry @ Rx
    R_inv = R.T

    coords_rot = np.tensordot(coords, R_inv, axes=([3], [1]))
    z_rot, y_rot, x_rot = coords_rot[..., 0], coords_rot[..., 1], coords_rot[..., 2]

    z_frac = np.clip(z_rot / cylinder_length, 0.0, 1.0)
    local_radius_y = radius_y * (1.0 - taper_z * z_frac)
    local_radius_x = radius_x * (1.0 - taper_z * z_frac)
    local_radius_y = np.clip(local_radius_y, a_min=0.0, a_max=None)
    local_radius_x = np.clip(local_radius_x, a_min=0.0, a_max=None)

    in_length = (z_rot >= 0) & (z_rot < cylinder_length)
    
    safe_radius_y = np.where(local_radius_y > 0, local_radius_y, 1e-10)
    safe_radius_x = np.where(local_radius_x > 0, local_radius_x, 1e-10)
    
    in_ellipse = (y_rot**2 / safe_radius_y**2 + x_rot**2 / safe_radius_x**2) <= 1.0

    return in_length & in_ellipse

