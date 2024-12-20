import numpy as np
from scipy.spatial.transform import Rotation 
import bpy


def setup_camera(cam_location, cam_rot=(0, 0, 0)):
    bpy.ops.object.camera_add(location=cam_location, rotation=cam_rot)
    camera = bpy.context.active_object
    camera.rotation_mode = 'XYZ'
    bpy.data.scenes["Scene"].camera = camera
    scene = bpy.context.scene
    camera.data.sensor_height = (
        camera.data.sensor_width * scene.render.resolution_y / scene.render.resolution_x
    )
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            area.spaces.active.region_3d.view_perspective = "CAMERA"
            break
    cam_info_ng = bpy.data.node_groups.get("nodegroup_active_cam_info")
    if cam_info_ng is not None:
        cam_info_ng.nodes["Object Info"].inputs["Object"].default_value = camera
    return camera


def convert_sphere_to_xyz(dist, elevation, azimuth):
    """
    Convert spherical to cartesian coordinates. Assume camera is always looking at origin.
    """
    elevation_rad = elevation * np.pi / 180.0
    azimuth_rad = azimuth * np.pi / 180.0
    cam_pos_x = dist * np.sin(elevation_rad) * np.cos(azimuth_rad)
    cam_pos_y = dist * np.sin(elevation_rad) * np.sin(azimuth_rad)
    cam_pos_z = dist * np.cos(elevation_rad)
    # rotation
    gaze_direction = -np.array([cam_pos_x, cam_pos_y, cam_pos_z])
    R_look_at = look_at_rotation_matrix(gaze_direction)
    cam_rot_x, cam_rot_y, cam_rot_z = rotation_matrix_to_euler_angles(R_look_at)
    return [cam_pos_x, cam_pos_y, cam_pos_z, cam_rot_x, cam_rot_y, cam_rot_z]

def look_at_rotation_matrix(gaze_direction, world_up=(0,0,1)):
    forward = gaze_direction / np.linalg.norm(gaze_direction)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    R_look_at = np.array([right, up, -forward]).T
    return R_look_at

def rotation_matrix_to_euler_angles(R):  
    r =  Rotation.from_matrix(R)
    angles = r.as_euler("xyz", degrees=False)

    return angles[0], angles[1], angles[2]
