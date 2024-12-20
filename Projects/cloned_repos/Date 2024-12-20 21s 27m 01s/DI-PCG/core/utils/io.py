import numpy as np
import bpy
from PIL import Image

def read_list_from_txt(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines

def save_points_as_ply(points, filename):
    # Define the PLY header
    # Save the points to a PLY file
    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex " + str(len(points)) + "\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for point in points:
            f.write("{} {} {}\n".format(point[0], point[1], point[2]))

def make_gif(save_name, image_list):
    frames = [Image.open(image) for image in image_list]
    frame_one = frames[0]
    frame_one.save(save_name, save_all=True, append_images=frames[1:], fps=15, loop=0, disposal=2, optimize=False, lossless=True)

def clean_scene():
   bpy.ops.object.select_all(action='SELECT')
   bpy_data = [bpy.data.actions,
         bpy.data.armatures,
         bpy.data.brushes,
         bpy.data.cameras,
         bpy.data.materials,
         bpy.data.meshes,
         bpy.data.objects,
         bpy.data.shape_keys,
         bpy.data.textures,
         bpy.data.collections,
         bpy.data.node_groups,
         bpy.data.images,
         bpy.data.movieclips,
         bpy.data.curves,
         bpy.data.particles]

   for bpy_data_iter in bpy_data:
      for data in bpy_data_iter:
          bpy_data_iter.remove(data, do_unlink=True)


