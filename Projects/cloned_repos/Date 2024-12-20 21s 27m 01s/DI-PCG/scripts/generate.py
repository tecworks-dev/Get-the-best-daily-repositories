import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2
import gin
import bpy
import gc
import logging
import argparse
from pathlib import Path
import importlib
import yaml

logging.basicConfig(
    format="[%(asctime)s.%(msecs)03d] [%(module)s] [%(levelname)s] | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


import infinigen
from infinigen.core import init, surface
from infinigen.assets.utils.decorate import read_co
from infinigen.assets.utils.misc import assign_material
from infinigen.core.util import blender as butil
from infinigen.assets.lighting import (
    hdri_lighting,
    holdout_lighting,
    sky_lighting,
    three_point_lighting,
)
from core.utils.vis_utils import colorObj, setMat_plastic


def generate(generator, params, seed, mesh_save_path, no_mod=False, no_ground=True):
    print(params)
    print("Generating")
    # reset to default
    bpy.ops.wm.read_homefile(app_template="")
    butil.clear_scene()
    # Suppress info messages
    bpy.ops.outliner.orphans_purge()
    gc.collect()
    # configurate infinigen
    gin.add_config_file_search_path("configs/infinigen")
    gin.parse_config_files_and_bindings(
        ["configs/infinigen/base.gin"],
        bindings=[],
        skip_unknown=True,
        finalize_config=False,
    )
    surface.registry.initialize_from_gin()
    print("Configured")
    
    # setup the scene
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.render.film_transparent = True
    bpy.context.preferences.system.scrollback = 0
    bpy.context.preferences.edit.undo_steps = 0
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for dt in prefs.get_device_types(bpy.context):
        prefs.get_devices_for_type(dt[0])
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"

    use_devices = [d for d in prefs.devices if d.type == "CUDA"]
    for d in prefs.devices:
        d.use = False
    for d in use_devices:
        d.use = True
    
    bpy.context.scene.world.node_tree.nodes["Background"].inputs[0].default_value[0:3] = (0.0, 0.0, 0.0)
    print("Setup done")
    
    # update the parameters
    generator.update_params(params)
    # generate the object
    asset = generator.spawn_asset(seed)
    generator.finalize_assets(asset)
    print("Generated")
    
    parent = asset
    if asset.type == "EMPTY":
        meshes = [o for o in asset.children_recursive if o.type == "MESH"]
        sizes = []
        for m in meshes:
            co = read_co(m)
            sizes.append((np.amax(co, 0) - np.amin(co, 0)).sum())
        i = np.argmax(np.array(sizes))
        asset = meshes[i]
    if not no_mod:
        if parent.animation_data is not None:
            drivers = parent.animation_data.drivers.values()
            for d in drivers:
                parent.driver_remove(d.data_path)
        co = read_co(asset)
        x_min, x_max = np.amin(co, 0), np.amax(co, 0)
        parent.location = -(x_min[0] + x_max[0]) / 2, -(x_min[1] + x_max[1]) / 2, 0
        butil.apply_transform(parent, loc=True)
        if not no_ground:
            bpy.ops.mesh.primitive_grid_add(
                size=5, x_subdivisions=400, y_subdivisions=400
            )
            plane = bpy.context.active_object
            plane.location[-1] = x_min[-1]
            plane.is_shadow_catcher = True
            material = bpy.data.materials.new("plane")
            material.use_nodes = True
            material.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (
                0.015,
                0.009,
                0.003,
                1,
            )
            assign_material(plane, material)
    
    # dump mesh model
    save_mesh_filepath = mesh_save_path
    bpy.ops.export_scene.gltf(filepath=save_mesh_filepath)
    print("Mesh saved in {}".format(save_mesh_filepath))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", type=str, required=True)
    argparser.add_argument('--output_path', type=str, required=True)
    argparser.add_argument('--params_path', type=str, required=True)
    argparser.add_argument('--seed', type=int, default=0)
    args = argparser.parse_args()

    # load config
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # load generators
    generators_choices = ["chair", "table", "vase", "basket", "flower", "dandelion"]
    factory_names = ["ChairFactory", "TableDiningFactory", "VaseFactory", "BasketBaseFactory", "FlowerFactory", "DandelionFactory"]

    category = args.config.split("/")[-1].split("_")[0]
    idx = generators_choices.index(category)
    
    # load generator
    module = importlib.import_module(f"core.assets.{category}")
    gen = getattr(module, factory_names[idx])
    generator = gen(args.seed)

    # load params
    params = np.load(args.params_path, allow_pickle=True).item()
    generate(generator, params, args.seed, args.output_path)