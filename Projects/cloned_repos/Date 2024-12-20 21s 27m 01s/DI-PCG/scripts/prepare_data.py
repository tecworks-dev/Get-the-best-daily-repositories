import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2
import gin
import bpy
import gc
import logging
import time
import argparse
from pathlib import Path
import importlib
import json
import copy
import imgaug
import imgaug.augmenters as iaa
from core.utils.io import read_list_from_txt
from multiprocessing import Pool
import torch
from core.utils.dinov2 import Dinov2Model

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
from core.utils.camera import convert_sphere_to_xyz, setup_camera
from core.utils.vis_utils import colorObj, setMat_plastic

# augment
color_aug = iaa.Sequential([
    # color aug
    iaa.WithBrightnessChannels(iaa.Add((-50, 50))),
    iaa.GammaContrast((0.7, 1.5), per_channel=True),
    iaa.AddToHueAndSaturation((-60, 60)),
    iaa.Grayscale((0.0, 0.8)),
])
flip_aug = iaa.Sequential([iaa.Fliplr(0.5)])
crop_aug = iaa.Sequential([
    iaa.CropAndPad(percent=(-0.1, 0.1), pad_mode='constant', pad_cval=(0, 0), keep_size=False),
    iaa.CropToFixedSize(height=256, width=256),
    iaa.PadToFixedSize(height=256, width=256)
])
crop_resize_aug = iaa.KeepSizeByResize(iaa.Crop(percent=(0, 0.1), sample_independently=False, keep_size=False))

def aug(name, save_root, num_aug, flip=True, crop=True):
    """Do the augmentation to RGBA image
    """
    id, name = name.split("/")
    img_rgba = cv2.imread(os.path.join(save_root, id, name), -1) # rgba
    img = cv2.cvtColor(img_rgba[:,:,:3], cv2.COLOR_BGR2RGB)
    img = np.concatenate([img, img_rgba[:,:,3:]], axis=2) # rgba

    save_dir = os.path.join(save_root, id)
    os.makedirs(save_dir, exist_ok=True)
    for j in range(num_aug):
        # do the augmentation here
        image, mask = copy.deepcopy(img[:,:,:3]), copy.deepcopy(img[:,:,3:])
        # aug color
        if np.random.rand() < 0.8:
            image = color_aug(image=image)
        # flip
        if flip:
            image = flip_aug(image=np.concatenate([image, mask], axis=-1))
        else:
            image = np.concatenate([image, mask], axis=-1)
        # crop
        if crop:
            if np.random.rand() < 0.5:
                # crop & pad
                image = crop_aug(image=image)
            else:
                # crop & resize
                image = crop_resize_aug(image=image)
        if np.random.rand() < 0.1:
            # binary image using mask
            image, mask = image[:, :, :3], image[:, :, 3:]
            # black image
            image = np.tile(255 * (1.0 - (mask > 0)), (1,1,3)).astype(np.uint8)
            image = np.concatenate([image, mask], axis=-1)
        if np.random.rand() < 0.2:
            image, mask = image[:, :, :3], image[:, :, 3:]
            edge = np.expand_dims(cv2.Canny(mask, 100, 200), -1)
            mask = (edge > 0).astype(np.uint8)
            # convert edge into black
            edge = 255 * (1.0 - mask)
            image = np.tile(edge, (1,1,3)).astype(np.uint8)
            image = np.concatenate([image, mask], axis=-1)
        # save
        save_name = os.path.join(save_dir, "{}_aug_{}.png".format(name[:-4], j))
        cv2.imwrite(save_name, image)
    print(name)


def randomize_params(params_dict):
    # Initialize the parameters
    selected_params = {}
    for key, value in params_dict.items():
        if value[0] == 'continuous':
            min_v, max_v = value[1][0], value[1][1]
            selected_params[key] = np.random.uniform(min_v, max_v)
        elif value[0] == 'discrete':
            choice_list = value[1]
            selected_params[key] = np.random.choice(choice_list)
        else:
            raise NotImplementedError
    return selected_params

def generate(generator, params, seed, save_dir=None, save_name=None, 
             save_blend=False, save_img=False, save_untexture_img=False, save_gif=False, save_mesh=False,
             cam_dists=[], cam_elevations=[], cam_azimuths=[], zoff=0, 
             resolution='256x256', sample=100, no_mod=False, no_ground=True,
             window=None, screen=None):
    print("Generating")
    # reset to default
    bpy.ops.wm.read_homefile(app_template="")
    butil.clear_scene()
    # Suppress info messages
    bpy.ops.outliner.orphans_purge()
    gc.collect()
    # configurate infinigen
    gin.add_config_file_search_path("./configs/infinigen")
    gin.parse_config_files_and_bindings(
        ["configs/infinigen/base.gin"],
        bindings=[],
        skip_unknown=True,
        finalize_config=False,
    )
    surface.registry.initialize_from_gin()
    
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
    
    scene.render.resolution_x, scene.render.resolution_y = map(
        int, resolution.split("x")
    )
    scene.cycles.samples = sample
    bpy.context.scene.render.use_persistent_data = True
    bpy.context.scene.world.node_tree.nodes["Background"].inputs[0].default_value[0:3] = (0.0, 0.0, 0.0)
    
    # update the parameters
    generator.update_params(params)
    # generate the object
    asset = generator.spawn_asset(seed)
    generator.finalize_assets(asset)
    
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
    
    if save_blend:
        # visualize the generated model by rendering
        butil.save_blend(f"{save_dir}/{save_name}.blend", autopack=True)
    
    # render image
    if save_img:
        sky_lighting.add_lighting()
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        nodes = bpy.data.worlds["World"].node_tree.nodes
        sky_texture = [n for n in nodes if n.name.startswith("Sky Texture")][-1]
        sky_texture.sun_elevation = np.deg2rad(60)
        sky_texture.sun_rotation = np.pi * 0.75
        
        for cd in cam_dists:
            for ce in cam_elevations:
                for ca in cam_azimuths:
                    save_name_full = f"{save_name}_{cd}_{ce}_{ca}"
                    cam_data = convert_sphere_to_xyz(cd, ce, ca)
                    cam_location, cam_rot = cam_data[:3], cam_data[3:]
                    cam_location[-1] += zoff # TODO: fix the table case
                    camera = setup_camera(cam_location=cam_location, cam_rot=cam_rot)
                    cam_info_ng = bpy.data.node_groups.get("nodegroup_active_cam_info")
                    if cam_info_ng is not None:
                        cam_info_ng.nodes["Object Info"].inputs["Object"].default_value = camera

                    image_path = str(f"{save_dir}/{save_name_full}_texture.png")
                    scene.render.filepath = image_path
                    bpy.ops.render.render(write_still=True)
                    # render untextured object
                    if save_untexture_img:
                        bpy.ops.object.shade_smooth()
                        asset.data.materials.clear()
                        # untextured model
                        #RGBA = (144.0/255, 210.0/255, 236.0/255, 1)
                        RGBA = (192.0/255, 192.0/255, 192.0/255, 1)
                        meshColor = colorObj(RGBA, 0.5, 1.0, 1.0, 0.0, 2.0)
                        setMat_plastic(asset, meshColor)
                    image_path = str(f"{save_dir}/{save_name_full}_geometry.png")
                    scene.render.filepath = image_path
                    bpy.ops.render.render(write_still=True)

    # render gif of object rotating
    if save_gif:
        save_gif_dir = os.path.join(save_dir, "gif")
        os.makedirs(save_gif_dir, exist_ok=True)
        bpy.context.scene.frame_end = 60
        asset_parent = asset if asset.parent is None else asset.parent
        asset_parent.driver_add("rotation_euler")[-1].driver.expression = f"frame/{60 / (2 * np.pi * 1)}"
        imgpath = str(f"{save_gif_dir}/{save_name}_###.png")
        scene.render.filepath = str(imgpath)
        bpy.ops.render.render(animation=True)
        from core.utils.io import make_gif
        all_imgpaths = [str(os.path.join(save_gif_dir, p)) for p in sorted(os.listdir(save_gif_dir)) if p.endswith('.png')]
        make_gif(f"{save_gif_dir}/{save_name}.gif", all_imgpaths)
    
    # dump mesh model
    if save_mesh:
        save_mesh_filepath = os.path.join(save_dir, save_name+".glb")
        bpy.ops.export_scene.gltf(filepath=save_mesh_filepath)
        print("Mesh saved in {}".format(save_mesh_filepath))

    return asset, image_path

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--generator', type=str, default='ChairFactory', 
                           help='Supported generator: [ChairFactory, VaseFactory, TableDiningFactory, BasketBaseFactory, FlowerFactory, DandelionFactory]')
    argparser.add_argument('--save_root', type=str, required=True)
    argparser.add_argument('--total_num', type=int, default=20000)
    argparser.add_argument('--aug_num', type=int, default=5)
    argparser.add_argument('--batch_size', type=int, default=1000)
    argparser.add_argument('--seed', type=int, default=0)
    args = argparser.parse_args()
    
    # setup
    """Training image rendering settings
    Chair: cam_dists - [1.8, 2.0], elevations: [50, 60, 80], azimuths: [0, 30, 60, 80], zoff: 0.0
    Vase: cam_dists - [1.2, 1.6, 2.0], elevations: [60, 80, 90], azimuths: [0], zoff: 0.3
    Table: cam_dists - [5.0, 6.0], elevations: [60, 70], azimuths: [0, 30, 60], zoff: 0.1
    Flower: cam_dists - [3.0, 4.0], elevations: [20, 30, 50, 60], azimuths: [0], zoff: 0
    Dandelion: cam_dists - [3.0], elevations: [90], azimuths: [0], zoff: 0.5
    Basket: cam_dists - [1.2, 1.6], elevations: [50, 60, 70], azimuths: [30, 60], zoff: 0.0
    """
    np.random.seed(args.seed)
    flip, crop = True, True
    # Different training data rendering settings for different generators to improve efficiency
    if args.generator == "ChairFactory":
        cam_dists = [1.8, 2.0]
        elevations = [50, 60, 80]
        azimuths = [0, 30, 60, 80]
        zoff = 0.0
    elif args.generator == "TableDiningFactory":
        cam_dists = [5.0, 6.0]
        elevations = [60, 70]
        azimuths = [0, 30, 60, 90]
        zoff = 0.1
    elif args.generator == "VaseFactory":
        cam_dists = [1.2, 1.6, 2.0]
        elevations = [60, 80, 90]
        azimuths = [0]
        zoff = 0.3
    elif args.generator == "BasketBaseFactory":
        cam_dists = [1.2, 1.6]
        elevations = [50, 60, 70]
        azimuths = [30, 60]
        zoff = 0.0
    elif args.generator == "FlowerFactory":
        cam_dists = [2.0, 3.0, 4.0]
        elevations = [30, 50, 60, 80]
        azimuths = [0]
        zoff = 0
    elif args.generator == "DandelionFactory":
        cam_dists = [3.0]
        elevations = [90]
        azimuths = [0]
        zoff = 0.5
        flip = False
    
    os.makedirs(args.save_root, exist_ok=True)
    train_ratio = 0.9
    sample = 100
    resolution = '256x256'
    
    
    # load the Blender procedural generator
    OBJECTS_PATH = Path("./core/assets/")
    assert OBJECTS_PATH.exists(), OBJECTS_PATH
    generator = None
    for subdir in sorted(list(OBJECTS_PATH.iterdir())):
        clsname = subdir.name.split(".")[0].strip()
        with gin.unlock_config():
            module = importlib.import_module(f"core.assets.{clsname}")
        if hasattr(module, args.generator):
            generator = getattr(module, args.generator)
            logger.info(f"Found {args.generator} in {subdir}")
            break
        logger.debug(f"{args.generator} not found in {subdir}")
    if generator is None:
        raise ModuleNotFoundError(f"{args.generator} not Found.")
    gen = generator(args.seed)
    
    # save params dict file
    params_dict_file = f"{args.save_root}/params_dict.txt"
    json.dump(gen.params_dict, open(params_dict_file, "w"))
    
    # generate data main loop
    for i in range(args.total_num):
        # sample parameters
        params = randomize_params(gen.params_dict)
        # fix dependent parameters
        params_fix_unused = gen.fix_unused_params(params)
        save_name = f"{i:05d}"
        save_dir = f"{args.save_root}/{save_name}"
        os.makedirs(save_dir, exist_ok=True)
        # generate and save rendering - for training data, skip the blend file to save storage
        if i < args.total_num * train_ratio:
            save_blend = False
        else:
            save_blend = True
        asset, img_path = generate(gen, params_fix_unused, args.seed, save_dir=save_dir, save_name=save_name,
                 save_blend=save_blend, save_img=True, cam_dists=cam_dists,
                 cam_elevations=elevations, cam_azimuths=azimuths, zoff=zoff, sample=sample, resolution=resolution)
        # save the parameters
        json.dump(params_fix_unused, open(f"{save_dir}/params.txt", "w"), default=str)
        
        if i % 100 == 0:
            logger.info(f"{i} / {args.total_num} finished")
    
    # write filelist
    f = open(os.path.join(args.save_root, "train_list_mv.txt"), "w")
    total_num = args.total_num
    for i in range(int(total_num * train_ratio)):
        for cam_dist in cam_dists:
            for elevation in elevations:
                for azimuth in azimuths:
                    f.write(
                        "{:05d}/{:05d}_{}_{}_{}.png\n".format(
                            i, i, cam_dist, elevation, azimuth
                        )
                    )
    f.close()
    f = open(os.path.join(args.save_root, "test_list_mv.txt"), "w")
    for i in range(int(total_num * train_ratio), total_num):
        for cam_dist in cam_dists:
            for elevation in elevations:
                for azimuth in azimuths:
                    f.write(
                        "{:05d}/{:05d}_{}_{}_{}.png\n".format(
                            i, i, cam_dist, elevation, azimuth
                            )
                    )
    f.close()
    # do the augmentation
    # main loop
    image_list = read_list_from_txt(os.path.join(args.save_root, "train_list_mv.txt"))
    print("Augmenting...Total data: {}".format(len(image_list)))
    p = Pool(16)
    for i, name in enumerate(image_list):
        p.apply_async(aug, args=(name, args.save_root, args.aug_num, flip, crop))
    p.close()
    p.join()

    # write the new list
    f = open(os.path.join(args.save_root, "train_list_mv_withaug.txt"), "w")
    for i in range(int(total_num * train_ratio)):
        for cam_dist in cam_dists:
            for elevation in elevations:
                for azimuth in azimuths:
                    f.write(
                        "{:05d}/{:05d}_{}_{}_{}.png\n".format(
                            i, i, cam_dist, elevation, azimuth
                        )
                    )
                    for j in range(args.aug_num):
                        f.write(
                            "{:05d}/{:05d}_{}_{}_{}_aug_{}.png\n".format(
                                i, i, cam_dist, elevation, azimuth, j
                            )
                        )
    f.close()
    
    # extract features
    # Setup PyTorch:
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    dinov2_model = Dinov2Model()
    
    # read image paths
    with open(os.path.join(args.save_root, "train_list_mv_withaug.txt"), "r") as f:
        image_paths = f.readlines()
    with open(os.path.join(args.save_root, "test_list_mv.txt"), "r") as f:
        test_image_paths = f.readlines()
    image_paths = image_paths + test_image_paths
    
    image_paths = [os.path.join(args.save_root, path.strip()) for path in image_paths]
    print(f"Number of images: {len(image_paths)}")
    for i in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[i:i + args.batch_size]
        # pre-process the image - RGBA to RGB with white background
        batch_images = []
        for path in batch_paths:
            image = cv2.imread(path, -1)
            mask = (image[...,-1:] > 0)
            image_rgb = cv2.cvtColor(image[...,:3], cv2.COLOR_BGR2RGB)
            # resize if not 256
            if image.shape[0] != 256 or image.shape[1] != 256:
                image_rgb = cv2.resize(image_rgb, (256, 256), interpolation=cv2.INTER_NEAREST)
                mask = cv2.resize((255 * mask[:,:,0]).astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
                mask = (mask > 128)[:,:,None]
            # convert the transparent pixels to white background
            image_whiteback = image_rgb * mask + 255 * (1 - mask)
            batch_images.append(np.array(image_whiteback).astype(np.uint8))

        
        batch_features = dinov2_model.encode_batch_imgs(batch_images, global_feat=False).detach().cpu().numpy()
        save_paths = [p.replace(".png", "_dino_token.npz") for p in batch_paths]
        # save the features
        for save_path, feature in zip(save_paths, batch_features):
            np.savez_compressed(save_path, feature)
        print(f"Extracted features for {i} images.")
