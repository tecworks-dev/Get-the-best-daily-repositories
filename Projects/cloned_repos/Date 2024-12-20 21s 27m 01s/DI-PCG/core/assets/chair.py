# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

import infinigen
from infinigen.assets.material_assignments import AssetList
from infinigen.assets.utils.decorate import (
    read_co,
    read_edge_center,
    read_edge_direction,
    remove_edges,
    remove_vertices,
    select_edges,
    solidify,
    subsurf,
    write_attribute,
    write_co,
)
from infinigen.assets.utils.draw import align_bezier, bezier_curve
from infinigen.assets.utils.nodegroup import geo_radius
from infinigen.assets.utils.object import join_objects, new_bbox
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.surface import NoApply
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform
from infinigen.core.util.random import random_general as rg


class ChairFactory(AssetFactory):
    back_types = {
        0: "whole",
        1: "partial",
        2: "horizontal-bar",
        3: "vertical-bar",
    }
    leg_types = {
        0: "vertical",
        1: "straight",
        2: "up-curved",
        3: "down-curved",
    }

    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)

        self.get_params_dict()
        # random init with seed
        with FixedSeed(self.factory_seed):
            self.width = uniform(0.4, 0.5)
            self.size = uniform(0.38, 0.45)
            self.thickness = uniform(0.04, 0.08)
            self.bevel_width = self.thickness * (0.1 if uniform() < 0.4 else 0.5)
            self.seat_back = uniform(0.7, 1.0) if uniform() < 0.75 else 1.0
            self.seat_mid = uniform(0.7, 0.8)
            self.seat_mid_x = uniform(
                self.seat_back + self.seat_mid * (1 - self.seat_back), 1
            )
            self.seat_mid_z = uniform(0, 0.5)
            self.seat_front = uniform(1.0, 1.2)
            self.is_seat_round = uniform() < 0.6
            self.is_seat_subsurf = uniform() < 0.5

            self.leg_thickness = uniform(0.04, 0.06)
            self.limb_profile = uniform(1.5, 2.5)
            self.leg_height = uniform(0.45, 0.5)
            self.back_height = uniform(0.4, 0.5)
            self.is_leg_round = uniform() < 0.5
            self.leg_type = np.random.choice(
                ["vertical", "straight", "up-curved", "down-curved"]
            )

            self.leg_x_offset = 0
            self.leg_y_offset = 0, 0
            self.back_x_offset = 0
            self.back_y_offset = 0

            self.has_leg_x_bar = uniform() < 0.6
            self.has_leg_y_bar = uniform() < 0.6
            self.leg_offset_bar = uniform(0.2, 0.4), uniform(0.6, 0.8)

            self.has_arm = uniform() < 0.7
            self.arm_thickness = uniform(0.04, 0.06)
            self.arm_height = self.arm_thickness * uniform(0.6, 1)
            self.arm_y = uniform(0.8, 1) * self.size
            self.arm_z = uniform(0.3, 0.6) * self.back_height
            self.arm_mid = np.array(
                [uniform(-0.03, 0.03), uniform(-0.03, 0.09), uniform(-0.09, 0.03)]
            )
            self.arm_profile = log_uniform(0.1, 3, 2)

            self.back_thickness = uniform(0.04, 0.05)
            self.back_type = rg(self.back_types)
            self.back_profile = [(0, 1)]
            self.back_vertical_cuts = np.random.randint(1, 4)
            self.back_partial_scale = uniform(1, 1.4)

            materials = AssetList["ChairFactory"]()
            self.limb_surface = materials["limb"].assign_material()
            self.surface = materials["surface"].assign_material()
            if uniform() < 0.3:
                self.panel_surface = self.surface
            else:
                self.panel_surface = materials["panel"].assign_material()

            scratch_prob, edge_wear_prob = materials["wear_tear_prob"]
            self.scratch, self.edge_wear = materials["wear_tear"]
            is_scratch = uniform() < scratch_prob
            is_edge_wear = uniform() < edge_wear_prob
            if not is_scratch:
                self.scratch = None
            if not is_edge_wear:
                self.edge_wear = None

            # from infinigen.assets.clothes import blanket
            # from infinigen.assets.scatters.clothes import ClothesCover
            # self.clothes_scatter = ClothesCover(factory_fn=blanket.BlanketFactory, width=log_uniform(.8, 1.2),
            #                                    size=uniform(.8, 1.2)) if uniform() < .3 else NoApply()
            self.clothes_scatter = NoApply()
            self.post_init()
    
    def get_params_dict(self):
        # all the parameters (key:name, value: [type, range]) used in this generator
        self.params_dict = {
            "width": ['continuous', [0.3, 0.8]], # seat width
            "size": ['continuous', [0.35, 0.5]], # seat length
            "thickness": ['continuous', [0.02, 0.1]], # seat thickness
            "bevel_width": ['discrete', [0.1, 0.5]],
            "seat_back": ['continuous', [0.6, 1.0]], # seat back width
            "seat_mid": ['continuous', [0.7, 0.8]],
            "seat_mid_z": ['continuous', [0.0, 0.7]], # seat mid point height
            "seat_front": ['continuous', [1.0, 1.2]], # seat front point
            "is_seat_round": ['discrete', [0, 1]],
            "is_seat_subsurf": ['discrete', [0, 1]],
            "leg_thickness": ['continuous', [0.02, 0.07]], # leg thickness
            "limb_profile": ['continuous', [1.5, 2.5]],
            "leg_height": ['continuous', [0.2, 1.0]], # leg height
            "is_leg_round": ['discrete', [0, 1]],
            "leg_type": ['discrete', [0,1,2,3]],
            "has_leg_x_bar": ['discrete', [0, 1]],
            "has_leg_y_bar": ['discrete', [0, 1]],
            "leg_offset_bar0": ['continuous', [0.1, 0.9]], # leg y bar offset, only for has_leg_y_bar is 1
            "leg_offset_bar1": ['continuous', [0.1, 0.9]], # leg x bar offset, only for has_leg_x_bar is 1
            "leg_x_offset": ['continuous', [0.0, 0.2]], # leg end point x offset
            "leg_y_offset0": ['continuous', [0.0, 0.2]],  # leg end point y offset
            "leg_y_offset1": ['continuous', [0.0, 0.2]],  # leg end point y offset
            "has_arm": ['discrete', [0, 1]],
            "arm_thickness": ['continuous', [0.02, 0.07]], # arm thickness, only for has_arm is 1
            "arm_height": ['continuous', [0.6, 1]], # only for has_arm is 1
            "arm_y": ['continuous', [0.5, 1]], # arm y end point, only for has_arm is 1
            "arm_z": ['continuous', [0.25, 0.6]], # arm z end point, only for has_arm is 1
            "arm_mid0": ['continuous', [-0.03, 0.03]], # arm mid point x coord, only for has_arm is 1
            "arm_mid1": ['continuous', [-0.03, 0.2]], # arm mid point y coord, only for has_arm is 1
            "arm_mid2": ['continuous', [-0.09, 0.03]], # arm mid point z coord, only for has_arm is 1
            "arm_profile0": ['continuous', [0.0, 2.0]], # arm curve control, only for has_arm is 1
            "arm_profile1": ['continuous', [0.0, 2]], # arm curve control, only for has_arm is 1
            "back_height": ['continuous', [0.3, 0.6]], # back height
            "back_thickness": ['continuous', [0.02, 0.07]], # back thickness
            "back_type": ['discrete', [0, 1, 2, 3]],
            "back_vertical_cuts": ['discrete', [1,2,3,4]], # only for back type 3
            "back_partial_scale": ['continuous', [1.0, 1.4]], # only for back type 1
            "back_x_offset": ['continuous', [-0.1, 0.15]], # back top x length
            "back_y_offset": ['continuous', [0.0, 0.4]], # back top y coord
            "back_profile_partial": ['continuous', [0.4, 0.8]], # only for back type 1
            "back_profile_horizontal_ncuts": ['discrete', [2, 3, 4]], # only for back type 2
            "back_profile_horizontal_locs0": ['continuous', [1, 2]], # only for back type 2
            "back_profile_horizontal_locs1": ['continuous', [1, 2]], # only for back type 2
            "back_profile_horizontal_locs2": ['continuous', [1, 2]], # only for back type 2
            "back_profile_horizontal_locs3": ['continuous', [1, 2]], # only for back type 2
            "back_profile_horizontal_ratio": ['continuous', [0.2, 0.8]], # only for back type 2
            "back_profile_horizontal_lowest": ['continuous', [0, 0.4]], # only for back type 2
            "back_profile_vertical": ['continuous', [0.8, 0.9]], # only for back type 3
        }
    
    def fix_unused_params(self, params):
        # check unused parameters inside a given parameter set, and fix them into mid value - for training
        if params['leg_type'] != 2 and params['leg_type'] != 3:
            params['limb_profile'] = (self.params_dict['limb_profile'][1][0] + self.params_dict['limb_profile'][1][-1]) / 2
        if params['has_leg_x_bar'] == 0:
            params['leg_offset_bar1'] = (self.params_dict['leg_offset_bar1'][1][0] + self.params_dict['leg_offset_bar1'][1][-1]) / 2
        if params['has_leg_y_bar'] == 0:
            params['leg_offset_bar0'] = (self.params_dict['leg_offset_bar0'][1][0] + self.params_dict['leg_offset_bar0'][1][-1]) / 2
        if params['has_arm'] == 0:
            params['arm_thickness'] = (self.params_dict['arm_thickness'][1][0] + self.params_dict['arm_thickness'][1][-1]) / 2
            params['arm_height'] = (self.params_dict['arm_height'][1][0] + self.params_dict['arm_height'][1][-1]) / 2
            params['arm_y'] = (self.params_dict['arm_y'][1][0] + self.params_dict['arm_y'][1][-1]) / 2
            params['arm_z'] = (self.params_dict['arm_z'][1][0] + self.params_dict['arm_z'][1][-1]) / 2
            params['arm_mid0'] = (self.params_dict['arm_mid0'][1][0] + self.params_dict['arm_mid0'][1][-1]) / 2
            params['arm_mid1'] = (self.params_dict['arm_mid1'][1][0] + self.params_dict['arm_mid1'][1][-1]) / 2
            params['arm_mid2'] = (self.params_dict['arm_mid2'][1][0] + self.params_dict['arm_mid2'][1][-1]) / 2
            params['arm_profile0'] = (self.params_dict['arm_profile0'][1][0] + self.params_dict['arm_profile0'][1][-1]) / 2
            params['arm_profile1'] = (self.params_dict['arm_profile1'][1][0] + self.params_dict['arm_profile1'][1][-1]) / 2
        if params['back_type'] != 3:
            params['back_vertical_cuts'] = (self.params_dict['back_vertical_cuts'][1][0] + self.params_dict['back_vertical_cuts'][1][-1]) / 2
            params['back_profile_vertical'] = (self.params_dict['back_profile_vertical'][1][0] + self.params_dict['back_profile_vertical'][1][-1]) / 2
        if params['back_type'] != 2:
            params['back_profile_horizontal_ncuts'] = (self.params_dict['back_profile_horizontal_ncuts'][1][0] + self.params_dict['back_profile_horizontal_ncuts'][1][-1]) / 2
            params['back_profile_horizontal_locs0'] = (self.params_dict['back_profile_horizontal_locs0'][1][0] + self.params_dict['back_profile_horizontal_locs0'][1][-1]) / 2
            params['back_profile_horizontal_locs1'] = (self.params_dict['back_profile_horizontal_locs1'][1][0] + self.params_dict['back_profile_horizontal_locs1'][1][-1]) / 2
            params['back_profile_horizontal_locs2'] = (self.params_dict['back_profile_horizontal_locs2'][1][0] + self.params_dict['back_profile_horizontal_locs2'][1][-1]) / 2
            params['back_profile_horizontal_ratio'] = (self.params_dict['back_profile_horizontal_ratio'][1][0] + self.params_dict['back_profile_horizontal_ratio'][1][-1]) / 2
            params['back_profile_horizontal_lowest'] = (self.params_dict['back_profile_horizontal_lowest'][1][0] + self.params_dict['back_profile_horizontal_lowest'][1][-1]) / 2
        if params['back_type'] != 1:
            params['back_partial_scale'] = (self.params_dict['back_partial_scale'][1][0] + self.params_dict['back_partial_scale'][1][-1]) / 2
            params['back_profile_partial'] = (self.params_dict['back_profile_partial'][1][0] + self.params_dict['back_profile_partial'][1][-1]) / 2
        return params
    
    def update_params(self, new_params):
        # replace the parameters and calculate all the new values
        self.width = new_params["width"]
        self.size = new_params["size"]
        self.thickness = new_params["thickness"]
        self.bevel_width = self.thickness * new_params["bevel_width"]
        self.seat_back = new_params["seat_back"]
        self.seat_mid = new_params["seat_mid"]
        self.seat_mid_x = uniform(
            self.seat_back + self.seat_mid * (1 - self.seat_back), 1
        )
        self.seat_mid_z = new_params["seat_mid_z"]
        self.seat_front = new_params["seat_front"]
        self.is_seat_round = new_params["is_seat_round"]
        self.is_seat_subsurf = new_params["is_seat_subsurf"]

        self.leg_thickness = new_params["leg_thickness"]
        self.limb_profile = new_params["limb_profile"]
        self.leg_height = new_params["leg_height"]
        self.back_height = new_params["back_height"]
        self.is_leg_round = new_params["is_leg_round"]
        self.leg_type = self.leg_types[new_params["leg_type"]]

        self.leg_x_offset = 0
        self.leg_y_offset = 0, 0
        self.back_x_offset = 0
        self.back_y_offset = 0

        self.has_leg_x_bar = new_params["has_leg_x_bar"]
        self.has_leg_y_bar = new_params["has_leg_y_bar"]
        self.leg_offset_bar = new_params["leg_offset_bar0"], new_params["leg_offset_bar1"]

        self.has_arm = new_params["has_arm"]
        self.arm_thickness = new_params["arm_thickness"]
        self.arm_height = self.arm_thickness * new_params["arm_height"]
        self.arm_y = new_params["arm_y"] * self.size
        self.arm_z = new_params["arm_z"] * self.back_height
        self.arm_mid = np.array(
            [new_params["arm_mid0"], new_params["arm_mid1"], new_params["arm_mid2"]]
        )
        self.arm_profile = (new_params["arm_profile0"], new_params["arm_profile1"])

        self.back_thickness = new_params["back_thickness"]
        self.back_type = self.back_types[new_params["back_type"]]
        self.back_profile = [(0, 1)]
        self.back_vertical_cuts = new_params["back_vertical_cuts"]
        self.back_partial_scale = new_params["back_partial_scale"]

        if self.leg_type == "vertical":
            self.leg_x_offset = 0
            self.leg_y_offset = 0, 0
            self.back_x_offset = 0
            self.back_y_offset = 0
        else:
            self.leg_x_offset = self.width * new_params["leg_x_offset"]
            self.leg_y_offset = self.size * np.array([new_params["leg_y_offset0"], new_params["leg_y_offset1"]])
            self.back_x_offset = self.width * new_params["back_x_offset"]
            self.back_y_offset = self.size * new_params["back_y_offset"]

        match self.back_type:
            case "partial":
                self.back_profile = ((new_params["back_profile_partial"], 1),)
            case "horizontal-bar":
                n_cuts = int(new_params["back_profile_horizontal_ncuts"])
                locs = np.array([new_params["back_profile_horizontal_locs0"], new_params["back_profile_horizontal_locs1"],
                                    new_params["back_profile_horizontal_locs2"], new_params["back_profile_horizontal_locs3"]])[:n_cuts].cumsum()
                locs = locs / locs[-1]
                ratio = new_params["back_profile_horizontal_ratio"]
                locs = np.array(
                    [
                        (p + ratio * (l - p), l)
                        for p, l in zip([0, *locs[:-1]], locs)
                    ]
                )
                lowest = new_params["back_profile_horizontal_lowest"]
                self.back_profile = locs * (1 - lowest) + lowest
            case "vertical-bar":
                self.back_profile = ((new_params["back_profile_vertical"], 1),)
            case _:
                self.back_profile = [(0, 1)]

        # TODO: handle the material into the optimization loop
        materials = AssetList["ChairFactory"]()
        self.limb_surface = materials["limb"].assign_material()
        self.surface = materials["surface"].assign_material()
        if uniform() < 0.3:
            self.panel_surface = self.surface
        else:
            self.panel_surface = materials["panel"].assign_material()

        scratch_prob, edge_wear_prob = materials["wear_tear_prob"]
        self.scratch, self.edge_wear = materials["wear_tear"]
        is_scratch = uniform() < scratch_prob
        is_edge_wear = uniform() < edge_wear_prob
        if not is_scratch:
            self.scratch = None
        if not is_edge_wear:
            self.edge_wear = None

        # from infinigen.assets.clothes import blanket
        # from infinigen.assets.scatters.clothes import ClothesCover
        # self.clothes_scatter = ClothesCover(factory_fn=blanket.BlanketFactory, width=log_uniform(.8, 1.2),
        #                                    size=uniform(.8, 1.2)) if uniform() < .3 else NoApply()
        self.clothes_scatter = NoApply()
        

    def post_init(self):
        with FixedSeed(self.factory_seed):
            if self.leg_type == "vertical":
                self.leg_x_offset = 0
                self.leg_y_offset = 0, 0
                self.back_x_offset = 0
                self.back_y_offset = 0
            else:
                self.leg_x_offset = self.width * uniform(0.05, 0.2)
                self.leg_y_offset = self.size * uniform(0.05, 0.2, 2)
                self.back_x_offset = self.width * uniform(-0.1, 0.15)
                self.back_y_offset = self.size * uniform(0.1, 0.25)

            match self.back_type:
                case "partial":
                    self.back_profile = ((uniform(0.4, 0.8), 1),)
                case "horizontal-bar":
                    n_cuts = np.random.randint(2, 4)
                    locs = uniform(1, 2, n_cuts).cumsum()
                    locs = locs / locs[-1]
                    ratio = uniform(0.5, 0.75)
                    locs = np.array(
                        [
                            (p + ratio * (l - p), l)
                            for p, l in zip([0, *locs[:-1]], locs)
                        ]
                    )
                    lowest = uniform(0, 0.4)
                    self.back_profile = locs * (1 - lowest) + lowest
                case "vertical-bar":
                    self.back_profile = ((uniform(0.8, 0.9), 1),)
                case _:
                    self.back_profile = [(0, 1)]

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        obj = new_bbox(
            -self.width / 2 - max(self.leg_x_offset, self.back_x_offset),
            self.width / 2 + max(self.leg_x_offset, self.back_x_offset),
            -self.size - self.leg_y_offset[1] - self.leg_thickness * 0.5,
            max(self.leg_y_offset[0], self.back_y_offset),
            -self.leg_height,
            self.back_height * 1.2,
        )
        obj.rotation_euler.z += np.pi / 2
        butil.apply_transform(obj)
        return obj

    def create_asset(self, **params) -> bpy.types.Object:
        obj = self.make_seat()
        legs = self.make_legs()
        backs = self.make_backs()

        parts = [obj] + legs + backs
        parts.extend(self.make_leg_decors(legs))
        if self.has_arm:
            parts.extend(self.make_arms(obj, backs))
        parts.extend(self.make_back_decors(backs))

        for obj in legs:
            self.solidify(obj, 2)
        for obj in backs:
            self.solidify(obj, 2, self.back_thickness)

        obj = join_objects(parts)
        obj.rotation_euler.z += np.pi / 2
        butil.apply_transform(obj)

        with FixedSeed(self.factory_seed):
            # TODO: wasteful to create unique materials for each individual asset
            self.surface.apply(obj)
            self.panel_surface.apply(obj, selection="panel")
            self.limb_surface.apply(obj, selection="limb")

        return obj

    def finalize_assets(self, assets):
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)

    def make_seat(self):
        x_anchors = (
            np.array(
                [
                    0,
                    -self.seat_back,
                    -self.seat_mid_x,
                    -1,
                    0,
                    1,
                    self.seat_mid_x,
                    self.seat_back,
                    0,
                ]
            )
            * self.width
            / 2
        )
        y_anchors = (
            np.array(
                [0, 0, -self.seat_mid, -1, -self.seat_front, -1, -self.seat_mid, 0, 0]
            )
            * self.size
        )
        z_anchors = (
            np.array([0, 0, self.seat_mid_z, 0, 0, 0, self.seat_mid_z, 0, 0])
            * self.thickness
        )
        vector_locations = [1, 7] if self.is_seat_round else [1, 3, 5, 7]
        obj = bezier_curve((x_anchors, y_anchors, z_anchors), vector_locations, 8)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.fill_grid(use_interp_simple=True)
        butil.modify_mesh(obj, "SOLIDIFY", thickness=self.thickness, offset=0)
        subsurf(obj, 1, not self.is_seat_subsurf)
        butil.modify_mesh(obj, "BEVEL", width=self.bevel_width, segments=8)
        return obj

    def make_legs(self):
        leg_starts = np.array(
            [[-self.seat_back, 0, 0], [-1, -1, 0], [1, -1, 0], [self.seat_back, 0, 0]]
        ) * np.array([[self.width / 2, self.size, 0]])
        leg_ends = leg_starts.copy()
        leg_ends[[0, 1], 0] -= self.leg_x_offset
        leg_ends[[2, 3], 0] += self.leg_x_offset
        leg_ends[[0, 3], 1] += self.leg_y_offset[0]
        leg_ends[[1, 2], 1] -= self.leg_y_offset[1]
        leg_ends[:, -1] = -self.leg_height
        return self.make_limb(leg_ends, leg_starts)

    def make_limb(self, leg_ends, leg_starts):
        limbs = []
        for leg_start, leg_end in zip(leg_starts, leg_ends):
            match self.leg_type:
                case "up-curved":
                    axes = [(0, 0, 1), None]
                    scale = [self.limb_profile, 1]
                case "down-curved":
                    axes = [None, (0, 0, 1)]
                    scale = [1, self.limb_profile]
                case _:
                    axes = None
                    scale = None
            limb = align_bezier(
                np.stack([leg_start, leg_end], -1), axes, scale, resolution=64
            )
            limb.location = (
                np.array(
                    [
                        1 if leg_start[0] < 0 else -1,
                        1 if leg_start[1] < -self.size / 2 else -1,
                        0,
                    ]
                )
                * self.leg_thickness
                / 2
            )
            butil.apply_transform(limb, True)
            limbs.append(limb)
        return limbs

    def make_backs(self):
        back_starts = (
            np.array([[-self.seat_back, 0, 0], [self.seat_back, 0, 0]]) * self.width / 2
        )
        back_ends = back_starts.copy()
        back_ends[:, 0] += np.array([self.back_x_offset, -self.back_x_offset])
        back_ends[:, 1] = self.back_y_offset
        back_ends[:, 2] = self.back_height
        return self.make_limb(back_starts, back_ends)

    def make_leg_decors(self, legs):
        decors = []
        if self.has_leg_x_bar:
            z_height = -self.leg_height * uniform(*self.leg_offset_bar)
            locs = []
            for leg in legs:
                co = read_co(leg)
                locs.append(co[np.argmin(np.abs(co[:, -1] - z_height))])
            decors.append(
                self.solidify(bezier_curve(np.stack([locs[0], locs[3]], -1)), 0)
            )
            decors.append(
                self.solidify(bezier_curve(np.stack([locs[1], locs[2]], -1)), 0)
            )
        if self.has_leg_y_bar:
            z_height = -self.leg_height * uniform(*self.leg_offset_bar)
            locs = []
            for leg in legs:
                co = read_co(leg)
                locs.append(co[np.argmin(np.abs(co[:, -1] - z_height))])
            decors.append(
                self.solidify(bezier_curve(np.stack([locs[0], locs[1]], -1)), 1)
            )
            decors.append(
                self.solidify(bezier_curve(np.stack([locs[2], locs[3]], -1)), 1)
            )
        for d in decors:
            write_attribute(d, 1, "limb", "FACE")
        return decors

    def make_back_decors(self, backs, finalize=True):
        obj = join_objects([deep_clone_obj(b) for b in backs])
        x, y, z = read_co(obj).T
        x += np.where(x > 0, self.back_thickness / 2, -self.back_thickness / 2)
        write_co(obj, np.stack([x, y, z], -1))
        smoothness = uniform(0, 1)
        profile_shape_factor = uniform(0, 0.4)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            center = read_edge_center(obj)
            for z_min, z_max in self.back_profile:
                select_edges(
                    obj,
                    (z_min * self.back_height <= center[:, -1])
                    & (center[:, -1] <= z_max * self.back_height),
                )
                bpy.ops.mesh.bridge_edge_loops(
                    number_cuts=32,
                    interpolation="LINEAR",
                    smoothness=smoothness,
                    profile_shape_factor=profile_shape_factor,
                )
            bpy.ops.mesh.select_loose()
            bpy.ops.mesh.delete()
        butil.modify_mesh(
            obj,
            "SOLIDIFY",
            thickness=np.minimum(self.thickness, self.back_thickness),
            offset=0,
        )
        if finalize:
            butil.modify_mesh(obj, "BEVEL", width=self.bevel_width, segments=8)
        parts = [obj]
        if self.back_type == "vertical-bar":
            other = join_objects([deep_clone_obj(b) for b in backs])
            with butil.ViewportMode(other, "EDIT"):
                bpy.ops.mesh.select_mode(type="EDGE")
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.bridge_edge_loops(
                    number_cuts=self.back_vertical_cuts,
                    interpolation="LINEAR",
                    smoothness=smoothness,
                    profile_shape_factor=profile_shape_factor,
                )
                bpy.ops.mesh.select_all(action="INVERT")
                bpy.ops.mesh.delete()
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.delete(type="ONLY_FACE")
            remove_edges(other, np.abs(read_edge_direction(other)[:, -1]) < 0.5)
            remove_vertices(other, lambda x, y, z: z < -self.thickness / 2)
            remove_vertices(
                other,
                lambda x, y, z: z
                > (self.back_profile[0][0] + self.back_profile[0][1])
                * self.back_height
                / 2,
            )
            parts.append(self.solidify(other, 2, self.back_thickness))
        elif self.back_type == "partial":
            co = read_co(obj)
            co[:, 1] *= self.back_partial_scale
            write_co(obj, co)
        for p in parts:
            write_attribute(p, 1, "panel", "FACE")
        return parts

    def make_arms(self, base, backs):
        co = read_co(base)
        end = co[np.argmin(co[:, 0] - (np.abs(co[:, 1] + self.arm_y) < 0.02))]
        end[0] += self.arm_thickness / 4
        end_ = end.copy()
        end_[0] = -end[0]
        arms = []
        co = read_co(backs[0])
        start = co[np.argmin(co[:, 0] - (np.abs(co[:, -1] - self.arm_z) < 0.02))]
        start[0] -= self.arm_thickness / 4
        start_ = start.copy()
        start_[0] = -start[0]
        for start, end in zip([start, start_], [end, end_]):
            mid = np.array(
                [
                    end[0] + self.arm_mid[0] * (-1 if end[0] > 0 else 1),
                    end[1] + self.arm_mid[1],
                    start[2] + self.arm_mid[2],
                ]
            )
            arm = align_bezier(
                np.stack([start, mid, end], -1),
                np.array(
                    [
                        [end[0] - start[0], end[1] - start[1], 0],
                        [0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
                        [0, 0, 1],
                    ]
                ),
                [1, *self.arm_profile, 1],
            )
            if self.is_leg_round:
                surface.add_geomod(
                    arm,
                    geo_radius,
                    apply=True,
                    input_args=[self.arm_thickness / 2, 32],
                    input_kwargs={"to_align_tilt": False},
                )
            else:
                with butil.ViewportMode(arm, "EDIT"):
                    bpy.ops.mesh.select_all(action="SELECT")
                    bpy.ops.mesh.extrude_edges_move(
                        TRANSFORM_OT_translate={
                            "value": (
                                self.arm_thickness
                                if end[0] < 0
                                else -self.arm_thickness,
                                0,
                                0,
                            )
                        }
                    )
                butil.modify_mesh(arm, "SOLIDIFY", thickness=self.arm_height, offset=0)
            write_attribute(arm, 1, "limb", "FACE")
            arms.append(arm)
        return arms

    def solidify(self, obj, axis, thickness=None):
        if thickness is None:
            thickness = self.leg_thickness
        if self.is_leg_round:
            solidify(obj, axis, thickness)
            butil.modify_mesh(obj, "BEVEL", width=self.bevel_width, segments=8)
        else:
            surface.add_geomod(
                obj, geo_radius, apply=True, input_args=[thickness / 2, 32]
            )
        write_attribute(obj, 1, "limb", "FACE")
        return obj
