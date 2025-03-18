# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import importlib
import os
from abc import ABC
from functools import partial
from typing import Callable, List

import numpy as np

YELLOW = np.array([1, 0.706, 0])
RED = np.array([128, 0, 0]) / 255.0
BLACK = np.array([0, 0, 0]) / 255.0
BLUE = np.array([0.4, 0.5, 0.9])
GREEN = np.array([0.4, 0.9, 0.5])
SPHERE_SIZE_KEYPOSES = 1.0
SPHERE_SIZE_ODOMETRY = 0.2


def transform_points(pcd, T):
    R = T[:3, :3]
    t = T[:3, -1]
    return pcd @ R.T + t


class StubVisualizer(ABC):
    def __init__(self):
        pass

    def update(self, slam):
        pass


class RegistrationVisualizer(StubVisualizer):
    # Public Interaface ----------------------------------------------------------------------------
    def __init__(self):
        try:
            self.o3d = importlib.import_module("open3d")
        except ModuleNotFoundError as err:
            print(f'open3d is not installed on your system, run "pip install open3d"')
            exit(1)

        # Initialize GUI controls
        self.block_vis = True
        self.play_crun = False
        self.reset_bounding_box = True

        # Create data
        self.local_map = self.o3d.geometry.PointCloud()
        self.closures = []
        self.key_poses = []
        self.key_frames = []
        self.global_frames = []
        self.odom_frames = []
        self.edges = []
        self.current_node = None

        # Initialize visualizer
        self.vis = self.o3d.visualization.VisualizerWithKeyCallback()
        self._register_key_callbacks()
        self._initialize_visualizer()

    def update(self, slam):
        self._update_geometries(slam)
        while self.block_vis:
            self.vis.poll_events()
            self.vis.update_renderer()
            if self.play_crun:
                break
        self.block_vis = not self.block_vis

    # Private Interaface ---------------------------------------------------------------------------
    def _initialize_visualizer(self):
        w_name = self.__class__.__name__
        self.vis.create_window(window_name=w_name, width=1920, height=1080)
        self.vis.add_geometry(self.local_map)
        self._set_black_background(self.vis)
        self.vis.get_render_option().point_size = 1
        self.vis.get_render_option().line_width = 10
        print(
            f"{w_name} initialized. Press:\n"
            "\t[SPACE] to pause/start\n"
            "\t  [ESC] to exit\n"
            "\t    [N] to step\n"
            "\t    [C] to center the viewpoint\n"
            "\t    [W] to toggle a white background\n"
            "\t    [B] to toggle a black background\n"
        )

    def _register_key_callback(self, keys: List, callback: Callable):
        for key in keys:
            self.vis.register_key_callback(ord(str(key)), partial(callback))

    def _register_key_callbacks(self):
        self._register_key_callback(["Ā", "Q", "\x1b"], self._quit)
        self._register_key_callback([" "], self._start_stop)
        self._register_key_callback(["N"], self._next_frame)
        self._register_key_callback(["C"], self._center_viewpoint)
        self._register_key_callback(["B"], self._set_black_background)
        self._register_key_callback(["W"], self._set_white_background)

    def _set_black_background(self, vis):
        vis.get_render_option().background_color = [0.0, 0.0, 0.0]

    def _set_white_background(self, vis):
        vis.get_render_option().background_color = [1.0, 1.0, 1.0]

    def _quit(self, vis):
        print("Destroying Visualizer")
        vis.destroy_window()
        os._exit(0)

    def _next_frame(self, vis):
        self.block_vis = not self.block_vis

    def _start_stop(self, vis):
        self.play_crun = not self.play_crun

    def _center_viewpoint(self, vis):
        vis.reset_view_point(True)

    def _add_line(self, pose0, pose1, color):
        lines = [[0, 1]]
        colors = [color for i in range(len(lines))]
        line_set_closure = self.o3d.geometry.LineSet()
        line_set_closure.points = self.o3d.utility.Vector3dVector([pose0, pose1])
        line_set_closure.lines = self.o3d.utility.Vector2iVector(lines)
        line_set_closure.colors = self.o3d.utility.Vector3dVector(colors)
        return line_set_closure

    def _add_frames(self, poses, size, color):
        frames = []
        for pose in poses:
            new_frame = self._add_frame(pose, size, color)
            frames.append(new_frame)
        return frames

    def _add_frame(self, pose, size, color):
        new_frame = self.o3d.geometry.TriangleMesh.create_sphere(size)
        new_frame.paint_uniform_color(color)
        new_frame.compute_vertex_normals()
        new_frame.transform(pose)
        return new_frame

    def _update_geometries(self, slam):
        current_node = slam.local_map_graph.last_local_map
        local_map_in_global = transform_points(slam.voxel_grid.point_cloud(), current_node.keypose)
        self.local_map.points = self.o3d.utility.Vector3dVector(local_map_in_global)
        self.local_map.paint_uniform_color(YELLOW)
        self.vis.update_geometry(self.local_map)

        # Odometry in current local map
        current_pose = current_node.endpose

        odom_frame = self._add_frame(current_pose, SPHERE_SIZE_ODOMETRY, GREEN)
        self.odom_frames.append(odom_frame)
        self.vis.add_geometry(
            odom_frame,
            reset_bounding_box=False,
        )

        # Optimized poses
        key_poses = slam.get_keyposes()
        if key_poses != self.key_poses:
            for frame in self.odom_frames:
                self.vis.remove_geometry(frame, reset_bounding_box=False)
            self.odom_frames = self._add_frames(slam.poses, SPHERE_SIZE_ODOMETRY, GREEN)
            for frame in self.odom_frames:
                self.vis.add_geometry(
                    frame,
                    reset_bounding_box=False,
                )

            # Vertices
            for frame in self.key_frames:
                self.vis.remove_geometry(frame, reset_bounding_box=False)
            self.key_frames = self._add_frames(key_poses, SPHERE_SIZE_KEYPOSES, BLUE)
            for frame in self.key_frames:
                self.vis.add_geometry(frame, reset_bounding_box=False)
            self.key_poses = key_poses

            # Edges
            for edge in self.edges:
                self.vis.remove_geometry(edge, reset_bounding_box=False)

            self.edges = []
            for frame0, frame1 in zip(self.key_frames[:-1], self.key_frames[1:]):
                pose0 = frame0.get_center()
                pose1 = frame1.get_center()
                self.edges.append(self._add_line(pose0, pose1, BLUE))
            for closure in self.closures:
                idx0, idx1 = closure
                pose0 = self.key_frames[idx0].get_center()
                pose1 = self.key_frames[idx1].get_center()
                self.edges.append(self._add_line(pose0, pose1, RED))

            for edge in self.edges:
                self.vis.add_geometry(edge, reset_bounding_box=False)

        if self.reset_bounding_box:
            self.vis.reset_view_point(True)
            self.reset_bounding_box = False
