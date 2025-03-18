# MIT License
#
# Copyright (c) 2025 Tiziano Guadagnino, Benedikt Mersch, Saurabh Gupta, Cyrill
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
import numpy as np
import open3d as o3d
from map_closures.map_closures import MapClosures

from kiss_slam.config import LoopCloserConfig
from kiss_slam.local_map_graph import LocalMapGraph
from kiss_slam.voxel_map import VoxelMap


class LoopCloser:
    def __init__(self, config: LoopCloserConfig):
        self.config = config
        self.detector = MapClosures(config.detector)
        self.local_map_voxel_size = config.detector.density_map_resolution
        self.icp_threshold = np.sqrt(3) * self.local_map_voxel_size
        self.icp_algorithm = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()
        self.termination_criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(
            relative_rmse=1e-4
        )
        self.overlap_threshold = config.overlap_threshold

    def compute(self, query_id, points, local_map_graph: LocalMapGraph):
        closure = self.detector.get_best_closure(query_id, points)
        is_good = False
        ref_id = -1
        pose_constraint = np.eye(4)
        if closure.number_of_inliers >= self.config.detector.inliers_threshold:
            ref_id = closure.source_id
            source = local_map_graph[ref_id].pcd
            target = local_map_graph[query_id].pcd
            print("\nKissSLAM| Closure Detected")
            is_good, pose_constraint = self.validate_closure(source, target, closure.pose)
        return is_good, ref_id, query_id, pose_constraint

    # This is the thing that takes the most time
    def validate_closure(self, source, target, initial_guess):
        registration_result = o3d.t.pipelines.registration.icp(
            source,
            target,
            self.icp_threshold,
            initial_guess,
            self.icp_algorithm,
            self.termination_criteria,
        )
        union_map = VoxelMap(self.local_map_voxel_size)
        source_pts = source.point.positions.numpy().astype(np.float64)
        target_pts = target.point.positions.numpy().astype(np.float64)
        pose = registration_result.transformation.numpy()
        union_map.integrate_frame(source_pts, pose)
        num_source_voxels = union_map.num_voxels()
        num_target_voxels = len(target_pts)
        union_map.add_points(target_pts)
        union = union_map.num_voxels()
        intersection = num_source_voxels + num_target_voxels - union
        overlap = intersection / np.min([num_source_voxels, num_target_voxels])
        closure_is_accepted = overlap > self.overlap_threshold
        print(f"KissSLAM| LocalMaps Overlap: {overlap}")
        if closure_is_accepted:
            print("KissSLAM| Closure Accepted")
        else:
            print(f"KissSLAM| Closure rejected for low overlap.")
        return closure_is_accepted, pose
