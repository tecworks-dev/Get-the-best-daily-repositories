import os

current_dir_path = os.path.dirname(__file__)

SMPLX_DIR = f"{current_dir_path}/../checkpoints"
MEAN_PARAMS = f"{current_dir_path}/../checkpoints/smpl_mean_params.npz"
CACHE_DIR_MULTIHMR = f"{current_dir_path}/../checkpoints/multiHMR"


SMPLX2SMPL_REGRESSOR = f"{current_dir_path}/../checkpoints/smplx/smplx2smpl.pkl"

DEVICE = "cuda"
MODEL_NAME = 'ABCGSUR8'
KEYPOINT_THR = 0.5
