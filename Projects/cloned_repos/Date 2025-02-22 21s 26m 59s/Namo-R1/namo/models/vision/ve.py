from namo.models.vision.ve_aim import AimV2VE
from namo.models.vision.ve_siglip_navit import SigLipNavitVE
from torch import nn


def get_ve(config, **kwargs):
    type_name = config.vision_config._name_or_path.lower()

    if "siglip" in type_name and "navit" not in type_name:
        return SigLipNavitVE(config, **kwargs)
    elif "aim" in type_name:
        return AimV2VE(config, **kwargs)
    else:
        raise ValueError(f"Unsupported vision model: {type_name}")
