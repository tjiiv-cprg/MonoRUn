from mmcv.utils import Registry, build_from_cfg

DIM_CODERS = Registry('dim_coder')
PROJ_ERROR_CODERS = Registry('proj_error_coder')
ROTATION_CODERS = Registry('rotation_coder')
IOU3D_SAMPLERS = Registry('iou3d_sampler')
COORD_CODERS = Registry('coord_coder')


def build_dim_coder(cfg, **default_args):
    return build_from_cfg(cfg, DIM_CODERS, default_args)

def build_proj_error_coder(cfg, **default_args):
    return build_from_cfg(cfg, PROJ_ERROR_CODERS, default_args)

def build_rotation_coder(cfg, **default_args):
    return build_from_cfg(cfg, ROTATION_CODERS, default_args)

def build_iou3d_sampler(cfg, **default_args):
    return build_from_cfg(cfg, IOU3D_SAMPLERS, default_args)

def build_coord_coder(cfg, **default_args):
    return build_from_cfg(cfg, COORD_CODERS, default_args)
