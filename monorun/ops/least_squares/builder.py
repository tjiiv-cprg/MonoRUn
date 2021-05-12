from mmcv.utils import Registry, build_from_cfg

PNP = Registry('pnp')


def build_pnp(cfg, **default_args):
    return build_from_cfg(cfg, PNP, default_args)
