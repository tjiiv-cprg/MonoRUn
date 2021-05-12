import torch
from ..builder import ROTATION_CODERS


@ROTATION_CODERS.register_module()
class Vec2DRotationCoder(object):

    def __init__(self):
        super(Vec2DRotationCoder, self).__init__()

    @staticmethod
    def encode(angles):
        if len(angles.shape) == 1:
            angles.unsqueeze_(-1)   # make new dim for cat
        vecs = torch.cat(
            (torch.cos(angles),
             torch.sin(angles)), dim=-1)
        return vecs

    @staticmethod
    def decode(vecs):
        raise NotImplementedError
