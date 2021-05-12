import torch
from ..builder import COORD_CODERS


@COORD_CODERS.register_module()
class NOCCoder(object):

    def __init__(self,
                 target_means=(-0.1, -0.5, 0.0),
                 target_stds=(0.35, 0.23, 0.34),
                 eps=1e-5):
        super(NOCCoder, self).__init__()
        self.target_means = target_means
        self.target_stds = target_stds
        self.eps = eps

    def encode(self, gt_coords_3d, gt_coords_3d_mask, dimensions, flip):
        """
        Args:
            gt_coords_3d (torch.Tensor): Shape (n, 3, h, w), masked coords
            gt_coords_3d_mask (torch.Tensor)： Shape (n, 1, h, w)
            dimensions (torch.Tensor): Shape (n, 3), [length, width, height]
            flip (bool)

        Returns:
            torch.Tensor: Shape (n, 3, h, w), masked normalized part encoding
            torch.Tensor
        """
        foreground = gt_coords_3d_mask >= self.eps

        # unmask and encode (avoid inf / nan)
        parts = (gt_coords_3d / gt_coords_3d_mask.clamp(min=self.eps)
                 / dimensions.clamp(min=self.eps)[..., None, None])
        # clone mask and set below-threshold region to 0
        parts_mask = gt_coords_3d_mask.clone()
        parts_mask[~foreground] = 0

        if flip:
            parts[:, 2].neg_()

        target_means = parts.new_tensor(
            self.target_means)[:, None, None]
        target_stds = parts.new_tensor(
            self.target_stds)[:, None, None]
        parts.sub_(target_means).div_(target_stds)  # normalize
        parts *= parts_mask  # mask encoding

        return parts, parts_mask

    def decode(self, part, part_var, dimensions, dimensions_var, flip):
        dimensions = dimensions[..., None, None]
        if dimensions_var is not None:
            dimensions_var = dimensions_var[..., None, None]

        target_means = part.new_tensor(
            self.target_means)[:, None, None]
        target_stds = part.new_tensor(
            self.target_stds)[:, None, None]
        part_norm = part * target_stds + target_means
        coords_3d = part_norm * dimensions

        if part_var is not None:
            part_norm_var = part_var * target_stds.square()
            coords_3d_var = part_norm_var * dimensions.square()
            if dimensions_var is not None:
                coords_3d_var += dimensions_var * part_norm.square() \
                                 + part_norm_var * dimensions_var
        elif dimensions_var is not None:
            coords_3d_var = dimensions_var * part_norm.square()
        else:
            coords_3d_var = None

        return coords_3d, coords_3d_var
