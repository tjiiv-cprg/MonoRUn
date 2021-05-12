import torch
from ..builder import PROJ_ERROR_CODERS


@PROJ_ERROR_CODERS.register_module()
class DistanceInvarProjErrorCoder(object):

    def __init__(self,
                 ref_length=1.6,
                 ref_focal_y=722,
                 target_std=0.25,
                 distance_min=0.1,
                 epistemic_std_gain=1.0):
        super(DistanceInvarProjErrorCoder, self).__init__()
        self.scaling_denomitor = ref_length * ref_focal_y * target_std
        self.ref_focal_y = ref_focal_y
        self.distance_min = distance_min
        self.epistemic_std_gain = epistemic_std_gain

    def encode(self, coords_2d_diff_std, distance):
        """
        Args:
            coords_2d_diff_std (torch.Tensor): Shape (N, C, H, W)
            distance (torch.Tensor): Shape (N, 1), usually gt distance

        Returns:
            torch.Tensor: Encoded projection error or std
        """
        proj_error_std = coords_2d_diff_std * (
            distance[..., None, None] / self.scaling_denomitor)
        return proj_error_std

    def decode(self, proj_error_std, distance):
        coords_2d_diff_std = proj_error_std * (
            self.scaling_denomitor
            / distance[..., None, None].clamp(min=self.distance_min))
        return coords_2d_diff_std

    def decode_logstd(self, proj_logstd, coords_3d_var, distance):
        distance_ = distance[..., None, None].clamp(min=self.distance_min) \
            if distance is not None \
            else proj_logstd.new_tensor([self.scaling_denomitor])
        if coords_3d_var is not None:
            n, _, h, w = coords_3d_var.shape
            coords_2d_var = coords_3d_var.new_empty((n, 2, h, w))
            coords_2d_var[:, 0] = \
                0.5 * (coords_3d_var[:, 0] + coords_3d_var[:, 2])
            coords_2d_var[:, 1] = coords_3d_var[:, 1]

            coords_2d_var = (
                coords_2d_var * (self.ref_focal_y * self.epistemic_std_gain) ** 2
                + (2 * proj_logstd).exp() * self.scaling_denomitor ** 2
            ) / distance_.square()
            coords_2d_logstd = 0.5 * torch.log(coords_2d_var)

        else:
            coords_2d_logstd = proj_logstd + torch.log(
                self.scaling_denomitor / distance_)

        return coords_2d_logstd

    def cov_correction(self, cov, distance):
        return cov * (self.scaling_denomitor / distance).square().view(-1, 1, 1)

