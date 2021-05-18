import torch
import torch.nn as nn
from mmdet.models import LOSSES, weighted_loss


@weighted_loss
def robust_kl_loss(pred, target, logstd=None, delta=1.414,
                   momentum=1.0, mean_inv_std=None, eps=1e-4, training=True):
    assert logstd is not None \
        and pred.size() == logstd.size()
    if isinstance(target, int):
        if target == 0:
            diff = torch.abs(pred)
        elif target == -1:
            diff = pred
        else:
            raise ValueError
    else:
        diff = torch.abs(pred - target)
    inverse_std = torch.exp(-logstd).clamp(max=1/eps)
    diff_weighted = diff * inverse_std
    loss = torch.where(diff_weighted < delta,
                       0.5 * torch.square(diff_weighted),
                       delta * (diff_weighted - 0.5 * delta)) + logstd
    if training:
        mean_inv_std *= 1 - momentum
        mean_inv_std += momentum * torch.mean(inverse_std.detach())
    loss.div_(mean_inv_std.clamp(min=1e-6))
    return loss


@LOSSES.register_module()
class RobustKLLoss(nn.Module):

    def __init__(self, delta=1.414, reduction='mean',
                 loss_weight=1.0, momentum=1.0, eps=1e-4):
        super(RobustKLLoss, self).__init__()
        self.delta = delta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('mean_inv_std', torch.tensor(1, dtype=torch.float))

    def forward(self,
                pred,
                target,
                logstd=None,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * robust_kl_loss(
            pred,
            target,
            weight,
            logstd=logstd,
            delta=self.delta,
            momentum=self.momentum,
            mean_inv_std=self.mean_inv_std,
            eps=self.eps,
            training=self.training,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox
