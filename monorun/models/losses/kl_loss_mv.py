import torch
import torch.nn as nn

from mmdet.models import LOSSES, weighted_loss


@weighted_loss
def kl_loss_mv(pred, target, inv_cov=None):
    assert inv_cov is not None and pred.shape[:-1] == inv_cov.shape[:-2]
    if isinstance(target, int):
        if target == 0:
            diff = pred
        else:
            raise ValueError
    else:
        assert pred.size() == target.size() and target.numel() > 0
        diff = pred - target  # (*, n)
    logdet = torch.logdet(inv_cov)
    # (*, 1, 1) = (*, 1, n) @ (*, n, n) @ (*, n, 1)
    diff_weighted = diff.unsqueeze(-2) @ inv_cov @ diff.unsqueeze(-1)
    # (*) = (*) + (*)
    loss = (diff_weighted.flatten() - logdet) / 2
    loss[torch.isnan(logdet) | torch.isinf(logdet)] = 0
    return loss.unsqueeze(-1)


@LOSSES.register_module()
class KLLossMV(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(KLLossMV, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                inv_cov=None,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * kl_loss_mv(
            pred,       # (*, n)
            target,     # (*, n)
            weight,     # (*, 1)  last dim only for broadcasting
            inv_cov=inv_cov,    # (*, n, n)
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss
