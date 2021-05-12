import functools
from mmcv.runner.hooks.hook import HOOKS, Hook
from mmdet.models.builder import build_loss


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


@HOOKS.register_module()
class LossUpdaterHook(Hook):
    """
    Args:
        step (list[int])
        loss_cfgs (list[list[dict]])
        by_epoch (bool)
    """

    def __init__(self, step, loss_cfgs, by_epoch=True):
        self.by_epoch = by_epoch
        assert isinstance(step, list) and isinstance(loss_cfgs, list)
        self.step = step
        self.loss_cfgs = loss_cfgs
        self.current_step_id = 0

    def get_step_id(self, runner):
        progress = runner.epoch if self.by_epoch else runner.iter
        step_id = len(self.step)
        for i, s in enumerate(self.step):
            if progress < s:
                step_id = i
                break
        if step_id > self.current_step_id:  # step forward
            self.set_loss(runner, step_id)
            self.current_step_id = step_id

    def set_loss(self, runner, step_id):
        loss_cfgs = self.loss_cfgs[step_id - 1]
        for loss in loss_cfgs:
            attr = loss.pop('attr')
            rsetattr(runner.model.module, attr,
                     build_loss(loss) if loss['type'] is not None else None)

    def before_train_iter(self, runner):
        if not self.by_epoch:
            self.get_step_id(runner)

    def before_train_epoch(self, runner):
        if self.by_epoch:
            self.get_step_id(runner)
