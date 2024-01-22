import math

from torch.optim import Optimizer


class CosineWDSchedule:
    def __init__(self, optimizer: Optimizer, t_max: int, eta_min: float = 0, last_epoch: int = -1) -> None:
        self.last_epoch = last_epoch
        self.base_wds = [group['weight_decay'] for group in optimizer.param_groups]
        self.t_max = t_max
        self.eta_min = eta_min
    
    def _get_wd(self, optimizer: Optimizer) -> list:
        if self.last_epoch == 0:
            return self.base_wds
        elif (self.last_epoch - 1 - self.t_max) % (2 * self.t_max) == 0:
            return [
              group['weight_decay'] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.t_max)) / 2
              for base_lr, group in zip(self.base_wds, optimizer.param_groups)
            ]
        return [(1 + math.cos(math.pi * self.last_epoch / self.t_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.t_max)) * (group['weight_decay'] - self.eta_min) +
                self.eta_min for group in optimizer.param_groups]
    
    def update_weight_decay(self, optimizer: Optimizer) -> None:
        self.last_epoch += 1
        values = self._get_wd(optimizer)
        for data in zip(optimizer.param_groups, values):
            param_group, wd = data
            # avoid updating weight decay or param_groups that should not be decayed
            if param_group['weight_decay'] > 0.:
                param_group['weight_decay'] = wd
