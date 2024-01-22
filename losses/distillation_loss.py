import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(
      self,
      base_criterion: nn.Module,
      teacher: nn.Module,
      distillation_type: str,
      alpha: float,
      tau: float,
    ) -> None:
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher = teacher
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau
    
    def forward(self, inputs: torch.Tensor, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss
        
        # we don't backprop through the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher(inputs)
        
        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
              F.log_softmax(outputs / T, dim=1),
              # we provide the teacher's targets in log probability because we use `log_target=True`
              # (as recommended in pytorch
              # https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
              # but it is possible to give just the probabilities and set `log_target=False`.
              # in our experiments we tried both.
              F.log_softmax(teacher_outputs / T, dim=1),
              reduction='sum',
              log_target=True
            ) * T ** 2 / outputs.numel()
            # we divide by outputs_kd.numel() to have the legacy PyTorch behavior.
            # but we also experiments outputs_kd.size(0)
            # see issue [61](https://github.com/facebookresearch/deit/issues/61) for more details.
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs, teacher_outputs.argmax(dim=1))
        
        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss
