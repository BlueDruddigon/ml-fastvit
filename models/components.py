import copy
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MobileOneBlock', 'ReparamLargeKernelConv']


class SEBlock(nn.Module):
    def __init__(self, in_channels: int, rd_ratio: float = 0.0625) -> None:
        """Squeeze and Excite Module.
        The PyTorch implementation of `Squeeze-and-Excitation Networks` -
        https://arxiv.org/pdf/1709.01507.pdf

        :param in_channels: number of input channels
        :param rd_ratio: reduction ratio for input channels
        """
        super().__init__()
        self.reduce = nn.Conv2d(
          in_channels=in_channels, out_channels=int(in_channels * rd_ratio), kernel_size=1, stride=1, bias=True
        )
        self.expand = nn.Conv2d(
          in_channels=int(in_channels * rd_ratio), out_channels=in_channels, kernel_size=1, stride=1, bias=True
        )
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, C, H, W = inputs.shape
        x = F.avg_pool2d(inputs, kernel_size=[H, W])
        x = F.relu(self.reduce(x))
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, C, 1, 1)
        return inputs * x


class MobileOneBlock(nn.Module):
    def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int,
      stride: int = 1,
      padding: int = 0,
      dilation: int = 1,
      groups: int = 1,
      inference_mode: bool = False,
      use_se: bool = False,
      use_act: bool = True,
      use_scale_branch: bool = True,
      num_conv_branches: int = 1,
      activation: nn.Module = nn.GELU,
    ):
        """MobileOne building block
        This block has a multi-branched architecture at train-time
        and plain-CNN style architecture at inference time.
        Original paper: `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf

        :param in_channels: number of channels in the input
        :param out_channels: number of channels for the block's output
        :param kernel_size: convolution's kernel_size
        :param stride: stride size. Default: 1
        :param padding: zero-padding size. Default: 0
        :param dilation: kernel dilation factor. Default: 1
        :param groups: number of groups. Default: 1
        :param inference_mode: whether in inference mode or not. Default: False
        :param use_se: whether to use SE-ReLU block. Default: False
        :param use_act: whether to use activation layer. default: True
        :param use_scale_branch: whether to use scale branch. Default: True
        :param num_conv_branches: number of linear convolution branches. Default: 1
        :param activation: the activation layer to use. Default: nn.GELU
        """
        super().__init__()
        self.inference_mode = inference_mode
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()
        
        if use_act:
            self.act = activation()
        else:
            self.act = nn.Identity()
        
        if inference_mode:
            self.reparam_conv = nn.Conv2d(
              in_channels,
              out_channels,
              kernel_size,
              stride=stride,
              padding=padding,
              dilation=dilation,
              groups=groups,
              bias=True
            )
        else:
            # Re-parameterizable skip connection
            self.rbr_skip = nn.BatchNorm2d(in_channels) if in_channels == out_channels and stride == 1 else None
            
            # Re-parameterizable conv branches
            if num_conv_branches > 0:
                self.rbr_conv = nn.ModuleList()
                for _ in range(num_conv_branches):
                    self.rbr_conv.append(self._conv_bn(kernel_size=kernel_size, padding=padding))
            else:
                self.rbr_conv = None
            
            # Re-parameterizable scale branch
            self.rbr_scale = None
            if use_scale_branch and kernel_size > 1:
                self.rbr_scale = self._conv_bn(kernel_size=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Inference mode forward pass
        if hasattr(self, 'reparam_conv'):
            return self.act(self.se(self.reparam_conv(x)))
        
        # Multi-branched train-time forward pass
        # skip connection branch
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)
        
        # scale branch
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)
        
        # other branches
        out = identity_out + scale_out
        if self.rbr_conv is not None:
            for rc in self.rbr_conv:
                out = out + rc(x)
        
        return self.act(self.se(out))
    
    def reparameterize(self) -> None:
        """
        following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        <https://arxiv.org/pdf/2101.03697.pdf>. We re-parameterize multi-branched
        architecture used at train-time to obtain a plain CNN-like structure
        for inference
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
          in_channels=self.in_channels,
          out_channels=self.out_channels,
          kernel_size=self.kernel_size,
          stride=self.stride,
          padding=self.padding,
          dilation=self.dilation,
          groups=self.groups,
          bias=True
        )
        
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias
        
        # Delete un-used branches
        for param in self.parameters():
            param.detach_()
        
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')
        if hasattr(self, 'rbr_scale'):
            self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_conv'):
            self.__delattr__('rbr_conv')
        
        self.inference_mode = True
    
    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """method to obtain re-parameterized kernel and bias.
        reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # pad scale branch kernel to match conv branch kernel size
            pad = self.kernel_size // 2
            kernel_scale = F.pad(kernel_scale, [pad, pad, pad, pad])
        
        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)
        
        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        if self.rbr_conv is not None:
            for rc in self.rbr_conv:
                _kernel, _bias = self._fuse_bn_tensor(rc)
                kernel_conv = kernel_conv + _kernel
                bias_conv = bias_conv + _bias
        
        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final
    
    def _fuse_bn_tensor(self, branch: Union[nn.Sequential, nn.BatchNorm2d]) -> Tuple[torch.Tensor, torch.Tensor]:
        """method to fuse batchnorm layer with preceeding conv layer.
        reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch: sequence of ops to be fused
        :return: tuple of (kernel, bias) after fusing batchnorm
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros(
                  (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
                  dtype=branch.weight.dtype,
                  device=branch.weight.device,
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        
        assert running_var is not None
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean*gamma/std
    
    def _conv_bn(self, kernel_size: int, padding: int) -> nn.Sequential:
        """Helper method to construct conv-bn layers

        :param kernel_size: convolution's kernel_size
        :param padding: zero-padding size
        :return: conv-bn sequential module
        """
        module_list = nn.Sequential()
        module_list.add_module(
          'conv',
          nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=padding,
            groups=self.groups,
            bias=False
          )
        )
        module_list.add_module('bn', nn.BatchNorm2d(self.out_channels))
        return module_list


class ReparamLargeKernelConv(nn.Module):
    def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int,
      stride: int,
      groups: int,
      small_kernel: int,
      inference_mode: bool = False,
      activation: nn.Module = nn.GELU
    ) -> None:
        """Building block of RepLKNet.
        This class defines over-parameterized large kernel conv block,
        which is instroduced in `RepLKNet` - <https://arxiv.org/pdf/2203.06717.pdf>

        reference: https://github.com/DingXiaoH/RepLKNet-pytorch

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: size of large kernel conv branch
        :param stride: stride size. Default: 1
        :param groups: number of groups. Default: 1
        :param small_kernel: size of small kernel conv branch
        :param inference_mode: whether initializing module as inference mode. Defaul: False
        :param activation: activation layer. Default: nn.GELU
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inference_mode = inference_mode
        self.act = activation
        
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        
        self.stride = stride
        self.groups = groups
        self.padding = kernel_size // 2
        
        if inference_mode:
            self.lkb_reparam = nn.Conv2d(
              in_channels=in_channels,
              out_channels=out_channels,
              kernel_size=kernel_size,
              stride=stride,
              padding=self.padding,
              dilation=1,
              groups=groups,
              bias=True
            )
        else:
            self.lkb_origin = self._conv_bn(kernel_size=kernel_size, padding=self.padding)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, \
                    'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = self._conv_bn(kernel_size=small_kernel, padding=small_kernel // 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(x)
        else:
            out = self.lkb_origin(x)
            if hasattr(self, 'small_conv'):
                out = out + self.small_conv(x)
        
        return self.act(out)
    
    def reparameterize(self) -> None:
        """
        following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        <https://arxiv.org/pdf/2101.03697.pdf>. We re-parameterize multi-branched
        architecture used at train-time to obtain a plain CNN-like structure
        for inference
        """
        if self.inference_mode:
            return
        eq_k, eq_b = self._get_kernel_bias()
        self.lkb_reparam = nn.Conv2d(
          in_channels=self.in_channels,
          out_channels=self.out_channels,
          kernel_size=self.kernel_size,
          stride=self.stride,
          padding=self.padding,
          dilation=self.lkb_origin.conv.dilation,
          groups=self.groups,
          bias=True
        )
        
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')
    
    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """method to obtain re-parameterized kernel and bias
        reference: https://github.com/DingXiaoH/RepLKNet-pytorch

        :return: tuple of (kernel, bias) after fusing branches.
        """
        eq_k, eq_b = self._fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = self._fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b = eq_b + small_b
            eq_k = eq_k + F.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)
        
        return eq_k, eq_b
    
    @staticmethod
    def _fuse_bn(conv: torch.Tensor, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
        """method to fuse batchnorm layer with conv layer

        :param conv: convolution kernel weights
        :param bn: batch norm 2d layer
        :return: tuple of (kernel, bias) after fusing batchnorm.
        """
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        assert running_var is not None
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean*gamma/std
    
    def _conv_bn(self, kernel_size: int, padding: int = 0) -> nn.Sequential:
        """Helper method to construct conv-bn layers

        :param kernel_size: convolution's kernel size
        :param padding: zero-padding size
        :return: conv-bn sequential module
        """
        module_list = nn.Sequential()
        module_list.add_module(
          'conv',
          nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=padding,
            groups=self.groups,
            bias=False
          )
        )
        module_list.add_module('bn', nn.BatchNorm2d(self.out_channels))
        return module_list


class Attention(nn.Module):
    def __init__(
      self,
      dim: int,
      head_dim: int = 32,
      qkv_bias: bool = False,
      attn_drop: float = 0.,
      proj_drop: float = 0.,
    ) -> None:
        """Multi-headed Self-Attention Module

        :param dim: number of embedding dimensions
        :param head_dim: number of hidden dimensions per head. Default: 32
        :param qkv_bias: whether using bias in qkv or not. Default: False
        :param attn_drop: dropout rate for attention tensor
        :param proj_drop: dropout rate for projection tensor
        """
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv[0].float(), qkv[1].float(), qkv[2].float()
        attn = (q @ k.transpose(-1, -2)) * self.scale
        attn = self.attn_drop(self.softmax(attn))
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


def reparameterize_model(model: nn.Module) -> nn.Module:
    """method to return a model where a multi-branched structure
    used in training is re-parameterized into a single branch
    for inference

    :param model: the multi-branched model in train mode
    :return: model in inference mode
    """
    # avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'reparameterize'):
            module.reparameterize()
    return model
