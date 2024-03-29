import os
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropPath, trunc_normal_
from timm.models import checkpoint_seq, register_model

from .components import Attention, MobileOneBlock, ReparamLargeKernelConv

warnings.filterwarnings('ignore')


def _cfg(url: str = '', **kwargs: Any) -> Dict[str, Any]:
    return {
      'url': url,
      'num_classes': 1000,
      'input_size': (3, 256, 256),
      'pool_size': None,
      'crop_pct': 0.9,
      'interpolation': 'bicubic',
      'mean': IMAGENET_DEFAULT_MEAN,
      'std': IMAGENET_DEFAULT_STD,
      'classifier': 'head',
      **kwargs
    }


default_cfgs = {
  'fastvit_t': _cfg(),
  'fastvit_s': _cfg(),
  'fastvit_m': _cfg(crop_pct=0.95),
}


class PatchEmbed(nn.Sequential):
    def __init__(
      self, in_channels: int, embed_dim: int, patch_size: int, stride: int, inference_mode: bool = False
    ) -> None:
        """Convolution Patch Embedding Layer

        :param in_channels: number of input channels
        :param embed_dim: the embedding dimension
        :param patch_size: patch size for embedding computation
        :param stride: stride for convolution embedding layer
        :param inference_mode: whether to initialize module in inference mode. Default: False
        """
        super().__init__(
          ReparamLargeKernelConv(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=stride,
            groups=in_channels,
            small_kernel=3,
            inference_mode=inference_mode
          ),
          MobileOneBlock(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1
          )
        )


class ConvStem(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, inference_mode: bool = False) -> None:
        """Building Convolutional Stem Block with nn.Sequential object of MobileOneBlock

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param inference_mode: flag to instantiate model in inference mode. Default: False
        """
        super().__init__(
          MobileOneBlock(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            inference_mode=inference_mode,
            num_conv_branches=1
          ),
          MobileOneBlock(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=out_channels,
            inference_mode=inference_mode,
            num_conv_branches=1
          ),
          MobileOneBlock(out_channels, out_channels, kernel_size=1, inference_mode=inference_mode, num_conv_branches=1),
        )


class RepMixer(nn.Module):
    def __init__(
      self,
      dim: int,
      kernel_size: int,
      use_layer_scale: bool = False,
      layer_scale_init_value: float = 1e-5,
      inference_mode: bool = False
    ) -> None:
        """Re-parameterizable Token Mixer (so called RepMixer)
        A Token Mixer Replacement in General MetaFormer architecture
        The original paper: `FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization` -
        https://arxiv.org/pdf/2303.14189.pdf

        :param dim: input feature map dimension.
        :param kernel_size: kernel size for spatial mixing. Default: 3
        :param use_layer_scale: whether to use additional learnable layer scale. Default: True
        :param layer_scale_init_value: initial value for layer scale. Default: 1e-5
        :param inference_mode: whether in inference mode or not. Default: False
        """
        super().__init__()
        
        if inference_mode:
            self.reparam_conv = nn.Conv2d(
              dim, dim, kernel_size, stride=1, padding=kernel_size // 2, groups=dim, bias=True
            )
        else:
            self.norm = MobileOneBlock(
              dim,
              dim,
              kernel_size,
              padding=kernel_size // 2,
              groups=dim,
              use_act=False,
              use_scale_branch=False,
              num_conv_branches=0
            )
            self.mixer = MobileOneBlock(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, use_act=False)
            self.use_layer_scale = use_layer_scale
            if use_layer_scale:
                self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'reparam_conv'):
            return self.reparam_conv(x)
        
        # the explanation for subtraction the norm from mixer is because
        # the `MobileOneBlock` do contain extra `identity` branch,
        # here we don't need this, so we subtract it from the mixer
        # after the subtraction, the mixer is containing dual branch of
        # `3x3` depth-wise conv and `1x1` point-wise conv
        if self.use_layer_scale:
            x = x + self.layer_scale * (self.mixer(x) - self.norm(x))
        else:
            x = x + self.mixer(x) - self.norm(x)
        return x


class RepCPE(nn.Module):
    def __init__(self, dim: int, spatial_dim: Tuple[int, int] = (7, 7), inference_mode: bool = False) -> None:
        """Conditional Positional Encoding with Re-parameterizable supports

        :param dim: number of embedding dimensions.
        :param spatial_dim: spatial shape of kernel for positional encoding. Default: (7, 7)
        :param inference_mode: flag to instantiate block in inference mode. Default: False
        """
        super().__init__()
        self.dim = dim
        self.inference_mode = inference_mode
        self.spatial_dim = spatial_dim
        
        # Depth-wise convolution
        if inference_mode:
            self.reparam_conv = nn.Conv2d(
              dim, dim, kernel_size=spatial_dim, stride=1, padding=int(spatial_dim[0] // 2), groups=dim, bias=True
            )
        else:
            self.pe = nn.Conv2d(
              dim, dim, kernel_size=spatial_dim, stride=1, padding=int(spatial_dim[0] // 2), groups=dim, bias=True
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'reparam_conv'):
            return self.reparam_conv(x)
        
        return self.pe(x) + x
    
    def reparameterize(self) -> None:
        if self.inference_mode:
            return
        
        # building equivalent id tensor
        kernel_value = torch.zeros(
          size=(self.dim, 1, self.spatial_dim[0], self.spatial_dim[1]),
          dtype=self.pe.weight.dtype,
          device=self.pe.weight.device,
        )
        for i in range(self.dim):
            kernel_value[i, 0, self.spatial_dim[0] // 2, self.spatial_dim[1] // 2] = 1
        id_tensor = kernel_value
        
        # Re-parameterize id tensor
        w_final = id_tensor + self.pe.weight
        b_final = self.pe.bias
        
        # introduce reparam conv
        self.reparam_conv = nn.Conv2d(
          self.dim,
          self.dim,
          kernel_size=self.spatial_dim,
          stride=1,
          padding=int(self.spatial_dim[0] // 2),
          groups=self.dim,
          bias=True
        )
        
        self.reparam_conv.weight.data = torch.tensor(w_final)
        self.reparam_conv.bias.data = torch.tensor(b_final)
        
        for param in self.parameters():
            param.detach_()
        self.__delattr__('pe')
        
        self.inference_mode = True


class ConvFFN(nn.Sequential):
    def __init__(
      self,
      in_channels: int,
      hidden_channels: Optional[int] = None,
      out_channels: Optional[int] = None,
      act_layer: Callable[..., nn.Module] = nn.GELU,
      dropout_rate: float = 0.
    ) -> None:
        """Convolutional FFN module.

        :param in_channels: number of input channels
        :param hidden_channels: number of channels after expansion. Default: None
        :param out_channels: number of output channels. Default: None
        :param act_layer: activation layer. Default: nn.GELU
        :param dropout_rate: dropout rate. Default: 0.
        """
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        
        dropout = nn.Dropout(p=dropout_rate)
        super().__init__(
          nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3, groups=in_channels, bias=False),
          nn.BatchNorm2d(out_channels),
          nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
          dropout,
          act_layer(),
          nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
          dropout,
        )


class RepMixerBlock(nn.Module):
    def __init__(
      self,
      dim: int,
      kernel_size: int,
      mlp_ratio: float = 4.,
      act_layer: Callable[..., nn.Module] = nn.GELU,
      dropout_rate: float = 0.,
      drop_path_rate: float = 0.,
      use_layer_scale: bool = True,
      layer_scale_init_value: float = 1e-5,
      inference_mode: bool = False
    ) -> None:
        """MetaFormer block with RepMixer as token mixer.
        For more details on MetaFormer structure, please refer to:
        `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`

        :param dim: number of embedding dimensions.
        :param kernel_size: kernel size for RepMixer. Default: 3
        :param mlp_ratio: MLP expansion ratio. Default: 4.
        :param act_layer: activation layer. Default: nn.GELU
        :param dropout_rate: dropout rate. Default: 0.
        :param drop_path_rate: drop path rate. Default: 0.
        :param use_layer_scale: flag to turn on layer scale. Default: True
        :param layer_scale_init_value: layer scale value at initialization. Default: 1e-5
        :param inference_mode: flag to instantiate block in inference mode. Default: False
        """
        super().__init__()
        self.inference_mode = inference_mode
        
        self.token_mixer = RepMixer(
          dim,
          kernel_size=kernel_size,
          use_layer_scale=use_layer_scale,
          layer_scale_init_value=layer_scale_init_value,
          inference_mode=inference_mode
        )
        
        hidden_channels = int(dim * mlp_ratio)
        self.ffn = ConvFFN(dim, hidden_channels=hidden_channels, act_layer=act_layer, dropout_rate=dropout_rate)
        
        # drop path
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
        # using layer scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_mixer(x)
        conv_feature = self.ffn(x)
        if hasattr(self, 'layer_scale'):
            conv_feature = self.layer_scale * conv_feature
        
        x = x + self.drop_path(conv_feature)
        return x


class AttentionBlock(nn.Module):
    def __init__(
      self,
      dim: int,
      mlp_ratio: float = 4.,
      act_layer: Callable[..., nn.Module] = nn.GELU,
      norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
      dropout_rate: float = 0.,
      drop_path_rate: float = 0.,
      use_layer_scale: bool = True,
      layer_scale_init_value: float = 1e-5
    ) -> None:
        """MetaFormer block with Multi-headed Self-Attention as token mixer
        For more details on MetaFormer structure, please refer to:
        `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`

        :param dim: number of embedding dimensions.
        :param mlp_ratio: MLP expansion ratio. Default: 4.
        :param act_layer: activation layer. Default: nn.GELU
        :param norm_layer: normalization layer. Default: nn.BatchNorm2d
        :param dropout_rate: dropout rate. Default: 0.
        :param drop_path_rate: drop path rate. Default: 0.
        :param use_layer_scale: flag to turn on layer scale. Default: True
        :param layer_scale_init_value: layer scale value at initialization. Default: 1e-5
        """
        super().__init__()
        
        self.token_mixer = Attention(dim)
        self.norm = norm_layer(dim)
        
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = ConvFFN(dim, hidden_channels=hidden_dim, act_layer=act_layer, dropout_rate=dropout_rate)
        
        # drop path
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
        # layer scale
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.mixer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True)
            self.ffn_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed_feature = self.token_mixer(self.norm(x))
        if self.use_layer_scale:
            x = x + self.drop_path(self.mixer_scale * mixed_feature)
            x = x + self.drop_path(self.ffn_scale * self.ffn(x))
        else:
            x = x + self.drop_path(mixed_feature)
            x = x + self.drop_path(self.ffn(x))
        
        return x


class BasicBlock(nn.Module):
    mixer_types = ['repmixer', 'attention']
    
    def __init__(
      self,
      dim: int,
      dim_out: int,
      depth: int,
      mixer_type: str = 'repmixer',
      downsample: bool = True,
      down_patch_size: int = 7,
      down_stride: int = 2,
      pos_embed_layer: Optional[nn.Module] = None,
      kernel_size: int = 3,
      mlp_ratio: float = 4.,
      act_layer: Callable[..., nn.Module] = nn.GELU,
      norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
      dropout_rate: float = 0.,
      drop_path: Union[float, Tuple[float, ...]] = 0.1,
      use_layer_scale: bool = True,
      layer_scale_init_value: float = 1e-5,
      inference_mode: bool = False,
    ) -> None:
        """Basic FastViT blocks within a stage.

        :param dim: number of embedding dimensions.
        :param dim_out: number of output dimensions.
        :param depth: number of blocks in stage.
        :param mixer_type: token mixer type. Default: 'repmixer'
        :param downsample: whether adding downsample layer or not. Default: True
        :param down_patch_size: patch size for downsample layer. Default: 7
        :param down_stride: stride for downsample layer. Default: 2
        :param pos_embed_layer: positional embedding layer to use (optional). Default: None
        :param kernel_size: kernel size for RepMixer. Default: 3
        :param mlp_ratio: MLP expansion ratio. Default: 4.
        :param act_layer: activation layer. Default: nn.GELU
        :param norm_layer: normalization layer. Default: nn.BatchNorm2d
        :param dropout_rate: dropout rate. Default: 0.
        :param drop_path: drop path rate. Default: 0.
        :param use_layer_scale: flag to turn on layer scale regularization. Default: True
        :param layer_scale_init_value: layer scale value at initialization. Default: 1e-5
        :param inference_mode: flag to instantiate block in inference mode. Default: False
        """
        super().__init__()
        self.grad_checkpointing = False
        assert mixer_type in self.mixer_types
        
        if downsample:
            self.downsample = PatchEmbed(
              in_channels=dim,
              embed_dim=dim_out,
              patch_size=down_patch_size,
              stride=down_stride,
              inference_mode=inference_mode
            )
        else:
            assert dim == dim_out
            self.downsample = nn.Identity()
        
        if pos_embed_layer is not None:
            self.pos_embed = pos_embed_layer(dim_out, inference_mode=inference_mode)
        else:
            self.pos_embed = nn.Identity()
        
        blocks = nn.ModuleList()
        for i in range(depth):
            if mixer_type == 'repmixer':
                blocks.append(
                  RepMixerBlock(
                    dim_out,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    dropout_rate=dropout_rate,
                    drop_path_rate=drop_path[i] if isinstance(drop_path, (tuple, list)) else drop_path,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                    inference_mode=inference_mode
                  )
                )
            elif mixer_type == 'attention':
                blocks.append(
                  AttentionBlock(
                    dim_out,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    dropout_rate=dropout_rate,
                    drop_path_rate=drop_path[i] if isinstance(drop_path, (tuple, list)) else drop_path,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value
                  )
                )
            else:
                raise NotImplementedError
        
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.pos_embed(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)  # type: ignore
        else:
            x = self.blocks(x)
        return x


class FastViT(nn.Module):
    def __init__(
      self,
      in_channels: int = 3,
      depths: Union[List[int], Tuple[int, ...]] = (2, 2, 6, 2),
      token_mixers: Union[List[str], Tuple[str, ...]] = ('repmixer', 'repmixer', 'repmixer', 'repmixer'),
      embed_dims: Union[List[int], Tuple[int, ...]] = (64, 128, 256, 512),
      mlp_ratios: Union[List[float], Tuple[float, ...]] = (4, ) * 4,
      downsamples: Union[List[bool], Tuple[bool, ...]] = (False, True, True, True),
      repmixer_kernel_size: int = 3,
      norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
      act_layer: Callable[..., nn.Module] = nn.GELU,
      num_classes: int = 1000,
      pos_embeds: Union[List[Optional[nn.Module]], Tuple[Optional[nn.Module], ...]] = (None, None, None, None),
      down_patch_size: int = 7,
      down_stride: int = 2,
      dropout_rate: float = 0.0,
      drop_path_rate: float = 0.0,
      use_layer_scale: bool = True,
      layer_scale_init_value: float = 1e-5,
      fork_feat: bool = False,
      pretrained: bool = False,
      pretrained_path: str = '',
      cls_ratio: float = 2.,
      inference_mode: bool = False,
      **kwargs: Any
    ) -> None:
        """This class implements `FastViT architecture <https://arxiv.org/pdf/2303.14189.pdf>`

        :param in_channels: number of input channels. Default: 3
        :param depths: list of layer's depth. Default: (2, 2, 6, 2)
        :param token_mixers: list of token mixer type to use. Default: ('repmixer', 'repmixer', 'repmixer', 'repmixer')
        :param embed_dims: list of embedding dimensions for each stage. Default: (64, 128, 256, 512)
        :param mlp_ratios: list of MLP expansion ratios. Default: (4, 4, 4, 4)
        :param downsamples: list of boolean values in which downsample layer is used. Default: (False, True, True, True)
        :param repmixer_kernel_size: kernel size for RepMixer. Default: 3
        :param norm_layer: normalization layer. Default: nn.BatchNorm2d
        :param act_layer: activation layer. Default: nn.GELU
        :param num_classes: number of classification labels. Default: 1000
        :param pos_embeds: list of positional encoding layers. Default: None
        :param down_patch_size: patch size for downsample layer. Default: 7
        :param down_stride: stride for downsample layer. Default: 2
        :param dropout_rate: dropout rate. Default: 0.
        :param drop_path_rate: drop path rate. Default: 0.
        :param use_layer_scale: flag to turn on layer scale regularization. Default: True
        :param layer_scale_init_value: layer scale value at initialization. Default: 1e-5
        :param fork_feat: flag to use model for dense prediction tasks. Default: False
        :param pretrained: flag to use pre-trained checkpoint. Default: False
        :param pretrained_path: path to pre-trained weights. Default: ''
        :param cls_ratio: classification ratio for last feature extraction. Default: 2.
        :param inference_mode: flag to instantiate model in inference mode. Default: False
        """
        super().__init__()
        
        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.pretrained = pretrained
        
        # convolutional stem
        self.patch_embed = ConvStem(in_channels, embed_dims[0], inference_mode=inference_mode)
        
        # stochastic depth decay rule
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        
        # building stages to form the backbone
        prev_dim = embed_dims[0]
        self.backbone = nn.ModuleList()
        for i_layer in range(len(depths)):
            downsample = downsamples[i_layer] or prev_dim != embed_dims[i_layer]
            stage = BasicBlock(
              dim=prev_dim,
              dim_out=embed_dims[i_layer],
              depth=depths[i_layer],
              mixer_type=token_mixers[i_layer],
              downsample=downsample,
              down_patch_size=down_patch_size,
              down_stride=down_stride,
              pos_embed_layer=pos_embeds[i_layer],
              kernel_size=repmixer_kernel_size,
              mlp_ratio=mlp_ratios[i_layer],
              act_layer=act_layer,
              norm_layer=norm_layer,
              dropout_rate=dropout_rate,
              drop_path=dpr[i_layer],
              use_layer_scale=use_layer_scale,
              layer_scale_init_value=layer_scale_init_value,
              inference_mode=inference_mode
            )
            self.backbone.append(stage)
            prev_dim = embed_dims[i_layer]
        
        # for segmentation and detection tasks, extract intermediate output
        if self.fork_feat:
            # adding post norm layer for each output of BasicBlock
            self.out_indices = [0, 1, 2, 3]
            for embed_idx, layer_idx in enumerate(self.out_indices):
                if embed_idx == 0 and os.environ.get('FORK_LAST3', False):
                    # for RetinaNet, `start_level=1`. the first norm layer will not used.
                    # cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[embed_idx])
                layer_name = f'norm{layer_idx}'
                self.add_module(layer_name, layer)
        else:  # adding classifier head
            self.conv_exp = MobileOneBlock(
              in_channels=embed_dims[-1],
              out_channels=int(embed_dims[-1] * cls_ratio),
              kernel_size=3,
              stride=1,
              padding=1,
              groups=embed_dims[-1],
              inference_mode=inference_mode,
              use_se=True,
              num_conv_branches=1
            )
            self.gap = nn.AdaptiveMaxPool2d(1)
            self.head = (
              nn.Linear(int(embed_dims[-1] * cls_ratio), self.num_classes) if num_classes > 0 else nn.Identity()
            )
        
        self.apply(self.cls_init_weights)
        
        # load pre-trained checkpoint
        if self.fork_feat and self.pretrained:
            self.init_weights(pretrained_path)
    
    def cls_init_weights(self, m: nn.Module) -> None:
        """method for classifier head's weight initialization

        :param m: the module to have weight initialization
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def _scrub_checkpoint(self, checkpoint: dict) -> dict:
        """method to load model's checkpoint strictly

        :param checkpoint: checkpoint dict to load
        :return: strict state_dict
        """
        sterile_dict = {}
        for k, v in checkpoint.items():
            if k not in self.state_dict():
                continue
            if v.shape == self.state_dict()[k].shape:
                sterile_dict[k] = v
        return sterile_dict
    
    def init_weights(self, path: str) -> None:
        """ method to load model checkpoint if `self.fork_feat` and `self.pretrained` are provided """
        ckpt = torch.load(path, map_location='cpu')
        state_dict = self._scrub_checkpoint(ckpt['state_dict'])
        self.load_state_dict(state_dict)
    
    @torch.jit.ignore
    def set_grad_checkpointing(self, enabled: bool = True) -> None:
        for s in self.backbone:
            s.grad_checkpointing = enabled
    
    def forward_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """method to extract the input image into patches

        :param x: input image tensor
        :return: patches of feature image
        """
        return self.patch_embed(x)
    
    def forward_tokens(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """method to forward input tensor through FastViT stages.

        :param x: input tensor
        :return: a last feature map tensor or list of feature map tensor for each stage depends on `self.fork_feat`
        """
        outputs = []
        for idx, block in enumerate(self.backbone):
            x = block(x)
            # post-normalization is applied for dense prediction tasks
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outputs.append(x_out)
        
        if self.fork_feat:
            # return the feature maps of 4 stages for dense prediction
            return outputs
        # return only the last feature map for image classification
        return x
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """the main forward method

        :param x: input image tensor
        :return: a classified result tensor or list of feature map tensors depends on `self.fork_feat`
        """
        # patch embedding
        x = self.forward_embeddings(x)
        # backbone forwarding
        outputs = self.forward_tokens(x)
        if self.fork_feat:
            # dense prediction task returning
            return outputs
        # image classification task
        x = self.conv_exp(outputs)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


@register_model
def fastvit_t8(pretrained: bool = False, **kwargs: Any) -> FastViT:
    """Instantiate FastViT-T8 model variant."""
    depths = [2, 2, 4, 2]
    embed_dims = [48, 96, 192, 384]
    mlp_ratios = [3., 3., 3., 3.]
    token_mixers = ['repmixer', 'repmixer', 'repmixer', 'repmixer']
    model = FastViT(
      in_channels=3,
      depths=depths,
      token_mixers=token_mixers,
      embed_dims=embed_dims,
      mlp_ratios=mlp_ratios,
      pretrained=pretrained,
      **kwargs
    )
    model.default_cfg = default_cfgs['fastvit_t']
    return model


@register_model
def fastvit_t12(pretrained: bool = False, **kwargs: Any) -> FastViT:
    """Instantiate FastViT-T12 model variant."""
    depths = [2, 2, 6, 2]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [3., 3., 3., 3.]
    token_mixers = ['repmixer', 'repmixer', 'repmixer', 'repmixer']
    model = FastViT(
      in_channels=3,
      depths=depths,
      token_mixers=token_mixers,
      embed_dims=embed_dims,
      mlp_ratios=mlp_ratios,
      pretrained=pretrained,
      **kwargs
    )
    model.default_cfg = default_cfgs['fastvit_t']
    return model


@register_model
def fastvit_s12(pretrained: bool = False, **kwargs: Any) -> FastViT:
    """Instantiate FastViT-S12 model variant."""
    depths = [2, 2, 6, 2]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [4., 4., 4., 4.]
    token_mixers = ['repmixer', 'repmixer', 'repmixer', 'repmixer']
    model = FastViT(
      in_channels=3,
      depths=depths,
      token_mixers=token_mixers,
      embed_dims=embed_dims,
      mlp_ratios=mlp_ratios,
      pretrained=pretrained,
      **kwargs
    )
    model.default_cfg = default_cfgs['fastvit_s']
    return model


@register_model
def fastvit_sa12(pretrained: bool = False, **kwargs: Any) -> FastViT:
    """Instantiate FastViT-SA12 model variant."""
    depths = [2, 2, 6, 2]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [4., 4., 4., 4.]
    pos_embeds = [None, None, None, partial(RepCPE, spatial_dim=(7, 7))]
    token_mixers = ['repmixer', 'repmixer', 'repmixer', 'attention']
    model = FastViT(
      in_channels=3,
      depths=depths,
      token_mixers=token_mixers,
      embed_dims=embed_dims,
      mlp_ratios=mlp_ratios,
      pos_embeds=pos_embeds,
      pretrained=pretrained,
      **kwargs
    )
    model.default_cfg = default_cfgs['fastvit_s']
    return model


@register_model
def fastvit_sa24(pretrained: bool = False, **kwargs: Any) -> FastViT:
    """Instantiate FastViT-SA24 model variant."""
    depths = [4, 4, 12, 4]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [4., 4., 4., 4.]
    pos_embeds = [None, None, None, partial(RepCPE, spatial_dim=(7, 7))]
    token_mixers = ['repmixer', 'repmixer', 'repmixer', 'attention']
    model = FastViT(
      in_channels=3,
      depths=depths,
      token_mixers=token_mixers,
      embed_dims=embed_dims,
      mlp_ratios=mlp_ratios,
      pos_embeds=pos_embeds,
      pretrained=pretrained,
      **kwargs
    )
    model.default_cfg = default_cfgs['fastvit_s']
    return model


@register_model
def fastvit_sa36(pretrained: bool = False, **kwargs: Any) -> FastViT:
    """Instantiate FastViT-SA36 model variant."""
    depths = [6, 6, 18, 6]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [4., 4., 4., 4.]
    pos_embeds = [None, None, None, partial(RepCPE, spatial_dim=(7, 7))]
    token_mixers = ['repmixer', 'repmixer', 'repmixer', 'attention']
    model = FastViT(
      in_channels=3,
      depths=depths,
      token_mixers=token_mixers,
      embed_dims=embed_dims,
      mlp_ratios=mlp_ratios,
      pos_embeds=pos_embeds,
      layer_scale_init_value=1e-6,
      pretrained=pretrained,
      **kwargs
    )
    model.default_cfg = default_cfgs['fastvit_m']
    return model


@register_model
def fastvit_ma36(pretrained: bool = False, **kwargs: Any) -> FastViT:
    """Instantiate FastViT-MA36 model variant."""
    depths = [6, 6, 18, 6]
    embed_dims = [76, 152, 304, 608]
    mlp_ratios = [4., 4., 4., 4.]
    pos_embeds = [None, None, None, partial(RepCPE, spatial_dim=(7, 7))]
    token_mixers = ['repmixer', 'repmixer', 'repmixer', 'attention']
    model = FastViT(
      in_channels=3,
      depths=depths,
      token_mixers=token_mixers,
      embed_dims=embed_dims,
      mlp_ratios=mlp_ratios,
      pos_embeds=pos_embeds,
      layer_scale_init_value=1e-6,
      pretrained=pretrained,
      **kwargs
    )
    model.default_cfg = default_cfgs['fastvit_m']
    return model
