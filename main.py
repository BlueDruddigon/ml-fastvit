import argparse
import glob
import logging
import os
from contextlib import suppress
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from timm.data import AugMixDataset, create_dataset, create_loader, FastCollateMixup, Mixup, resolve_data_config
from timm.layers import convert_splitbn_model
from timm.loss import BinaryCrossEntropy, JsdCrossEntropy, LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model, load_checkpoint, resume_checkpoint, safe_model_name
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import (
  CheckpointSaver,
  distribute_bn,
  get_outdir,
  ModelEmaV2,
  NativeScaler,
  random_seed,
  setup_default_logging,
  update_summary,
)
from torch.nn.parallel import DistributedDataParallel as NativeDDP

import models
from losses.distillation_loss import DistillationLoss
from trainer import train_one_epoch, validate
from utils.cosine_annealing import CosineWDSchedule

try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    has_native_amp = False

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')


def _parse_args():
    config_parser = argparse.ArgumentParser(description='training config', add_help=False)
    config_parser.add_argument(
      '-c', '--config', default='', type=str, metavar='FILE', help='YAML config file specifying default arguments'
    )
    
    parser = argparse.ArgumentParser(description='pytorch imagenet training')
    
    # Dataset parameters
    parser.add_argument('data_dir', metavar='DIR', help='path to dataset')
    parser.add_argument(
      '-d', '--dataset', default='', metavar='NAME', help='dataset type. default: ImageFolder/ImageTar if empty'
    )
    parser.add_argument('--train-split', metavar='NAME', default='train', help='dataset train split. default: train')
    parser.add_argument(
      '--valid-split', metavar='NAME', default='validation', help='dataset validation split. default: validation'
    )
    parser.add_argument(
      '--dataset-download',
      action='store_true',
      default=False,
      help='allow download of dataset for torch/ and tfds/ datasets that support it.'
    )
    parser.add_argument(
      '--class-map', default='', type=str, metavar='FILENAME', help='path to class to idx mapping file. default: '
      ''
    )
    
    # model parameters
    parser.add_argument(
      '--model', default='resnet50', type=str, metavar='MODEL', help='name of model to train. default: "resnet50"'
    )
    parser.add_argument(
      '--pretrained',
      action='store_true',
      default=False,
      help='start with pretrained version of specified network (if available)'
    )
    parser.add_argument(
      '--initial-checkpoint',
      default='',
      type=str,
      metavar='PATH',
      help='initialize model from this checkpoint. default: None'
    )
    parser.add_argument(
      '--resume',
      default='',
      type=str,
      metavar='PATH',
      help='resume full model and optimizer state from checkpoint. default: None'
    )
    parser.add_argument(
      '--no-resume-opt',
      action='store_true',
      default=False,
      help='prevent resume of optimizer state when resuming model',
    )
    parser.add_argument(
      '--num-classes',
      type=int,
      default=None,
      metavar='N',
      help='number of label classes (Model default if None)',
    )
    parser.add_argument(
      '--gp',
      default=None,
      type=str,
      metavar='POOL',
      help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.',
    )
    parser.add_argument(
      '--img-size',
      type=int,
      default=None,
      metavar='N',
      help='Image patch size. default: None => model default',
    )
    parser.add_argument(
      '--input-size',
      default=None,
      nargs=3,
      type=int,
      metavar='N N N',
      help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty',
    )
    parser.add_argument(
      '--crop-pct',
      default=None,
      type=float,
      metavar='N',
      help='Input image center crop percent (for validation only)',
    )
    parser.add_argument(
      '--mean',
      type=float,
      nargs='+',
      default=None,
      metavar='MEAN',
      help='Override mean pixel value of dataset',
    )
    parser.add_argument(
      '--std',
      type=float,
      nargs='+',
      default=None,
      metavar='STD',
      help='Override std deviation of of dataset',
    )
    parser.add_argument(
      '--interpolation',
      default='',
      type=str,
      metavar='NAME',
      help='Image resize interpolation type (overrides model)',
    )
    parser.add_argument(
      '-b',
      '--batch-size',
      type=int,
      default=128,
      metavar='N',
      help='input batch size for training. default: 128',
    )
    parser.add_argument(
      '-vb',
      '--validation-batch-size',
      type=int,
      default=None,
      metavar='N',
      help='validation batch size override. default: None',
    )
    
    # Optimizer parameters
    parser.add_argument(
      '--opt',
      default='adamw',
      type=str,
      metavar='OPTIMIZER',
      help='Optimizer. default: "adamw"',
    )
    parser.add_argument(
      '--opt-eps',
      default=None,
      type=float,
      metavar='EPSILON',
      help='Optimizer Epsilon. default: None, use opt default',
    )
    parser.add_argument(
      '--opt-betas',
      default=None,
      type=float,
      nargs='+',
      metavar='BETA',
      help='Optimizer Betas. default: None, use opt default',
    )
    parser.add_argument(
      '--momentum',
      type=float,
      default=0.9,
      metavar='M',
      help='Optimizer momentum. default: 0.9',
    )
    parser.add_argument('--weight-decay', type=float, default=0.05, help='weight decay. default: 0.05')
    parser.add_argument(
      '--clip-grad',
      type=float,
      default=None,
      metavar='NORM',
      help='Clip gradient norm. default: None, no clipping',
    )
    parser.add_argument(
      '--clip-mode',
      type=str,
      default='norm',
      help='Gradient clipping mode. One of ("norm", "value", "agc")',
    )
    parser.add_argument(
      '--wd-schedule',
      type=str,
      default='none',
      help='Weight decay scheduler. One of ("cosine", "none")',
    )
    
    # Learning rate schedule parameters
    parser.add_argument(
      '--sched',
      default='cosine',
      type=str,
      metavar='SCHEDULER',
      help='LR scheduler. default: "step"',
    )
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate. default: 1e-3')
    parser.add_argument(
      '--lr-noise',
      type=float,
      nargs='+',
      default=None,
      metavar='pct, pct',
      help='learning rate noise on/off epoch percentages',
    )
    parser.add_argument(
      '--lr-noise-pct',
      type=float,
      default=0.67,
      metavar='PERCENT',
      help='learning rate noise limit percent. default: 0.67',
    )
    parser.add_argument(
      '--lr-noise-std',
      type=float,
      default=1.0,
      metavar='STDDEV',
      help='learning rate noise std-dev. default: 1.0',
    )
    parser.add_argument(
      '--lr-cycle-mul',
      type=float,
      default=1.0,
      metavar='MULT',
      help='learning rate cycle len multiplier. default: 1.0',
    )
    parser.add_argument(
      '--lr-cycle-decay',
      type=float,
      default=0.5,
      metavar='MULT',
      help='amount to decay each learning rate cycle. default: 0.5',
    )
    parser.add_argument(
      '--lr-cycle-limit',
      type=int,
      default=1,
      metavar='N',
      help='learning rate cycle limit, cycles enabled if > 1',
    )
    parser.add_argument(
      '--lr-k-decay',
      type=float,
      default=1.0,
      help='learning rate k-decay for cosine/poly. default: 1.0',
    )
    parser.add_argument(
      '--warmup-lr',
      type=float,
      default=1e-6,
      metavar='LR',
      help='warmup learning rate. default: 1e-6',
    )
    parser.add_argument(
      '--min-lr',
      type=float,
      default=1e-5,
      metavar='LR',
      help='lower lr bound for cyclic schedulers that hit 0 (1e-5)',
    )
    parser.add_argument(
      '--epochs',
      type=int,
      default=300,
      metavar='N',
      help='number of epochs to train. default: 300',
    )
    parser.add_argument(
      '--epoch-repeats',
      type=float,
      default=0.0,
      metavar='N',
      help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).',
    )
    parser.add_argument(
      '--start-epoch',
      default=None,
      type=int,
      metavar='N',
      help='manual epoch number (useful on restarts)',
    )
    parser.add_argument(
      '--decay-epochs',
      type=float,
      default=100,
      metavar='N',
      help='epoch interval to decay LR',
    )
    parser.add_argument(
      '--warmup-epochs',
      type=int,
      default=5,
      metavar='N',
      help='epochs to warmup LR, if scheduler supports',
    )
    parser.add_argument(
      '--cooldown-epochs',
      type=int,
      default=10,
      metavar='N',
      help='epochs to cooldown LR at min_lr, after cyclic schedule ends',
    )
    parser.add_argument(
      '--patience-epochs',
      type=int,
      default=10,
      metavar='N',
      help='patience epochs for Plateau LR scheduler (default: 10',
    )
    parser.add_argument(
      '--decay-rate',
      '--dr',
      type=float,
      default=0.1,
      metavar='RATE',
      help='LR decay rate. default: 0.1',
    )
    
    # Augmentation & regularization parameters
    parser.add_argument(
      '--no-aug',
      action='store_true',
      default=False,
      help='Disable all training augmentation, override other train aug args',
    )
    parser.add_argument(
      '--scale',
      type=float,
      nargs='+',
      default=[0.08, 1.0],
      metavar='PCT',
      help='Random resize scale. default: 0.08 1.0',
    )
    parser.add_argument(
      '--ratio',
      type=float,
      nargs='+',
      default=[3.0 / 4.0, 4.0 / 3.0],
      metavar='RATIO',
      help='Random resize aspect ratio. default: 0.75 1.33',
    )
    parser.add_argument('--hflip', type=float, default=0.5, help='Horizontal flip training aug probability')
    parser.add_argument('--vflip', type=float, default=0.0, help='Vertical flip training aug probability')
    parser.add_argument(
      '--color-jitter',
      type=float,
      default=0.4,
      metavar='PCT',
      help='Color jitter factor. default: 0.4',
    )
    parser.add_argument(
      '--aa',
      type=str,
      default='rand-m9-mstd0.5-inc1',
      metavar='NAME',
      help='Use AutoAugment policy. "v0" or "original". default: rand-m9-mstd0.5-inc1',
    )
    parser.add_argument(
      '--aug-repeats',
      type=int,
      default=0,
      help='Number of augmentation repetitions (distributed training only). default: 0',
    )
    parser.add_argument(
      '--aug-splits',
      type=int,
      default=0,
      help='Number of augmentation splits. default: 0, valid: 0 or >=2',
    )
    parser.add_argument(
      '--jsd-loss',
      action='store_true',
      default=False,
      help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.',
    )
    parser.add_argument(
      '--bce-loss',
      action='store_true',
      default=False,
      help='Enable BCE loss w/ Mixup/CutMix use.',
    )
    parser.add_argument(
      '--bce-target-thresh',
      type=float,
      default=None,
      help='Threshold for binarizing softened BCE targets. default: None, disabled',
    )
    parser.add_argument(
      '--reprob',
      type=float,
      default=0.25,
      metavar='PCT',
      help='Random erase prob. default: 0.25',
    )
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode. default: "pixel"')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count. default: 1')
    parser.add_argument(
      '--resplit',
      action='store_true',
      default=False,
      help='Do not random erase first (clean) augmentation split',
    )
    parser.add_argument(
      '--mixup',
      type=float,
      default=0.8,
      help='mixup alpha, mixup enabled if > 0.. default: 0.8',
    )
    parser.add_argument(
      '--cutmix',
      type=float,
      default=1.0,
      help='cutmix alpha, cutmix enabled if > 0.. default: 1.0',
    )
    parser.add_argument(
      '--cutmix-minmax',
      type=float,
      nargs='+',
      default=None,
      help='cutmix min/max ratio, overrides alpha and enables cutmix if set. default: None',
    )
    parser.add_argument(
      '--mixup-prob',
      type=float,
      default=1.0,
      help='Probability of performing mixup or cutmix when either/both is enabled',
    )
    parser.add_argument(
      '--mixup-switch-prob',
      type=float,
      default=0.5,
      help='Probability of switching to cutmix when both mixup and cutmix enabled',
    )
    parser.add_argument(
      '--mixup-mode',
      type=str,
      default='batch',
      help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )
    parser.add_argument(
      '--mixup-off-epoch',
      default=0,
      type=int,
      metavar='N',
      help='Turn off mixup after this epoch, disabled if 0. default: 0',
    )
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing. default: 0.1')
    parser.add_argument(
      '--train-interpolation',
      type=str,
      default='bicubic',
      help='Training interpolation (random, bilinear, bicubic). default: "random"',
    )
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout rate. default: 0.')
    parser.add_argument(
      '--drop-connect',
      type=float,
      default=None,
      metavar='PCT',
      help='Drop connect rate, DEPRECATED, use drop-path. default: None',
    )
    parser.add_argument(
      '--drop-path',
      type=float,
      default=None,
      metavar='PCT',
      help='Drop path rate. default: None',
    )
    parser.add_argument(
      '--drop-block',
      type=float,
      default=None,
      metavar='PCT',
      help='Drop block rate. default: None',
    )
    
    # Batch norm parameters (only works with gen_efficientnet based models currently)
    parser.add_argument(
      '--bn-tf',
      action='store_true',
      default=False,
      help='Use Tensorflow BatchNorm defaults for models that support it. default: False',
    )
    parser.add_argument(
      '--bn-momentum',
      type=float,
      default=None,
      help='BatchNorm momentum override (if not None)',
    )
    parser.add_argument(
      '--bn-eps',
      type=float,
      default=None,
      help='BatchNorm epsilon override (if not None)',
    )
    parser.add_argument(
      '--sync-bn',
      action='store_true',
      help='Enable NVIDIA Apex or Torch synchronized BatchNorm.',
    )
    parser.add_argument(
      '--dist-bn',
      type=str,
      default='reduce',
      help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")',
    )
    parser.add_argument(
      '--split-bn',
      action='store_true',
      help='Enable separate BN layers per augmentation split.',
    )
    
    # Model Exponential Moving Average
    parser.add_argument(
      '--model-ema',
      action='store_true',
      default=True,
      help='Enable tracking moving average of model weights',
    )
    parser.add_argument(
      '--model-ema-force-cpu',
      action='store_true',
      default=False,
      help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.',
    )
    parser.add_argument(
      '--model-ema-decay',
      type=float,
      default=0.9995,
      help='decay factor for model weights moving average. default: 0.9998',
    )
    
    # Distillation parameters
    parser.add_argument(
      '--teacher-model',
      default='regnety_160',
      type=str,
      metavar='MODEL',
      help='Name of teacher model to train. default: "regnety_160"',
    )
    parser.add_argument(
      '--teacher-path',
      type=str,
      default='https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth',
      help='Location of teacher model',
    )
    parser.add_argument(
      '--distillation-type',
      default='none',
      choices=['none', 'soft', 'hard'],
      type=str,
      help='DeiT style hardening or standard softening.',
    )
    parser.add_argument(
      '--distillation-alpha',
      default=0.5,
      type=float,
      help='loss scale: alpha * base_loss + (1-alpha) * kd_loss',
    )
    parser.add_argument(
      '--distillation-tau',
      default=1.0,
      type=float,
      help='Temperature to soften teacher logits',
    )
    
    # Misc
    parser.add_argument(
      '--finetune',
      action='store_true',
      default=False,
      help='If used, loading loss_scaler will be disabled in resume',
    )
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed. default: 42')
    parser.add_argument('--worker-seeding', type=str, default='all', help='worker seed mode. default: all')
    parser.add_argument(
      '--log-interval',
      type=int,
      default=50,
      metavar='N',
      help='how many batches to wait before logging training status',
    )
    parser.add_argument(
      '--recovery-interval',
      type=int,
      default=0,
      metavar='N',
      help='how many batches to wait before writing recovery checkpoint',
    )
    parser.add_argument(
      '--checkpoint-hist',
      type=int,
      default=10,
      metavar='N',
      help='number of checkpoints to keep. default: 10',
    )
    parser.add_argument(
      '-j',
      '--workers',
      type=int,
      default=8,
      metavar='N',
      help='how many training processes to use. default: 8',
    )
    parser.add_argument(
      '--save-images',
      action='store_true',
      default=False,
      help='save images of input bathes every log interval for debugging',
    )
    parser.add_argument(
      '--amp',
      action='store_true',
      default=False,
      help='use NVIDIA Apex AMP or Native AMP for mixed precision training',
    )
    parser.add_argument(
      '--no-ddp-bb',
      action='store_true',
      default=False,
      help='Force broadcast buffers for native DDP to off.',
    )
    parser.add_argument(
      '--channels-last',
      action='store_true',
      default=False,
      help='Use channels_last memory layout',
    )
    parser.add_argument(
      '--pin-mem',
      action='store_true',
      default=False,
      help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.',
    )
    parser.add_argument(
      '--no-prefetcher',
      action='store_true',
      default=False,
      help='disable fast prefetcher',
    )
    parser.add_argument(
      '--output',
      default='',
      type=str,
      metavar='PATH',
      help='path to output folder. default: none, current dir',
    )
    parser.add_argument(
      '--experiment',
      default='',
      type=str,
      metavar='NAME',
      help='name of train experiment, name of sub-folder for output',
    )
    parser.add_argument(
      '--eval-metric',
      default='top1',
      type=str,
      metavar='EVAL_METRIC',
      help='Best metric. default: "top1"',
    )
    parser.add_argument(
      '--tta',
      type=int,
      default=0,
      metavar='N',
      help='test/inference time augmentation (oversampling) factor. 0=None. default: 0',
    )
    parser.add_argument(
      '--use-multi-epochs-loader',
      action='store_true',
      default=False,
      help='use the multi-epochs-loader to save time at the beginning of every epoch',
    )
    parser.add_argument(
      '--torchscript',
      dest='torchscript',
      action='store_true',
      help='convert model torchscript for inference',
    )
    parser.add_argument(
      '--imagenet-trainset-size',
      default=1281167,
      help='size of imagenet training set for weight decay scheduling.',
    )
    
    # do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    
    # the main argument parser parses the rest of the args, the usual
    # default will have been overridden if config file specified
    args = parser.parse_args(remaining)
    
    # cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    
    return args, args_text


def main():
    setup_default_logging()
    args, args_text = _parse_args()
    
    # setup distributed
    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = 1
    args.rank = 0  # global rank
    
    if args.distributed:
        dist.init_process_group(backend='nccl', init_method='env://')
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
        torch.cuda.set_device(args.rank)
        _logger.info(
          f'training in distributed mode with multiple processes, 1 GPU per process. '
          f'process: {args.rank}, total {args.world_size}'
        )
    else:
        _logger.info('training with a single process on 1 GPU.')
    
    assert args.rank >= 0
    
    # resolve AMP
    use_amp = None
    if args.amp and has_native_amp:
        use_amp = 'native'
        _logger.info('using native PyTorch AMP.')
    else:
        _logger.info('AMP is not used')
    
    # random state
    random_seed(args.seed, args.rank)
    
    model = create_model(
      args.model,
      pretrained=args.pretrained,
      num_classes=args.num_classes,
      drop_rate=args.drop,
      drop_path_rate=args.drop_path,
      drop_block_rate=args.drop_block,
      global_pool=args.gp,
      bn_tf=args.bn_tf,
      bn_momentum=args.bn_momentum,
      bn_eps=args.bn_eps,
      scriptable=args.torchscript,
      checkpoint_path=args.initial_checkpoint,
    )
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'model must have `num_classes` attribute if not set on config.'
        args.num_classes = model.num_classes  # FIXME: handle model default vs config `num_classes` more elegantly
    
    if args.rank == 0:
        _logger.info(
          f'model {safe_model_name(args.model)} is created, '
          f'param count: {sum([m.numel() for m in model.parameters()])}'
        )
    
    data_config = resolve_data_config(vars(args), model=model, verbose=args.rank == 0)
    
    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'a split of 1 makes no sense'
        num_aug_splits = args.aug_splits
    
    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))
    
    # move model to GPU, enable channels last layout if set
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    
    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        assert not args.split_bn
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.rank == 0:
            _logger.info(
              'converted model to use SynchBatchNorm. WARNING: you may have issues if using '
              'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.'
            )
    
    if args.torchscript:
        assert not args.sync_bn, 'cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)
    
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
    
    # setup amp loss scaling and operator casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.rank == 0:
            _logger.info('using native PyTorch AMP. training in mixed precision.')
    else:
        if args.rank == 0:
            _logger.info('AMP is not enabled. training on float32.')
    
    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        if not args.finetune:
            # if folder give, pick the last checkpoint from a sorted list of checkpoints
            if os.path.isdir(args.resume):
                ckpt_paths = sorted(glob.glob(os.path.join(args.resume, '*.pth')))
                resume_path = ckpt_paths[-1]
                setattr(args, 'resume', resume_path)
                print(f'resuming from {resume_path}')
            
            resume_epoch = resume_checkpoint(
              model,
              args.resume,
              optimizer=None if args.no_resume_opt else optimizer,
              loss_scaler=None if args.no_resume_opt else loss_scaler,
              log_info=args.rank == 0
            )
        else:
            print('finetune option is selected, not loading optimizer state and loss_scaler')
            _ = resume_checkpoint(model, args.resume, optimizer=None, loss_scaler=None, log_info=args.rank == 0)
            
            data_config['crop_pct'] = 1.
            print(f'data config: {data_config}')
    
    # setup exponential moving average of model weights, SWA could be here too
    model_ema = None
    if args.model_ema:
        # important to cerate EMA model after cuda(), DP wrapper and AMP but before SyncBatchNorm and DDP wrapper
        model_ema = ModelEmaV2(model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        # do not load EMA model when running in finetune mode
        if args.resume and not args.finetune:
            print('loading EMA model')
            load_checkpoint(model_ema.module, args.resume, use_ema=True)
    
    # set distributed training
    if args.distributed:
        # NOTE: EMA model does not need to be wrapped by DDP
        if args.rank == 0:
            _logger.info('using native PyTorch DDP')
        model = NativeDDP(
          model,
          device_ids=[args.rank],
          broadcast_buffers=not args.no_ddp_bb,
        )
    
    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # specified `start_epoch` will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)
    
    # setup weight decay scheduler
    wd_scheduler = None
    if args.wd_schedule == 'cosine':
        wd_scheduler = CosineWDSchedule(
          optimizer=optimizer,
          eta_min=args.weight_decay * 0.1,
          t_max=int(args.epoch * args.imagenet_trainset_size // args.batch_size // args.world_size)
        )
    
    if args.rank == 0:
        _logger.info(f'scheduled epochs: {num_epochs}')
    
    # instantiate teacher model, if distillation is requested
    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f'creating teacher model: {args.teacher_model}')
        teacher_model = create_model(
          args.teacher_model,
          pretrained=False,
          num_classes=args.num_classes,
          global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.cuda()
        teacher_model.eval()
    
    # create the train and eval datasets
    train_dataset = create_dataset(
      args.dataset,
      root=args.data_dir,
      split=args.train_split,
      is_training=True,
      class_map=args.class_map,
      download=args.dataset_download,
      batch_size=args.batch_size,
      repeats=args.epoch_repeats
    )
    eval_dataset = create_dataset(
      args.dataset,
      root=args.data_dir,
      split=args.valid_split,
      is_training=False,
      class_map=args.class_map,
      download=args.dataset_download,
      batch_size=args.batch_size
    )
    
    # setup mixup/cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
          mixup_alpha=args.mixup,
          cutmix_alpha=args.cutmix,
          cutmix_minmax=args.cutmix_minmax,
          prob=args.mixup_prob,
          switch_prob=args.mixup_switch_prob,
          mode=args.mixup_mode,
          label_smoothing=args.smoothing,
          num_classes=args.num_classes
        )
        if args.prefetcher:
            assert not num_aug_splits  # collate_fn conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)
    
    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        train_dataset = AugMixDataset(train_dataset, num_splits=num_aug_splits)
    
    # create data loaders w/ augmentation pipeline
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    train_loader = create_loader(
      train_dataset,
      input_size=data_config['input_size'],
      batch_size=args.batch_size,
      is_training=True,
      use_prefetcher=args.prefetcher,
      no_aug=args.no_aug,
      re_prob=args.reprob,
      re_mode=args.remode,
      re_count=args.recount,
      re_split=args.resplit,
      scale=args.scale,
      ratio=args.ratio,
      hflip=args.hflip,
      vflip=args.vflip,
      color_jitter=args.color_jitter,
      auto_augment=args.aa,
      num_aug_repeats=args.aug_repeats,
      num_aug_splits=num_aug_splits,
      interpolation=train_interpolation,
      mean=data_config['mean'],
      std=data_config['std'],
      num_workers=args.workers,
      distributed=args.distributed,
      collate_fn=collate_fn,
      pin_memory=args.pin_mem,
      use_multi_epochs_loader=args.use_multi_epochs_loader,
      worker_seeding=args.worker_seeding
    )
    
    eval_loader = create_loader(
      eval_dataset,
      input_size=data_config['input_size'],
      batch_size=args.validation_batch_size or args.batch_size,
      is_training=False,
      use_prefetcher=args.prefetcher,
      interpolation=data_config['interpolation'],
      mean=data_config['mean'],
      std=data_config['std'],
      num_workers=args.workers,
      distributed=args.distributed,
      crop_pct=data_config['crop_pct'],
      pin_memory=args.pin_mem,
    )
    
    # setup loss function
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh, smoothing=args.smoothing)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()
    
    # use distillation loss wrapper, which returns base loss when distillation is disabled
    train_loss_fn = DistillationLoss(
      train_loss_fn, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )
    
    # setup checkpoint saver and evaluate metrics tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
              datetime.now().strftime('%Y%m%d-%H%M%S'),
              safe_model_name(args.model),
              str(data_config['input_size'][-1])
            ])
        output_dir = get_outdir(args.output if args.output else './output/train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(
          model=model,
          optimizer=optimizer,
          args=args,
          model_ema=model_ema,
          amp_scaler=loss_scaler,
          checkpoint_dir=output_dir,
          recovery_dir=output_dir,
          decreasing=decreasing,
          max_history=args.checkpoint_hist
        )
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)
    
    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            
            train_metrics = train_one_epoch(
              epoch,
              model,
              train_loader,
              optimizer,
              train_loss_fn,
              args,
              lr_scheduler=lr_scheduler,
              saver=saver,
              output_dir=output_dir,
              amp_autocast=amp_autocast,
              loss_scaler=loss_scaler,
              model_ema=model_ema,
              mixup_fn=mixup_fn,
              wd_scheduler=wd_scheduler,
              logger=_logger,
            )
        if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
            if args.rank == 0:
                _logger.info('distributing BatchNorm running means and vars')
            distribute_bn(model, args.world_size, args.dist_bn == 'reduce')
        
        eval_metrics = validate(model, eval_loader, validate_loss_fn, args, amp_autocast=amp_autocast, logger=_logger)
        
        if model_ema is not None and not args.model_ema_force_cpu:
            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
            ema_eval_metrics = validate(
              model_ema.module,
              eval_loader,
              validate_loss_fn,
              args,
              amp_autocast=amp_autocast,
              log_suffix=' [EMA]',
              logger=_logger
            )
            eval_metrics = ema_eval_metrics
        
        if lr_scheduler is not None:
            # step LR for next epoch
            lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])
        
        if output_dir is not None:
            update_summary(
              epoch,
              train_metrics,
              eval_metrics,
              os.path.join(output_dir, 'summary.csv'),
              write_header=best_metric is None
            )
        
        if saver is not None:
            # save proper checkpoint with eval metric
            save_metric = eval_metrics[eval_metric]
            best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)
    
    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info(f'** Best metric: {best_metric} (epoch {best_epoch}) **')


if __name__ == '__main__':
    main()
