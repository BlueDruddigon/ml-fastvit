import logging
import os
import time
from argparse import Namespace
from contextlib import suppress
from typing import Optional, OrderedDict, Union

import torch
import torch.nn as nn
import torchvision
from timm.data.loader import PrefetchLoader
from timm.models import model_parameters
from timm.scheduler.scheduler import Scheduler
from timm.utils import (
  accuracy,
  ApexScaler,
  AverageMeter,
  CheckpointSaver,
  dispatch_clip_grad,
  ModelEma,
  ModelEmaV2,
  NativeScaler,
  reduce_tensor,
)
from torch.optim import Optimizer


def train_one_epoch(
  epoch: int,
  model: nn.Module,
  loader: PrefetchLoader,
  optimizer: Optimizer,
  loss_fn: nn.Module,
  args: Namespace,
  lr_scheduler: Optional[Scheduler] = None,
  saver: Optional[CheckpointSaver] = None,
  output_dir: Optional[str] = None,
  amp_autocast: Union[suppress, torch.cuda.amp.autocast] = suppress,
  loss_scaler: Optional[Union[ApexScaler, NativeScaler]] = None,
  model_ema: Optional[Union[ModelEma, ModelEmaV2]] = None,
  mixup_fn: Optional[None] = None,
  wd_scheduler: Optional[None] = None,
  logger: Optional[logging.Logger] = None
) -> OrderedDict[str, float]:
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False
    
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_timer_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    
    model.train()
    
    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        
        with amp_autocast():
            output = model(input)
            loss = loss_fn(input, output, target)
        
        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))
        
        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
              loss,
              optimizer,
              clip_grad=args.clip_grad,
              clip_mode=args.clip_mode,
              parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
              create_graph=second_order
            )
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                  model_parameters(model, exclude_head='agc' in args.clip_mode),
                  value=args.clip_grad,
                  mode=args.clip_mode
                )
            optimizer.step()
        
        if model_ema is not None:
            model_ema.update(model)
        
        torch.cuda.synchronize()
        num_updates += 1
        batch_timer_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            wd0 = list(optimizer.param_groups)[0]['weight_decay']
            wd1 = list(optimizer.param_groups)[1]['weight_decay']
            
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))
            
            if args.local_rank == 0:
                logger.info(
                  f'Train: {epoch} [{batch_idx:>4d}/{len(loader)} ({100. * batch_idx /last_idx:>3.0f}%)] '
                  f'Loss: {losses_m.val:#.4g} ({losses_m.avg:#.3g}) '
                  f'Time: {batch_timer_m.val:.3f}s, {input.size(0) * args.world_size / batch_timer_m.val:>7.2f}/s '
                  f'({batch_timer_m.avg:.3f}s, {input.size(0) * args.world_size / batch_timer_m.avg:>7.2f}/s) '
                  f'LR: {lr:.3e}, WD0: {wd0:.6e}, WD1: {wd1:.6e} '
                  f'Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})'
                )
                
                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                      input, os.path.join(output_dir, f'train-batch-{batch_idx}.jpg'), padding=0, normalize=True
                    )
        
        if saver is not None and args.recovery_interval and (last_batch or (batch_idx+1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)
        
        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)
        if wd_scheduler is not None:
            wd_scheduler.update_weight_decay(optimizer)
        
        end = time.time()  # end for
    
    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()
    
    return OrderedDict([('loss', losses_m.avg)])


def validate(
  model: nn.Module,
  loader: PrefetchLoader,
  loss_fn: nn.Module,
  args: Namespace,
  amp_autocast: Union[suppress, torch.cuda.amp.autocast] = suppress,
  log_suffix: str = '',
  logger: Optional[logging.Logger] = None
) -> OrderedDict[str, float]:
    batch_timer_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    
    model.eval()
    
    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)
            
            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]
            
            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]
            
            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data
            
            torch.cuda.synchronize()
            
            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))
            
            batch_timer_m.update(time.time() - end)
            end = time.time()
            
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                logger.info(
                  f'{log_name}: [{batch_idx:>4d}/{last_idx}] '
                  f'Time: {batch_timer_m.val:.3f} ({batch_timer_m.avg:.3f}) '
                  f'Loss: {losses_m.val:>7.4f} ({losses_m.avg:>7.4f}) '
                  f'Acc@1: {top1_m.val:>7.4f} ({top1_m.avg:>7.4f}) '
                  f'Acc@5: {top5_m.val:>7.4f} ({top5_m.avg:>7.4f})'
                )
    
    return OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
