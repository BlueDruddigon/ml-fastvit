#!/usr/bin/env python3
""" ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets. It prioritizes
canonical PyTorch, standard Python style, and good performance. Repurpose as you see fit.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import csv
import glob
import json
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from functools import partial

import torch
import torch.nn as nn
from timm.data import create_dataset, create_loader, RealLabelsImagenet, resolve_data_config
from timm.layers import apply_test_time_pool
from timm.models import create_model, is_model, list_models, load_checkpoint
from timm.utils import accuracy, AverageMeter, natural_key, set_jit_legacy, setup_default_logging
from torch.nn.parallel import DataParallel as DP

import models
from models.components import reparameterize_model

try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    has_native_amp = False

_logger = logging.getLogger('validate')


def _parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
    parser.add_argument('data', metavar='DIR', help='Path to dataset')
    parser.add_argument(
      '--dataset',
      '-d',
      metavar='NAME',
      default='',
      help='Dataset type. default: ImageFolder/ImageTar if empty',
    )
    parser.add_argument(
      '--split',
      metavar='NAME',
      default='validation',
      help='Dataset split. default: validation',
    )
    parser.add_argument(
      '--num-samples',
      default=None,
      type=int,
      metavar='N',
      help='Manually specify num samples in dataset split, for IterableDatasets.'
    )
    parser.add_argument(
      '--dataset-download',
      action='store_true',
      default=False,
      help='Allow download of dataset for torch/ and tfds/ datasets that support it.',
    )
    parser.add_argument(
      '--class-map',
      default='',
      type=str,
      metavar='FILENAME',
      help='Path to class to idx mapping file. default: ""',
    )
    parser.add_argument('--input-key', default=None, type=str, help='Dataset key for input images.')
    parser.add_argument(
      '--input-img-mode', default=None, type=str, help='Dataset image conversion mode for input images.'
    )
    parser.add_argument('--target-key', default=None, type=str, help='Dataset key for target labels.')
    
    parser.add_argument(
      '--model',
      '-m',
      metavar='NAME',
      default='dpn92',
      help='Model architecture, default: dpn92',
    )
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='Use pre-trained model')
    parser.add_argument(
      '-j',
      '--workers',
      default=4,
      type=int,
      metavar='N',
      help='Number of data loading workers. default: 2',
    )
    parser.add_argument(
      '-b',
      '--batch-size',
      default=256,
      type=int,
      metavar='N',
      help='Mini-batch size. default: 256',
    )
    parser.add_argument(
      '--img-size',
      default=None,
      type=int,
      metavar='N',
      help='Input image dimension, uses model default if empty',
    )
    parser.add_argument(
      '--in-chans', type=int, default=None, metavar='N', help='Image input channels, default: None => 3'
    )
    parser.add_argument(
      '--input-size',
      default=[3, 256, 256],
      nargs=3,
      type=int,
      metavar='N N N',
      help='Input all image dimensions (d h w, e.g. --input-size 3 256 256), uses model default if empty',
    )
    parser.add_argument(
      '--crop-pct',
      default=None,
      type=float,
      metavar='N',
      help='Input image center crop pct',
    )
    parser.add_argument(
      '--crop-mode',
      default=None,
      type=str,
      metavar='N',
      help='Input image crop mode (squash, border, center). Model default is None.'
    )
    parser.add_argument('--crop-border-pixels', type=int, default=None, help='Crop pixels from image border.')
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
    parser.add_argument('--num-classes', type=int, default=None, help='Number classes in dataset')
    parser.add_argument(
      '--gp',
      default=None,
      type=str,
      metavar='POOL',
      help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.',
    )
    parser.add_argument(
      '--log-freq',
      default=10,
      type=int,
      metavar='N',
      help='batch logging frequency. default: 10',
    )
    parser.add_argument(
      '--checkpoint',
      default='',
      type=str,
      metavar='PATH',
      help='Path to latest checkpoint. default: none',
    )
    parser.add_argument('--num-gpu', type=int, default=1, help='Number of GPUS to use')
    parser.add_argument('--test-pool', dest='test_pool', action='store_true', help='Enable test time pool')
    parser.add_argument(
      '--no-prefetcher',
      action='store_true',
      default=False,
      help='Disable fast prefetcher',
    )
    parser.add_argument(
      '--pin-mem',
      action='store_true',
      default=False,
      help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.',
    )
    parser.add_argument(
      '--channels-last',
      action='store_true',
      default=False,
      help='Use channels_last memory layout',
    )
    parser.add_argument('--device', default='cuda', type=str, help='Device (accelerator) to use.')
    parser.add_argument(
      '--amp',
      action='store_true',
      default=False,
      help='Use AMP mixed precision. Defaults to native Torch AMP.',
    )
    parser.add_argument('--amp-dtype', default='float16', type=str, help='Lower precision AMP dtype. default: float16')
    parser.add_argument(
      '--tf-preprocessing',
      action='store_true',
      default=False,
      help='Use Tensorflow preprocessing pipeline (require CPU TF installed)',
    )
    parser.add_argument(
      '--use-ema',
      dest='use_ema',
      action='store_true',
      help='Use ema version of weights if present',
    )
    
    parser.add_argument(
      '--torchscript',
      dest='torchscript',
      action='store_true',
      help='Convert model torchscript for inference',
    )
    parser.add_argument(
      '--legacy-jit',
      dest='legacy_jit',
      action='store_true',
      help='Use legacy jit mode for pytorch 1.5/1.5.1/1.6 to get back fusion performance',
    )
    
    parser.add_argument(
      '--results-file',
      default='',
      type=str,
      metavar='FILENAME',
      help='Output csv file for validation results (summary)',
    )
    parser.add_argument(
      '--results-format', default='csv', type=str, help='Format for results file one of (csv, json). default: csv'
    )
    parser.add_argument(
      '--real-labels',
      default='',
      type=str,
      metavar='FILENAME',
      help='Real labels JSON file for imagenet evaluation',
    )
    parser.add_argument(
      '--valid-labels',
      default='',
      type=str,
      metavar='FILENAME',
      help='Valid label indices txt file for validation of partial label space',
    )
    parser.add_argument(
      '--use-inference-mode',
      dest='use_inference_mode',
      action='store_true',
      default=False,
      help='Use inference mode version of model definition.',
    )
    
    return parser.parse_args()


def validate(args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    device = torch.device(args.device)
    
    # resolve AMP arguments based on PyTorch / Apex is_availability
    amp_autocast = suppress  # do nothing
    if args.amp and has_native_amp:
        assert args.amp_dtype in ('float16', 'bfloat16')
        amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        _logger.info('validating in mixed precision with native PyTorch AMP.')
    else:
        _logger.info('validating in float32. AMP is not enabled')
    
    if args.legacy_jit:
        set_jit_legacy()
    
    # create model
    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]
    
    model = create_model(
      args.model,
      pretrained=args.pretrained,
      num_classes=args.num_classes,
      in_chans=in_chans,
      global_pool=args.gp,
      scriptable=args.torchscript,
      inference_mode=args.use_inference_mode
    )
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'model must have `num_classes` attribute if not set on config.'
        args.num_classes = model.num_classes
    
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)
    
    # re-parameterize model
    model.eval()
    if not args.use_inference_mode:
        _logger.info(f're-parameterizing model {args.model}')
        model = reparameterize_model(model)
    setattr(model, 'pretrained_cfg', model.__dict__['default_cfg'])
    
    data_config = resolve_data_config(vars(args), model=model, use_test_size=True, verbose=True)
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config, use_test_size=True)
    
    if args.torchscript:
        torch.jit.optimized_execution(True)
        model = torch.jit.script(model)
    
    model = model.to(device=device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    
    if args.num_gpu > 1:
        model = DP(model, device_ids=list(range(args.num_gpu)))
    
    criterion = nn.CrossEntropyLoss().to(device=device)
    
    if args.input_img_mode is None:
        input_img_mode = 'RGB' if data_config['input_size'][0] == 3 else 'L'
    else:
        input_img_mode = args.input_img_mode
    dataset = create_dataset(
      root=args.data,
      name=args.dataset,
      split=args.split,
      download=args.dataset_download,
      load_bytes=args.tf_preprocessing,
      class_map=args.class_map,
      num_samples=args.num_samples,
      input_key=args.input_key,
      target_key=args.target_key,
      input_img_mode=input_img_mode
    )
    
    if args.valid_labels:
        with open(args.valid_labels, 'r') as f:
            valid_labels = {int(line.rstrip()) for line in f}
            valid_labels = [i in valid_labels for i in range(args.num_classes)]
    else:
        valid_labels = None
    
    if args.real_labels:
        real_labels = RealLabelsImagenet(dataset.filenames(basename=True), real_json=args.real_labels)
    else:
        real_labels = None
    
    crop_pct = 1.0 if test_time_pool else data_config["crop_pct"]
    loader = create_loader(
      dataset,
      input_size=data_config['input_size'],
      batch_size=args.batch_size,
      use_prefetcher=args.prefetcher,
      interpolation=data_config['interpolation'],
      mean=data_config['mean'],
      std=data_config['std'],
      num_workers=args.workers,
      crop_pct=crop_pct,
      crop_mode=data_config['crop_mode'],
      crop_border_pixels=args.crop_border_pixels,
      pin_memory=args.pin_mem,
      device=device,
      tf_preprocessing=args.tf_preprocessing,
    )
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    model.eval()
    with torch.no_grad():
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        input = torch.randn((args.batch_size, ) + tuple(data_config['input_size'])).to(device=device)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        with amp_autocast():
            model(input)
        
        end = time.time()
        for batch_idx, (input, target) in enumerate(loader):
            if args.no_prefetcher:
                target = target.to(device=device)
                input = input.to(device=device)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)
            
            # compute output
            with amp_autocast():
                output = model(input)
                
                if valid_labels is not None:
                    output = output[:, valid_labels]
                loss = criterion(output, target)
            
            if real_labels is not None:
                real_labels.add_result(output)
            
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if batch_idx % args.log_freq == 0:
                _logger.info(
                  f'Test: [{batch_idx:>4d}/{len(loader)}] '
                  f'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {input.size(0) / batch_time.avg:>7.2f}/s) '
                  f'Loss: {losses.val:>7.4f} ({losses.avg:>6.4f}) '
                  f'Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f}) '
                  f'Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})'
                )
    
    if real_labels is not None:
        # real labels mode replaces topk values at the end
        top1a, top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
    else:
        top1a, top5a = top1.avg, top5.avg
    results = OrderedDict(
      top1=round(top1a, 4),
      top1_err=round(100 - top1a, 4),
      top5=round(top5a, 4),
      top5_err=round(100 - top5a, 4),
      img_size=data_config['input_size'][-1],
      cropt_pct=crop_pct,
      interpolation=data_config['interpolation'],
    )
    
    _logger.info(
      f' * Acc@1 {results["top1"]:.3f} ({results["top1_err"]:.3f}) '
      f'Acc@5 {results["top5"]:.3f} ({results["top5_err"]:.3f})'
    )
    
    return results


def main():
    setup_default_logging()
    args = _parse_args()
    
    model_cfgs = []
    model_names = []
    if os.path.isdir(args.checkpoint):
        # validate all checkpoints in a path with same model
        checkpoints = glob.glob(args.checkpoint + '/*.pth.tar')
        checkpoints += glob.glob(args.checkpoint + '/*.pth')
        model_names = list_models(args.model)
        model_cfgs = [(args.model, c) for c in sorted(checkpoints, key=natural_key)]
    else:
        if args.model == 'all':
            # validate all models in a list of names with pretrained checkpoints
            args.pretrained = True
            model_names = list_models(pretrained=True, exclude_filters=['*_in21k', '*_in22k'])
            model_cfgs = [(n, '') for n in model_names]
        elif not is_model(args.model):
            # model name doesn't exist, try as wildcard filter
            model_names = list_models(args.model, pretrained=True)
            model_cfgs = [(n, '') for n in model_names]
        
        if not model_cfgs and os.path.isfile(args.model):
            with open(args.model) as f:
                model_names = [line.rstrip() for line in f]
            model_cfgs = [(n, None) for n in model_names if n]
    
    if len(model_cfgs):
        results_file = args.results_file or './results-all.csv'
        _logger.info(f'running bulk validation on these pretrained models: {", ".join(model_names)}')
        results = []
        try:
            start_batch_size = args.start_batch_size
            for m, c in model_cfgs:
                batch_size = start_batch_size
                args.model = m
                args.checkpoint = c
                result = OrderedDict(model=args.model)
                r = {}
                while not r and batch_size >= args.num_gpu:
                    if torch.cuda.is_available() and 'cuda' in args.device:
                        torch.cuda.empty_cache()
                    try:
                        args.batch_size = batch_size
                        print(f'validating with batch size: {args.batch_size}')
                        r = validate(args)
                    except RuntimeError as e:
                        if batch_size <= args.num_gpu:
                            print('validation failed with no ability to reduce batch size. exitting')
                            raise e
                        batch_size = max(batch_size // 2, args.num_gpu)
                        print('validation failed, reducing batch size by 50%')
                result.update(r)
                if args.checkpoint:
                    result['checkpoint'] = args.checkpoint
                results.append(result)
        except KeyboardInterrupt:
            pass
        results = sorted(results, key=lambda x: x['top1'], reverse=True)
        if len(results):
            write_results(results_file, results)
    else:
        validate(args)
    
    if args.results_file:
        write_results(args.results_file, results, format=args.results_format)
    
    # output results in JSON to stdout w/ delimiters for runner script
    print(f'--result\n{json.dumps(results, indent=4)}')


def write_results(results_file: str, results: dict, format: str = 'csv') -> None:
    with open(results_file, 'w') as cf:
        if format == 'json':
            json.dump(results, cf, indent=4)
        else:
            if not isinstance(results, (list, tuple)):
                results = [results]
            if not results:
                return
            dw = csv.DictWriter(cf, fieldnames=results[0].keys())
            dw.writeheader()
            for r in results:
                dw.writerow(r)
            cf.flush()


if __name__ == '__main__':
    main()

