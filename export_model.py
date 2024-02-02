import argparse
import os
from typing import Optional

import coremltools # pyright: ignore [reportMissingImports]
import torch

import models
from models.components import reparameterize_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='script to export coreml package file')
    parser.add_argument('--variant', type=str, required=True, help='provide FastViT model variant name.')
    parser.add_argument('--output-dir', type=str, default='.', help='provide location to save exported models.')
    parser.add_argument('--checkpoint', type=str, default=None, help='provide location of trained checkpoint.')
    return parser.parse_args()


def export(variant: str, output_dir: str, checkpoint: Optional[str] = None) -> None:
    """method to export coreml package for mobile inference.

    :param variant: FastViT model variant.
    :param output_dir: path to save exported model.
    :param checkpoint: path to trained checkpoint. Default: None
    """
    # create output directory.
    os.makedirs(output_dir, exist_ok=True)
    
    # random input tensor for tracing purposes.
    inputs = torch.rand(1, 3, 256, 256)
    inputs_tensor = [coremltools.TensorType(name='images', shape=inputs.shape)]
    
    # instantiate model variant.
    model = getattr(models, variant)()
    print(f'export and convert model: {variant}')
    
    # always re-parameterize before exporting
    reparameterized_model = reparameterize_model(model)
    if checkpoint is not None:
        print(f'load checkpint {checkpoint}')
        ckpt = torch.load(checkpoint)
        reparameterized_model.load_state_dict(ckpt['state_dict'])
    reparameterized_model.eval()
    
    # trace and export
    traced_model = torch.jit.trace(reparameterized_model, torch.Tensor(inputs))
    output_path = os.path.join(output_dir, variant)
    pt_name = output_path + '.pt'
    traced_model.save(pt_name)
    ml_model = coremltools.convert(
      model=pt_name, outputs=None, inputs=inputs_tensor, convert_to='mlprogram', debug=False
    )
    ml_model.save(output_path + '.mlpackage')


if __name__ == '__main__':
    args = _parse_args()
    export(args.variant, args.output_dir, args.checkpoint)
