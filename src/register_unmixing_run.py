"""
@brief Register an already trained single-image CNNAEU run in the manifest.
"""

import argparse
import pathlib

import torchseg.utils

try:
    import torchseg.unmixing_manifest as unmixing_manifest
except ModuleNotFoundError:
    import unmixing_manifest

try:
    import torch
except ImportError:
    torch = None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Register an existing CNNAEU run in the unmixing manifest.')
    parser.add_argument('--run-dir', required=True, type=str,
                        help='Run directory containing config.json/model_best.pth.')
    parser.add_argument('--manifest', default=None, type=str,
                        help='Manifest path. Defaults to auto from image fold.')
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = pathlib.Path(args.run_dir)
    config_path = run_dir / 'config.json'
    if not config_path.is_file():
        raise FileNotFoundError('Missing config.json in {}'.format(run_dir))

    config = torchseg.utils.read_json(config_path)
    checkpoint_path = unmixing_manifest.latest_checkpoint(run_dir)
    checkpoint = None
    if torch is not None:
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location='cpu', weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if args.manifest is None:
        image_path = config['data_loader']['args']['image_path']
        manifest_path = unmixing_manifest.infer_manifest_path(
            image_path)
    else:
        manifest_path = pathlib.Path(args.manifest)

    entry = unmixing_manifest.entry_from_training_run(
        config, run_dir, checkpoint_path, checkpoint=checkpoint)
    result = unmixing_manifest.update_manifest(
        manifest_path, entry)
    print(result)


if __name__ == '__main__':
    main()
