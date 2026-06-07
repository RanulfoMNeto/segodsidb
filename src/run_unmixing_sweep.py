"""
@brief Run CNNAEU unmixing configs.

Single-image ODSI-DB configs default to R=auto, where train.py sets the number
of endmembers from the annotated classes present in the image mask.
"""

import argparse
import copy
import json
import pathlib
import subprocess
import sys
import tempfile

import torchseg.utils


def parse_cmdline_params():
    args = argparse.ArgumentParser(description='Run unmixing sweep.')
    args.add_argument('-c', '--conf', required=True, type=str,
                      help='config file path')
    args.add_argument('-l', '--logconf', default=None, type=str,
                      help='logger config file path')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable')
    args.add_argument('--image-path', default=None, type=str,
                      help='override single-image HSI path')
    args.add_argument('--unmixing-manifest', default=None, type=str,
                      help='manifest JSON updated by each single-image run')
    args.add_argument('--data-dir', default=None, type=str,
                      help='legacy override for dataset-based training')
    args.add_argument('--test-data-dir', default=None, type=str,
                      help='legacy override for dataset-based testing')
    return args.parse_args()


def resolve_data_dir(path):
    data_dir = pathlib.Path(path)
    if data_dir.is_dir():
        return str(data_dir)

    # The segmentation templates assume the dataset lives inside the checkout
    # as odsi_db/..., but in this workspace it commonly lives next to it.
    sibling = pathlib.Path.cwd().parent / path
    if sibling.is_dir():
        return str(sibling)

    raise FileNotFoundError(
        'ODSI-DB data directory not found: {}\n'
        'Tried: {}\n'
        '       {}\n'
        'Use --data-dir and --test-data-dir to point to your fold directories, '
        'for example:\n'
        '  --data-dir ../odsi_db/folds/fold_0/train '
        '--test-data-dir ../odsi_db/folds/fold_0/test'.format(
            path, data_dir.resolve(), sibling.resolve()))


def resolve_image_path(path):
    image_path = pathlib.Path(path)
    if image_path.is_file():
        return str(image_path)

    sibling = pathlib.Path.cwd().parent / path
    if sibling.is_file():
        return str(sibling)

    raise FileNotFoundError(
        'ODSI-DB image file not found: {}\n'
        'Tried: {}\n'
        '       {}'.format(path, image_path.resolve(), sibling.resolve()))


def apply_data_dir_overrides(config, args):
    if 'image_path' in config['data_loader']['args'] or \
            args.image_path is not None:
        image_path = args.image_path
        if image_path is None:
            image_path = config['data_loader']['args']['image_path']
        config['data_loader']['args']['image_path'] = \
            resolve_image_path(image_path)
        return

    train_dir = args.data_dir
    test_dir = args.test_data_dir

    if train_dir is None:
        train_dir = config['data_loader']['args']['data_dir']
    config['data_loader']['args']['data_dir'] = resolve_data_dir(train_dir)

    if 'testing' not in config:
        return

    for dataset in config['testing']['datasets']:
        dataset_dir = test_dir
        if dataset_dir is None:
            dataset_dir = dataset['args']['data_dir']
        dataset['args']['data_dir'] = resolve_data_dir(dataset_dir)


def is_auto_single_image_unmixing_config(config):
    data_loader = config.get('data_loader', {})
    data_args = data_loader.get('args', {})
    unmixing = config.get('unmixing', {})
    model = config.get('model', {})
    return (
        data_loader.get('type') == 'OdsiDbSingleImageUnmixingDataLoader'
        and 'image_path' in data_args
        and model.get('type') == 'CNNAEU'
        and unmixing.get('auto_from_training_mask', True)
    )


def run_train(config, args):
    if args.unmixing_manifest is not None:
        config.setdefault('unmixing', {})
        config['unmixing']['manifest_path'] = args.unmixing_manifest

    with tempfile.NamedTemporaryFile('w', suffix='.json',
                                    delete=False) as tmp:
        json.dump(config, tmp, indent=4)
        tmp_config_path = tmp.name

    cmd = [sys.executable, '-m', 'torchseg.train', '-c', tmp_config_path]
    if args.logconf is not None:
        cmd += ['-l', args.logconf]
    if args.device is not None:
        cmd += ['-d', args.device]

    subprocess.check_call(cmd)


def main():
    args = parse_cmdline_params()
    base_config = torchseg.utils.read_json(args.conf)
    apply_data_dir_overrides(base_config, args)

    if is_auto_single_image_unmixing_config(base_config):
        config = copy.deepcopy(base_config)
        config.pop('sweep', None)
        config['model']['args']['num_endmembers'] = 'auto'
        run_train(config, args)
        return

    sweep = base_config.get('sweep', {})
    num_endmembers_values = sweep.get('num_endmembers')
    if not num_endmembers_values:
        raise ValueError(
            'Config must define sweep.num_endmembers when '
            'unmixing.auto_from_training_mask is disabled.')

    base_name = base_config['name']
    for num_endmembers in num_endmembers_values:
        config = copy.deepcopy(base_config)
        config.pop('sweep', None)
        config['name'] = '{}_R{}'.format(base_name, num_endmembers)
        config['model']['args']['num_endmembers'] = num_endmembers
        run_train(config, args)


if __name__ == '__main__':
    main()
