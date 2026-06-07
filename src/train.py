"""
@brief  Main script to kick off the training. 
@author Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   1 Jun 2021.
"""

import argparse
import collections
import torch
import numpy as np

# My imports
import torchseg.config.parser
import torchseg.data_loader
import torchseg.model
import torchseg.machine
import torchseg.utils

try:
    import torchseg.unmixing_manifest as unmixing_manifest
except ModuleNotFoundError:
    import unmixing_manifest

# Fix random seeds for reproducibility
SEED = 18303
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def help(short_option):
    """
    @returns The string with the help information for each command line option.
    """
    help_msg = {
        '-c': 'config file path (default: None)',
        '-l': 'logger config file path (default: None)',
        '-r': 'path to latest checkpoint (default: None)',
        '-d': 'indices of GPUs to enable (default: all)',
    }
    return help_msg[short_option]


def parse_cmdline_params():
    """@returns The argparse args object."""
    args = argparse.ArgumentParser(description='PyTorch segmenter.')
    args.add_argument('-c', '--conf', default=None, type=str, help=help('-c'))
    args.add_argument('-l', '--logconf', default=None, type=str, help=help('-l'))
    args.add_argument('-r', '--resume', default=None, type=str, help=help('-r'))
    args.add_argument('-d', '--device', default=None, type=str, help=help('-d'))
    return args


def parse_config(args):
    """
    @brief Combines parameters from both JSON and command line into a 
    @param[in]  args  Argparse args object.
    @returns A torchseg.config.parser.ConfigParser object.
    """
    # Custom CLI options to modify the values provided in the JSON configuration
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning-rate'], type=float, 
            target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch-size'], type=int, 
            target='data_loader;args;batch_size'),
        CustomArgs(['--data-dir'], type=str, 
            target='data_loader;args;data_dir'),
        CustomArgs(['--image-path'], type=str,
            target='data_loader;args;image_path'),
        CustomArgs(['--unmixing-manifest'], type=str,
            target='unmixing;manifest_path'),
        CustomArgs(['--save-dir'], type=str, 
            target='machine;args;save_dir'),
    ]
    config = torchseg.config.parser.ConfigParser.from_args(args, options)
    return config


def _is_single_image_unmixing_config(config):
    data_loader_config = config.config.get('data_loader', {})
    data_loader_args = data_loader_config.get('args', {})
    return (
        data_loader_config.get('type') == 'OdsiDbSingleImageUnmixingDataLoader'
        and 'image_path' in data_loader_args
        and config.config.get('model', {}).get('type') == 'CNNAEU'
    )


def auto_configure_single_image_unmixing(config, logger=None):
    """
    @brief Set R and semantic reference classes from the training image mask.
    @details This uses the ODSI-DB annotation only to choose experiment
             metadata/hyperparameters. The unmixing loader still does not
             return labels and the CNNAEU training loss remains self-supervised.
    """
    if not _is_single_image_unmixing_config(config):
        return None

    cfg = config.config
    unmixing_cfg = cfg.setdefault('unmixing', {})
    auto_enabled = unmixing_cfg.get('auto_from_training_mask', True)
    model_args = cfg['model']['args']
    if not auto_enabled:
        if model_args.get('num_endmembers') == 'auto':
            raise ValueError(
                'model.args.num_endmembers is "auto", but '
                'unmixing.auto_from_training_mask is disabled. Set an '
                'integer num_endmembers or enable auto_from_training_mask.')
        return None

    data_args = cfg['data_loader']['args']
    image_path = data_args['image_path']
    mask_path = unmixing_cfg.get('mask_path')
    min_pixels = int(unmixing_cfg.get('auto_min_pixels', 1))

    class_info, resolved_mask_path = \
        torchseg.data_loader.OdsiDbDataLoader.LoadImage \
        .present_label_class_info_from_image(
            image_path, mask_path=mask_path, min_pixels=min_pixels)

    if class_info is None:
        raise FileNotFoundError(
            'Cannot set model.args.num_endmembers automatically because no '
            'ODSI-DB mask was found for {}. Add the sibling mask file or set '
            'unmixing.mask_path.'.format(image_path))

    if len(class_info) == 0:
        raise RuntimeError(
            'No labelled ODSI-DB classes with at least {} pixels were found '
            'in {}.'.format(min_pixels, resolved_mask_path))

    class_indices = [item['class_index'] for item in class_info]
    class_names = [item['class_name'] for item in class_info]
    pixel_counts = [item['pixels'] for item in class_info]
    num_endmembers = len(class_names)

    model_args['num_endmembers'] = num_endmembers
    unmixing_cfg['reference_class_names'] = class_names
    unmixing_cfg['auto_from_training_mask'] = True
    unmixing_cfg['auto_from_training_mask_status'] = 'applied'
    unmixing_cfg['auto_from_training_mask_info'] = {
        'image_path': image_path,
        'mask_path': resolved_mask_path,
        'min_pixels': min_pixels,
        'num_endmembers': num_endmembers,
        'class_indices': class_indices,
        'class_names': class_names,
        'pixel_counts': pixel_counts,
    }
    cfg.pop('sweep', None)

    torchseg.utils.write_json(cfg, config.save_dir / 'config.json')
    if logger is not None:
        logger.info(
            'Auto-configured CNNAEU from mask %s: num_endmembers=%d, '
            'classes=%s', resolved_mask_path, num_endmembers, class_names)
    return unmixing_cfg['auto_from_training_mask_info']


def auto_update_single_image_unmixing_manifest(config, logger=None):
    """
    @brief Register the best checkpoint of a single-image CNNAEU run.
    @details The manifest stores at most one run per image. If a previous run
             exists, it is replaced only when the new monitored score is better.
    """
    if not _is_single_image_unmixing_config(config):
        return None

    unmixing_cfg = config.config.get('unmixing', {})
    if not unmixing_cfg.get('manifest_auto_update', True):
        if logger is not None:
            logger.info('Skipping unmixing manifest update: disabled by config.')
        return None

    checkpoint_path = unmixing_manifest.latest_checkpoint(
        config.save_dir)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    manifest_path = unmixing_manifest.resolve_manifest_path(config)
    entry = unmixing_manifest.entry_from_training_run(
        config, config.save_dir, checkpoint_path, checkpoint=checkpoint)
    result = unmixing_manifest.update_manifest(
        manifest_path, entry)

    if logger is not None:
        action = 'updated' if result['updated'] else 'kept existing entry in'
        logger.info(
            'Unmixing manifest %s %s for %s (%s).',
            result['manifest_path'], action,
            result['image_basename'], result['reason'])
    return result


def main():
    args = parse_cmdline_params()
    config = parse_config(args)
    logger = config.get_logger('train')
    auto_configure_single_image_unmixing(config, logger)
    torchseg.model.configure_odsi_db_unmixing_from_config(config)

    # Note: the 'type' in config.json indicates the class, and the 'args' in
    # config.json will be passed as parameters to the constructor of that class

    # Setup data loader
    data_loader = config.init_obj('data_loader', torchseg.data_loader)
    valid_data_loader = data_loader.split_validation()

    # Create model 
    model = config.init_obj('model', torchseg.model)
    logger.info(model)

    # Prepare for (multi-device) GPU training
    device, device_ids = torchseg.utils.setup_gpu_devices(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # Get function handles of loss and evaluation metrics
    criterion = torchseg.model.loss.get_loss_function(config['loss'])
    metrics = [getattr(torchseg.model.metric, m) for m in config['metrics']]

    # Create optmizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    # Learning rate scheduler
    lr_scheduler = None
    if 'lr_scheduler' in config.config:
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler,
            optimizer)

    # Create learning machine
    machine_cls = getattr(torchseg.machine, config['machine']['type'])
    acc_steps = config['machine']['args'].get('acc_steps', 1)
    trainer = machine_cls(model, criterion, metrics,
                          optimizer,
                          config=config,
                          device=device,
                          data_loader=data_loader,
                          valid_data_loader=valid_data_loader,
                          lr_scheduler=lr_scheduler,
                          acc_steps=acc_steps)
    
    # Launch training
    trainer.train()
    auto_update_single_image_unmixing_manifest(config, logger)


if __name__ == '__main__':
    main()
