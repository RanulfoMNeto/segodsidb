"""
@brief Test trained CNNAEU unmixing checkpoints.
"""

import argparse
import copy
import json
import os
import pathlib

import numpy as np
import torch
import tqdm

import torchseg.data_loader
import torchseg.model
import torchseg.utils


def parse_args():
    args = argparse.ArgumentParser(description='Test unmixing checkpoint.')
    args.add_argument('-r', '--resume', required=True, type=str,
                      help='checkpoint path, usually model_best.pth')
    args.add_argument('-c', '--conf', default=None, type=str,
                      help='optional config override; defaults to checkpoint config.json')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable')
    args.add_argument('-o', '--output', default=None, type=str,
                      help='optional JSON output path')
    args.add_argument('--image-path', default=None, type=str,
                      help='override single-image HSI path')
    args.add_argument('--test-data-dir', default=None, type=str,
                      help='legacy override for all testing dataset directories')
    args.add_argument('--max-batches', default=None, type=int,
                      help='stop after this many batches, useful for smoke tests')
    args.add_argument('--mapping-source', default='auto',
                      choices=['auto', 'validation', 'batch', 'mask',
                               'reference', 'none'],
                      help='source for semantic endmember/class mapping')
    args.add_argument('--mapping-max-batches', default=None, type=int,
                      help='limit validation batches used to learn mapping')
    args.add_argument('--mapping-class-scope', default='all-interest',
                      choices=['all-interest', 'present-only'],
                      help='classes eligible in mask/validation Hungarian '
                           'mapping: all configured reference classes, or '
                           'only classes with labelled pixels')
    args.add_argument('--mask-path', default=None, type=str,
                      help='optional ODSI mask override used with '
                           '--mapping-source mask/auto for single-image '
                           'unmixing')
    args.add_argument('--reference-endmembers', default=None, type=str,
                      help='external .npy/.npz/.csv reference endmember '
                           'spectra used when --mapping-source reference')
    args.add_argument('--reference-class-indices', default=None, type=str,
                      help='optional comma-separated integers or .npy/.csv/.txt '
                           'file with one class/reference index per spectrum')
    args.add_argument('--reference-class-names', default=None, type=str,
                      help='optional comma-separated names or text file with '
                           'one display name per reference spectrum')
    args.add_argument('--reference-max-sad', default=None, type=float,
                      help='optional maximum SAD in radians for accepting a '
                           'reference Hungarian assignment')
    args.add_argument('--eval-split', default='all',
                      choices=['all', 'train', 'validation'],
                      help='patch split to evaluate for single-image configs')
    return args.parse_args()


def reference_valid_for_mapping_scope(pixel_counts, mapping_class_scope):
    """
    @brief Return the reference_valid mask for Hungarian mapping.
    @details all-interest keeps every configured reference class eligible.
             present-only reproduces the older behavior: only classes with
             labelled pixels in the mask/validation split are eligible.
    """
    if mapping_class_scope == 'all-interest':
        return None
    if mapping_class_scope == 'present-only':
        return pixel_counts > 0
    raise ValueError('Unknown mapping_class_scope: {}'.format(
        mapping_class_scope))


def resolve_data_dir(path):
    data_dir = pathlib.Path(path)
    if data_dir.is_dir():
        return str(data_dir)

    sibling = pathlib.Path.cwd().parent / path
    if sibling.is_dir():
        return str(sibling)

    raise FileNotFoundError(
        'ODSI-DB data directory not found: {}\n'
        'Tried: {}\n'
        '       {}'.format(path, data_dir.resolve(), sibling.resolve()))


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


def resolve_file_path(path, description='file'):
    file_path = pathlib.Path(path)
    if file_path.is_file():
        return str(file_path)

    sibling = pathlib.Path.cwd().parent / path
    if sibling.is_file():
        return str(sibling)

    raise FileNotFoundError(
        '{} not found: {}\n'
        'Tried: {}\n'
        '       {}'.format(
            description, path, file_path.resolve(), sibling.resolve()))


def infer_odsi_mask_path(image_path):
    image_path = pathlib.Path(image_path)
    candidates = []
    if image_path.suffix:
        candidates.append(
            image_path.with_name(image_path.stem + '_masks' +
                                 image_path.suffix))
    candidates.append(image_path.with_name(image_path.name + '_masks.tif'))

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def resolve_mask_path(mask_path, image_path=None, required=False):
    if mask_path is not None:
        return resolve_file_path(mask_path, 'ODSI mask file')

    inferred = None if image_path is None else infer_odsi_mask_path(image_path)
    if inferred is not None:
        return inferred

    if required:
        raise FileNotFoundError(
            'ODSI mask file not found. Expected a sibling file named '
            '<image_stem>_masks.tif or pass --mask-path explicitly.')
    return None


def is_single_image_config(config):
    return 'image_path' in config['data_loader']['args']


def read_config(args):
    resume = pathlib.Path(args.resume)
    if args.conf is None:
        config_path = resume.parent / 'config.json'
    else:
        config_path = pathlib.Path(args.conf)

    config = torchseg.utils.read_json(config_path)
    if args.image_path is not None:
        config['data_loader']['args']['image_path'] = args.image_path

    if is_single_image_config(config):
        config['data_loader']['args']['image_path'] = resolve_image_path(
            config['data_loader']['args']['image_path'])
        return config

    if args.test_data_dir is not None:
        for dataset in config['testing']['datasets']:
            dataset['args']['data_dir'] = args.test_data_dir

    for dataset in config['testing']['datasets']:
        dataset['args']['data_dir'] = resolve_data_dir(
            dataset['args']['data_dir'])

    return config


def load_model(config, checkpoint_path, device):
    model = getattr(torchseg.model, config['model']['type'])(
        **config['model']['args'])
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device,
                                weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict']
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k.replace('module.', '', 1): v
                      for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, checkpoint


def safe_class_name(class_idx, class_names_by_index=None):
    class_idx = int(class_idx)
    if class_idx < 0:
        return None
    if class_names_by_index is not None and class_idx in class_names_by_index:
        return class_names_by_index[class_idx]

    idx2class = torchseg.data_loader.OdsiDbDataLoader.OdsiDbDataset.classnames
    if class_idx in idx2class:
        return idx2class[class_idx]
    return 'reference_{}'.format(class_idx)


def mapping_records(mapping, class_names_by_index=None):
    records = []
    for endmember_idx, class_idx in enumerate(mapping.tolist()):
        class_idx = int(class_idx)
        records.append({
            'endmember': int(endmember_idx),
            'class_index': class_idx,
            'class_name': safe_class_name(class_idx, class_names_by_index)
                if class_idx >= 0 else None,
        })
    return records


def _looks_like_wavelength_column(values):
    values = np.asarray(values)
    finite = np.isfinite(values)
    if values.ndim != 1 or finite.sum() < 2:
        return False
    values = values[finite]
    return (
        np.all(np.diff(values) > 0)
        and np.nanmin(values) >= 100.0
        and np.nanmax(values) <= 3000.0
    )


def _first_array_from_npz(path):
    with np.load(path) as data:
        if 'endmembers' in data:
            return np.asarray(data['endmembers'])
        for key in data.files:
            array = np.asarray(data[key])
            if array.ndim == 2:
                return array
    raise ValueError(
        'Reference NPZ must contain a 2D array, preferably named '
        '"endmembers".')


def _load_csv_array(path, delimiter=','):
    first_line = pathlib.Path(path).read_text().splitlines()[0]
    has_header = any(ch.isalpha() for ch in first_line)
    if has_header:
        structured = np.genfromtxt(path, delimiter=delimiter, names=True,
                                   dtype=float)
        if structured.dtype.names is not None:
            names = list(structured.dtype.names)
            data_columns = []
            for name in names:
                lowered = name.lower()
                if lowered in ('wavelength', 'wavelength_nm', 'wl', 'lambda'):
                    continue
                data_columns.append(np.asarray(structured[name]))
            if not data_columns:
                raise ValueError('CSV file has no numeric spectra columns.')
            return np.column_stack(data_columns)

    return np.loadtxt(path, delimiter=delimiter)


def _orient_reference_endmembers(array, expected_bands=None, path=None):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(
            'Reference endmembers must be a 2D array, got shape {}.'.format(
                array.shape))

    if array.shape[1] > 1 and _looks_like_wavelength_column(array[:, 0]):
        array = array[:, 1:]

    if expected_bands is not None:
        expected_bands = int(expected_bands)
        if array.shape[1] == expected_bands:
            pass
        elif array.shape[0] == expected_bands:
            array = array.T
        else:
            raise ValueError(
                'Reference endmembers in {} have shape {}, but the model '
                'expects {} spectral bands. Expected either (n_refs, {}) or '
                '({}, n_refs).'.format(
                    path, array.shape, expected_bands, expected_bands,
                    expected_bands))

    if not np.isfinite(array).all():
        raise ValueError('Reference endmembers contain NaN or infinite values.')
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError('Reference endmembers cannot be empty.')

    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError('Reference endmembers cannot contain all-zero spectra.')

    return array.astype(np.float32, copy=False)


def load_reference_endmembers(path, expected_bands=None):
    path = pathlib.Path(resolve_file_path(path, 'reference endmember file'))
    suffix = path.suffix.lower()
    if suffix == '.npy':
        array = np.load(path)
    elif suffix == '.npz':
        array = _first_array_from_npz(path)
    elif suffix == '.csv':
        array = _load_csv_array(path)
    elif suffix == '.txt':
        try:
            array = _load_csv_array(path)
        except ValueError:
            array = np.loadtxt(path)
    else:
        raise ValueError(
            'Unsupported reference endmember format: {}. Use .npy, .npz, '
            '.csv, or .txt.'.format(path.suffix))

    return _orient_reference_endmembers(
        array, expected_bands=expected_bands, path=str(path)), str(path)


def _parse_int_sequence(value):
    return np.asarray([
        int(item.strip())
        for item in value.replace('\n', ',').split(',')
        if item.strip() != ''
    ], dtype=np.int64)


def load_reference_class_indices(value, n_references):
    if value is None:
        return np.arange(n_references, dtype=np.int64)

    maybe_path = pathlib.Path(value)
    sibling = pathlib.Path.cwd().parent / value
    if maybe_path.is_file() or sibling.is_file():
        path = pathlib.Path(resolve_file_path(value, 'reference class index file'))
        suffix = path.suffix.lower()
        if suffix == '.npy':
            indices = np.load(path)
        elif suffix == '.npz':
            with np.load(path) as data:
                key = 'class_indices' if 'class_indices' in data \
                    else data.files[0]
                indices = np.asarray(data[key])
        else:
            try:
                indices = np.loadtxt(path, delimiter=',', dtype=np.int64)
            except ValueError:
                indices = np.loadtxt(path, dtype=np.int64)
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    else:
        indices = _parse_int_sequence(value)

    if indices.shape[0] != n_references:
        raise ValueError(
            'Expected {} reference class indices, got {}.'.format(
                n_references, indices.shape[0]))
    if np.any(indices < 0):
        raise ValueError('Reference class indices must be non-negative.')
    if np.unique(indices).shape[0] != indices.shape[0]:
        raise ValueError('Reference class indices must be unique.')
    return indices.astype(np.int64, copy=False)


def _parse_name_sequence(value):
    return [item.strip() for item in value.replace('\n', ',').split(',')
            if item.strip() != '']


def load_reference_class_names(value, n_references, class_indices,
                               default_prefix=None):
    if value is None:
        if default_prefix is not None:
            return {
                int(class_idx): '{}_{}'.format(default_prefix, pos)
                for pos, class_idx in enumerate(class_indices.tolist())
            }
        return {}

    maybe_path = pathlib.Path(value)
    sibling = pathlib.Path.cwd().parent / value
    if maybe_path.is_file() or sibling.is_file():
        path = pathlib.Path(resolve_file_path(value, 'reference class name file'))
        names = [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() != ''
        ]
    else:
        names = _parse_name_sequence(value)

    if len(names) != n_references:
        raise ValueError(
            'Expected {} reference class names, got {}.'.format(
                n_references, len(names)))
    return {
        int(class_idx): name
        for class_idx, name in zip(class_indices.tolist(), names)
    }


def reference_mapping_info(source, mapping, cost_matrix, class_indices,
                           reference_path, reference_endmembers,
                           class_names_by_index=None, max_sad=None):
    class_indices = np.asarray(class_indices, dtype=np.int64)
    class_names = [
        safe_class_name(int(idx), class_names_by_index)
        for idx in class_indices.tolist()
    ]
    return {
        'source': source,
        'method': 'spectral_sad',
        'reference_endmembers': str(reference_path),
        'reference_shape': list(np.asarray(reference_endmembers).shape),
        'max_sad': max_sad,
        'class_indices': class_indices.tolist(),
        'class_names': class_names,
        'cost_matrix_sad': np.asarray(cost_matrix).tolist(),
        'endmember_to_class': mapping_records(
            mapping, class_names_by_index=class_names_by_index),
    }


def compute_reference_spectral_mapping(model, device, reference_endmember_path,
                                       reference_class_indices=None,
                                       reference_class_names=None,
                                       max_sad=None):
    if reference_endmember_path is None:
        raise ValueError(
            '--reference-endmembers is required when --mapping-source '
            'reference.')

    base_model = model.module if hasattr(model, 'module') else model
    estimated_endmembers = base_model.get_endmembers().detach().to(device)
    expected_bands = estimated_endmembers.shape[1]
    reference_endmembers, resolved_path = load_reference_endmembers(
        reference_endmember_path, expected_bands=expected_bands)
    class_indices = load_reference_class_indices(
        reference_class_indices, reference_endmembers.shape[0])
    class_names_by_index = load_reference_class_names(
        reference_class_names, reference_endmembers.shape[0], class_indices,
        default_prefix='reference' if (
            reference_class_indices is None and reference_class_names is None)
        else None)

    mapping, cost_matrix = torchseg.model.metric \
        .odsi_db_unmixing_hungarian_mapping_from_reference_endmembers(
            estimated_endmembers,
            torch.as_tensor(reference_endmembers, device=device),
            torch.as_tensor(class_indices, device=device),
            max_sad=max_sad,
            device='cpu',
            return_cost_matrix=True)

    return (
        mapping,
        cost_matrix,
        class_indices,
        reference_endmembers,
        class_names_by_index,
        resolved_path,
    )


def learn_reference_spectral_mapping(model, device, reference_endmember_path,
                                     reference_class_indices=None,
                                     reference_class_names=None,
                                     max_sad=None):
    mapping, cost_matrix, class_indices, reference_endmembers, \
        class_names_by_index, resolved_path = compute_reference_spectral_mapping(
            model, device, reference_endmember_path,
            reference_class_indices=reference_class_indices,
            reference_class_names=reference_class_names,
            max_sad=max_sad)

    info = reference_mapping_info(
        'reference', mapping, cost_matrix, class_indices, resolved_path,
        reference_endmembers, class_names_by_index=class_names_by_index,
        max_sad=max_sad)
    return mapping, info


def compute_mask_spectral_mapping(model, device, image_path, mode,
                                  mask_path=None, max_sad=None,
                                  mapping_class_scope='all-interest'):
    resolved_image_path = resolve_image_path(image_path)
    resolved_mask_path = resolve_mask_path(
        mask_path, image_path=resolved_image_path, required=True)

    base_model = model.module if hasattr(model, 'module') else model
    estimated_endmembers = base_model.get_endmembers().detach().to(device)
    n_endmembers = estimated_endmembers.shape[0]

    im_hyper, wl, _, _ = torchseg.data_loader.read_stiff(
        resolved_image_path, silent=True, rgb_only=False)
    target = torchseg.data_loader.OdsiDbDataLoader.LoadImage \
        .read_hyper_reflectance(
            resolved_image_path, mode, im_hyper=im_hyper, wl=wl)
    labels = torchseg.data_loader.OdsiDbDataLoader.LoadImage.read_label(
        resolved_mask_path)

    target = torch.from_numpy(target.transpose((2, 0, 1))) \
        .unsqueeze(0).to(device)
    labels = torch.from_numpy(labels).unsqueeze(0).to(device)
    spectral_sums, pixel_counts, class_indices = \
        torchseg.model.metric.odsi_db_unmixing_reference_sums(
            target, labels, n_endmembers)

    reference_endmembers = spectral_sums / \
        pixel_counts.clamp_min(1.0).unsqueeze(1)
    reference_valid = reference_valid_for_mapping_scope(
        pixel_counts, mapping_class_scope)
    mapping, cost_matrix = torchseg.model.metric \
        .odsi_db_unmixing_hungarian_mapping_from_reference_endmembers(
            estimated_endmembers, reference_endmembers, class_indices,
            reference_valid=reference_valid,
            max_sad=max_sad,
            device='cpu',
            return_cost_matrix=True)

    return (
        mapping,
        cost_matrix,
        class_indices.detach().cpu().numpy().astype(np.int64),
        reference_endmembers.detach().cpu().numpy(),
        pixel_counts.detach().cpu().numpy(),
        resolved_mask_path,
    )


def learn_mask_spectral_mapping(model, device, image_path, mode,
                                mask_path=None, max_sad=None,
                                mapping_class_scope='all-interest'):
    mapping, cost_matrix, class_indices, reference_endmembers, \
        pixel_counts, resolved_mask_path = compute_mask_spectral_mapping(
            model, device, image_path, mode, mask_path=mask_path,
            max_sad=max_sad, mapping_class_scope=mapping_class_scope)

    idx2class = torchseg.data_loader.OdsiDbDataLoader.OdsiDbDataset.classnames
    info = {
        'source': 'mask',
        'method': 'spectral_sad',
        'mask_path': str(resolved_mask_path),
        'max_sad': max_sad,
        'mapping_class_scope': mapping_class_scope,
        'class_indices': class_indices.tolist(),
        'class_names': [idx2class[int(i)] for i in class_indices.tolist()],
        'reference_pixel_counts': pixel_counts.tolist(),
        'cost_matrix_sad': np.asarray(cost_matrix).tolist(),
        'endmember_to_class': mapping_records(mapping),
    }
    return mapping, info


def learn_validation_spectral_mapping(config, model, device,
                                      max_batches=None,
                                      mapping_class_scope='all-interest'):
    data_loader_config = copy.deepcopy(config['data_loader'])
    data_loader_config['args']['num_workers'] = 0
    data_loader_config['args']['data_dir'] = resolve_data_dir(
        data_loader_config['args']['data_dir'])

    data_loader = getattr(torchseg.data_loader, data_loader_config['type'])(
        **data_loader_config['args'])
    valid_loader = data_loader.split_validation()
    if valid_loader is None:
        raise RuntimeError(
            'Cannot learn spectral Hungarian mapping: config.data_loader '
            'has no validation split.')

    base_model = model.module if hasattr(model, 'module') else model
    estimated_endmembers = base_model.get_endmembers().detach().to(device)
    n_endmembers = estimated_endmembers.shape[0]

    spectral_sums = None
    pixel_counts = None
    class_indices = None
    batches_used = 0

    with torch.no_grad():
        for batch_idx, raw_data in enumerate(valid_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            target = raw_data['target_reflectance'].to(device)
            labels = raw_data['label'].to(device)
            batch_sums, batch_counts, batch_class_indices = \
                torchseg.model.metric.odsi_db_unmixing_reference_sums(
                    target, labels, n_endmembers)

            if class_indices is None:
                class_indices = batch_class_indices.detach().cpu()
                spectral_sums = torch.zeros_like(batch_sums)
                pixel_counts = torch.zeros_like(batch_counts)
            elif not torch.equal(class_indices.to(batch_class_indices.device),
                                 batch_class_indices):
                raise RuntimeError(
                    'Validation class subset changed between batches.')

            spectral_sums += batch_sums.detach()
            pixel_counts += batch_counts.detach()
            batches_used += 1

    if spectral_sums is None:
        raise RuntimeError(
            'Cannot learn spectral Hungarian mapping: no validation batches.')

    reference_endmembers = spectral_sums / \
        pixel_counts.clamp_min(1.0).unsqueeze(1)
    reference_valid = reference_valid_for_mapping_scope(
        pixel_counts, mapping_class_scope)
    mapping, cost_matrix = \
        torchseg.model.metric \
        .odsi_db_unmixing_hungarian_mapping_from_reference_endmembers(
            estimated_endmembers, reference_endmembers,
            class_indices.to(device),
            reference_valid=reference_valid,
            device='cpu',
            return_cost_matrix=True)

    class_indices_np = class_indices.numpy().astype(np.int64)
    idx2class = torchseg.data_loader.OdsiDbDataLoader.OdsiDbDataset.classnames
    info = {
        'source': 'validation',
        'method': 'spectral_sad',
        'batches_used': int(batches_used),
        'mapping_class_scope': mapping_class_scope,
        'class_indices': class_indices_np.tolist(),
        'class_names': [idx2class[int(i)] for i in class_indices_np.tolist()],
        'reference_pixel_counts': pixel_counts.detach().cpu().numpy().tolist(),
        'cost_matrix_sad': cost_matrix.tolist(),
        'endmember_to_class': mapping_records(mapping),
    }
    return mapping, info


def make_single_image_eval_loader(config, eval_split):
    data_loader_config = copy.deepcopy(config['data_loader'])
    data_loader_config['args']['num_workers'] = 0

    if eval_split == 'all':
        data_loader_config['args']['validation_split'] = 0.0
        data_loader_config['args']['shuffle'] = False
        data_loader = getattr(torchseg.data_loader, data_loader_config['type'])(
            **data_loader_config['args'])
        data_loader.training = False
        return data_loader

    data_loader = getattr(torchseg.data_loader, data_loader_config['type'])(
        **data_loader_config['args'])
    data_loader.training = False
    if eval_split == 'train':
        return data_loader

    valid_loader = data_loader.split_validation()
    if valid_loader is None:
        raise RuntimeError(
            'Cannot evaluate validation split: config.data_loader has no '
            'validation split.')
    return valid_loader


def evaluate_dataset(model, dataset_config, loss_fn, metric_fns, device,
                     max_batches=None, semantic_mapping=None,
                     disable_semantic_mapping=False):
    data_loader = getattr(torchseg.data_loader, dataset_config['type'])(
        **dataset_config['args'])
    data_loader.training = False
    return evaluate_loader(model, data_loader, loss_fn, metric_fns, device,
                           max_batches=max_batches,
                           semantic_mapping=semantic_mapping,
                           disable_semantic_mapping=disable_semantic_mapping)


def evaluate_loader(model, data_loader, loss_fn, metric_fns, device,
                    max_batches=None, semantic_mapping=None,
                    disable_semantic_mapping=False):

    total_loss = 0.0
    total_metrics = {metric.__name__: 0.0 for metric in metric_fns}
    total_samples = 0
    total_batches = 0

    with torch.no_grad():
        for batch_idx, raw_data in enumerate(tqdm.tqdm(data_loader)):
            if max_batches is not None and batch_idx >= max_batches:
                break

            data = raw_data['image'].to(device)
            target = raw_data['target_reflectance'].to(device)
            output = model(data)
            if isinstance(output, dict) and 'label' in raw_data:
                output['label'] = raw_data['label'].to(device)
            if isinstance(output, dict) and semantic_mapping is not None:
                output['semantic_mapping'] = semantic_mapping.to(device)
            if isinstance(output, dict) and disable_semantic_mapping:
                output['disable_semantic_mapping'] = True

            batch_size = data.shape[0]
            loss = loss_fn(output, target)
            total_loss += loss.item() * batch_size
            for metric in metric_fns:
                total_metrics[metric.__name__] += \
                    metric(output, target) * batch_size
            total_samples += batch_size
            total_batches += 1

    if total_samples == 0:
        raise RuntimeError('No samples were evaluated.')

    result = {
        'loss': total_loss / total_samples,
        'samples': total_samples,
        'batches': total_batches,
    }
    result.update({
        name: value / total_samples
        for name, value in total_metrics.items()
    })
    return result


def main():
    args = parse_args()
    if args.device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    config = read_config(args)
    active_reference_class_names = \
        torchseg.model.configure_odsi_db_unmixing_from_config(config)
    device, _ = torchseg.utils.setup_gpu_devices(config['n_gpu'])
    model, checkpoint = load_model(config, args.resume, device)

    loss_fn = torchseg.model.loss.get_loss_function(config['loss'])
    metric_fns = [getattr(torchseg.model.metric, metric)
                  for metric in config['metrics']]

    results = {
        'checkpoint': args.resume,
        'checkpoint_epoch': checkpoint.get('epoch'),
        'active_reference_class_names': active_reference_class_names,
        'mapping_class_scope': args.mapping_class_scope,
        'semantic_mapping': None,
        'datasets': [],
    }

    semantic_mapping = None
    single_image = is_single_image_config(config)
    if single_image and args.mapping_source in ('validation', 'batch'):
        raise RuntimeError(
            'Single-image unmixing configs do not contain labels for semantic '
            'mapping from validation/batch labels. Use --mapping-source none '
            'or --mapping-source mask/reference.')

    effective_mapping_source = args.mapping_source
    if args.mapping_source == 'auto':
        if single_image:
            mask_path = resolve_mask_path(
                args.mask_path,
                image_path=config['data_loader']['args']['image_path'],
                required=False)
            effective_mapping_source = 'mask' if mask_path is not None \
                else 'none'
        else:
            effective_mapping_source = 'validation'

    if effective_mapping_source == 'validation':
        semantic_mapping, mapping_info = learn_validation_spectral_mapping(
            config, model, device, max_batches=args.mapping_max_batches,
            mapping_class_scope=args.mapping_class_scope)
        results['semantic_mapping'] = mapping_info
    elif effective_mapping_source == 'mask':
        data_args = config['data_loader']['args']
        semantic_mapping, mapping_info = learn_mask_spectral_mapping(
            model, device, data_args['image_path'], data_args['mode'],
            mask_path=args.mask_path, max_sad=args.reference_max_sad,
            mapping_class_scope=args.mapping_class_scope)
        if args.mapping_source == 'auto':
            mapping_info['source'] = 'auto_mask'
        results['semantic_mapping'] = mapping_info
    elif effective_mapping_source == 'reference':
        semantic_mapping, mapping_info = learn_reference_spectral_mapping(
            model, device, args.reference_endmembers,
            reference_class_indices=args.reference_class_indices,
            reference_class_names=args.reference_class_names,
            max_sad=args.reference_max_sad)
        results['semantic_mapping'] = mapping_info
    elif effective_mapping_source == 'batch':
        results['semantic_mapping'] = {
            'source': 'batch',
            'method': 'spectral_sad',
            'note': (
                'No frozen mapping was learned; metrics compute a per-batch '
                'spectral Hungarian diagnostic when target labels are present.')
        }
    else:
        results['semantic_mapping'] = {
            'source': 'none',
            'method': None,
            'note': 'No semantic mapping was available or requested.'
        }

    if single_image:
        data_loader = make_single_image_eval_loader(config, args.eval_split)
        result = evaluate_loader(model, data_loader, loss_fn, metric_fns,
                                 device, max_batches=args.max_batches,
                                 semantic_mapping=semantic_mapping,
                                 disable_semantic_mapping=
                                 effective_mapping_source == 'none')
        result['type'] = config['data_loader']['type']
        result['image_path'] = config['data_loader']['args']['image_path']
        result['eval_split'] = args.eval_split
        results['datasets'].append(result)
    else:
        for dataset_config in config['testing']['datasets']:
            result = evaluate_dataset(model, dataset_config, loss_fn,
                                      metric_fns, device,
                                      max_batches=args.max_batches,
                                      semantic_mapping=semantic_mapping,
                                      disable_semantic_mapping=
                                      effective_mapping_source == 'none')
            result['type'] = dataset_config['type']
            result['data_dir'] = dataset_config['args']['data_dir']
            results['datasets'].append(result)

    print(json.dumps(results, indent=4))
    if args.output is not None:
        torchseg.utils.write_json(results, args.output)


if __name__ == '__main__':
    main()
