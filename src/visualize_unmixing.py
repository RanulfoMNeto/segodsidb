"""
@brief Export practical visualisations for trained CNNAEU unmixing checkpoints.
"""

import argparse
import copy
import json
import os
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import torchseg.data_loader
import torchseg.model.metric
import torchseg.test_unmixing
import torchseg.utils


DEFAULT_RELEVANT_CLASSES = \
    torchseg.model.metric.ODSI_DB_UNMIXING_RELEVANT_CLASS_NAMES.copy()


def parse_args():
    args = argparse.ArgumentParser(description='Visualize unmixing outputs.')
    args.add_argument('-r', '--resume', required=True, type=str,
                      help='checkpoint path, usually model_best.pth')
    args.add_argument('-o', '--output-dir', required=True, type=str,
                      help='directory where PNG/NPY/CSV files will be saved')
    args.add_argument('-c', '--conf', default=None, type=str,
                      help='optional config override; defaults to checkpoint config.json')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable')
    args.add_argument('--image-path', default=None, type=str,
                      help='override single-image HSI path')
    args.add_argument('--test-data-dir', default=None, type=str,
                      help='legacy override for all testing dataset directories')
    args.add_argument('--dataset-index', default=0, type=int,
                      help='index inside config.testing.datasets')
    args.add_argument('--num-samples', default=3, type=int,
                      help='number of patches to export')
    args.add_argument('--seed', default=0, type=int,
                      help='seed for reproducible random crops')
    args.add_argument('--class-names', default=None,
                      nargs='+',
                      help='class names used to select interpretable patches; '
                           'defaults to unmixing.reference_class_names')
    args.add_argument('--include-all-classes', action='store_true',
                      help='ignore --class-names and use any annotated class')
    args.add_argument('--min-label-fraction', default=0.5, type=float,
                      help='minimum patch fraction covered by the selected class')
    args.add_argument('--max-tries-per-image', default=64, type=int,
                      help='random crop attempts per image and target class')
    args.add_argument('--max-samples-per-image', default=1, type=int,
                      help='limits repeated patches from the same source image')
    args.add_argument('--full-images', default=0, type=int,
                      help='number of whole images to export with tiled inference')
    args.add_argument('--tile-size', default=256, type=int,
                      help='output tile size for whole-image inference')
    args.add_argument('--full-top-abundances', default=3, type=int,
                      help='deprecated; individual abundance PNGs are no longer exported')
    args.add_argument('--mapping-source', default='auto',
                      choices=['auto', 'validation', 'mask', 'reference',
                               'none'],
                      help='source used to learn endmember-to-class mapping')
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
    return args.parse_args()


def spectral_angle_map(reconstruction, target, eps=1e-8):
    target = target.to(device=reconstruction.device,
                       dtype=reconstruction.dtype)
    reconstruction = reconstruction.unsqueeze(0)
    target = target.unsqueeze(0)
    rec_norm = F.normalize(reconstruction, p=2, dim=1, eps=eps)
    target_norm = F.normalize(target, p=2, dim=1, eps=eps)
    cosine = torch.sum(rec_norm * target_norm, dim=1)
    cosine = torch.clamp(cosine, min=-1.0 + eps, max=1.0)
    cosine = torch.where(cosine > 1.0 - 1e-6,
                         torch.ones_like(cosine), cosine)
    return torch.acos(cosine)[0]


def save_endmembers(model, wavelengths, output_dir):
    endmembers = model.get_endmembers().detach().cpu().numpy()
    np.save(output_dir / 'endmembers.npy', endmembers)

    header = ['wavelength_nm']
    header += ['endmember_{}'.format(i) for i in range(endmembers.shape[0])]
    table = np.column_stack([wavelengths, endmembers.T])
    np.savetxt(output_dir / 'endmembers.csv', table, delimiter=',',
               header=','.join(header), comments='')

    fig, ax = plt.subplots(figsize=(9, 5))
    for idx in range(endmembers.shape[0]):
        ax.plot(wavelengths, endmembers[idx], linewidth=1.5,
                label='E{}'.format(idx))
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Decoder endmember value')
    ax.set_title('Learned endmember spectra')
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / 'endmembers.png', dpi=200)
    plt.close(fig)


def class_name(class_idx, class_names_by_index=None):
    return torchseg.test_unmixing.safe_class_name(
        class_idx, class_names_by_index=class_names_by_index)


def abundance_grid_titles(mapping, n_endmembers, class_names_by_index=None):
    if mapping is None:
        return ['E{}'.format(idx) for idx in range(n_endmembers)]

    mapping = np.asarray(mapping, dtype=np.int64)
    titles = []
    for endmember_idx in range(n_endmembers):
        class_idx = int(mapping[endmember_idx]) \
            if endmember_idx < len(mapping) else -1
        if class_idx >= 0:
            titles.append('E{} ->\n{}'.format(
                endmember_idx, class_name(class_idx, class_names_by_index)))
        else:
            titles.append('E{} ->\nunmapped'.format(endmember_idx))
    return titles


def save_abundance_grid(abundances, path, titles=None,
                        figure_title='Abundance maps'):
    n_endmembers = abundances.shape[0]
    ncols = min(5, n_endmembers)
    nrows = int(np.ceil(n_endmembers / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(2.8 * ncols, 2.55 * nrows),
                             squeeze=False)
    last_im = None
    for idx in range(nrows * ncols):
        ax = axes[idx // ncols][idx % ncols]
        ax.axis('off')
        if idx >= n_endmembers:
            continue
        last_im = ax.imshow(abundances[idx], vmin=0.0, vmax=1.0,
                            cmap='viridis')
        title = 'E{}'.format(idx) if titles is None else titles[idx]
        ax.set_title(title, fontsize=7)
    fig.suptitle(figure_title, fontsize=11)
    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.025,
                     pad=0.01)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_map(image, path, title, cmap='viridis', vmin=None, vmax=None,
             colorbar=True):
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def class_palette(class_idx):
    palettes = ['tab20', 'tab20b', 'tab20c']
    palette = plt.get_cmap(palettes[class_idx % len(palettes)])
    color_idx = (class_idx // len(palettes)) % palette.N
    return palette(color_idx)


def save_class_index_map(class_index_map, path, title, negative_labels=None,
                         class_names_by_index=None):
    if negative_labels is None:
        negative_labels = {-1: 'unmapped'}
    class_index_map = np.asarray(class_index_map, dtype=np.int16)
    present_values = sorted(np.unique(class_index_map).astype(int).tolist())
    if len(present_values) == 0:
        present_values = [-1]

    value_to_display = {
        class_idx: display_idx
        for display_idx, class_idx in enumerate(present_values)
    }
    display = np.zeros(class_index_map.shape, dtype=np.int16)
    for class_idx, display_idx in value_to_display.items():
        display[class_index_map == class_idx] = display_idx

    colors = []
    tick_labels = []
    for class_idx in present_values:
        if class_idx < 0:
            colors.append({
                -2: (0.88, 0.88, 0.88, 1.0),
                -1: (0.45, 0.45, 0.45, 1.0),
            }.get(class_idx, (0.75, 0.75, 0.75, 1.0)))
            tick_labels.append(negative_labels.get(class_idx, str(class_idx)))
        else:
            colors.append(class_palette(class_idx))
            tick_labels.append('{} {}'.format(
                class_idx, class_name(class_idx, class_names_by_index)))

    cmap = matplotlib.colors.ListedColormap(colors)
    norm = matplotlib.colors.BoundaryNorm(
        np.arange(-0.5, len(present_values) + 0.5), cmap.N)

    fig_width = 5.8 if len(tick_labels) <= 12 else 6.8
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    im = ax.imshow(display, cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_title(title)
    ax.axis('off')
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(len(present_values)),
                        fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=7 if len(tick_labels) <= 12 else 5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def filter_class_index_map(class_index_map, selected_class_indices,
                           other_value=-2):
    selected_class_indices = set(int(idx) for idx in selected_class_indices)
    class_index_map = np.asarray(class_index_map, dtype=np.int16)
    selected_map = class_index_map.copy()
    known_class = selected_map >= 0
    selected_class = np.isin(selected_map, list(selected_class_indices))
    selected_map[np.logical_and(known_class, ~selected_class)] = other_value
    return selected_map.astype(np.int16)


def class_index_summary(class_index_map, negative_labels=None,
                        class_names_by_index=None):
    if negative_labels is None:
        negative_labels = {-2: 'other', -1: 'unmapped'}
    summary = {}
    values, counts = np.unique(class_index_map, return_counts=True)
    for value, count in zip(values.tolist(), counts.tolist()):
        value = int(value)
        if value < 0:
            name = negative_labels.get(value, str(value))
        else:
            name = class_name(value, class_names_by_index)
        summary[name] = int(count)
    return summary


def apply_mapping_to_dominant(dominant, mapping):
    mapped = np.full(dominant.shape, -1, dtype=np.int16)
    if mapping is None:
        return mapped

    mapping = np.asarray(mapping, dtype=np.int64)
    for endmember_idx, class_idx in enumerate(mapping.tolist()):
        if class_idx >= 0:
            mapped[dominant == endmember_idx] = class_idx
    return mapped


def mapping_records(mapping, class_names_by_index=None):
    records = []
    for endmember_idx, class_idx in enumerate(np.asarray(mapping).tolist()):
        class_idx = int(class_idx)
        records.append({
            'endmember': int(endmember_idx),
            'class_index': class_idx,
            'class_name': class_name(class_idx, class_names_by_index)
                if class_idx >= 0 else None,
        })
    return records


def save_mapping(output_dir, mapping, cost_matrix, class_indices, source,
                 batches_used, reference_endmembers=None,
                 reference_counts=None, class_names_by_index=None,
                 reference_path=None, max_sad=None,
                 mapping_class_scope=None):
    mapping = np.asarray(mapping, dtype=np.int64)
    cost_matrix = np.asarray(cost_matrix, dtype=np.float64)
    class_indices = np.asarray(class_indices, dtype=np.int64)

    np.save(output_dir / 'hungarian_mapping.npy', mapping)
    np.save(output_dir / 'hungarian_cost_matrix_sad.npy', cost_matrix)
    files = {
        'mapping_npy': str(output_dir / 'hungarian_mapping.npy'),
        'cost_matrix_sad_npy':
            str(output_dir / 'hungarian_cost_matrix_sad.npy'),
    }
    if reference_endmembers is not None:
        reference_endmembers = np.asarray(reference_endmembers,
                                          dtype=np.float32)
        np.save(output_dir / 'reference_endmembers.npy',
                reference_endmembers)
        files['reference_endmembers_npy'] = \
            str(output_dir / 'reference_endmembers.npy')
    if reference_counts is not None:
        reference_counts = np.asarray(reference_counts, dtype=np.float64)
        np.save(output_dir / 'reference_pixel_counts.npy',
                reference_counts)
        files['reference_pixel_counts_npy'] = \
            str(output_dir / 'reference_pixel_counts.npy')

    mapping_info = {
        'source': source,
        'method': 'spectral_sad',
        'batches_used': int(batches_used),
        'reference_endmembers': None if reference_path is None
            else str(reference_path),
        'max_sad': max_sad,
        'mapping_class_scope': mapping_class_scope,
        'class_indices': class_indices.tolist(),
        'class_names': [
            class_name(int(i), class_names_by_index)
            for i in class_indices.tolist()
        ],
        'reference_pixel_counts': None if reference_counts is None
            else reference_counts.tolist(),
        'cost_matrix_sad': cost_matrix.tolist(),
        'endmember_to_class': mapping_records(
            mapping, class_names_by_index=class_names_by_index),
        'files': files,
    }
    with (output_dir / 'hungarian_mapping.json').open('w') as f:
        json.dump(mapping_info, f, indent=4)
    return mapping_info


def selected_mapping_coverage(selected_class_indices, mapping_class_indices,
                              mapping):
    if selected_class_indices is None:
        return None

    mapping_class_indices = set(
        int(idx) for idx in np.asarray(mapping_class_indices).tolist())
    mapping = np.asarray(mapping, dtype=np.int64)

    records = []
    for class_idx in selected_class_indices:
        class_idx = int(class_idx)
        endmembers = np.where(mapping == class_idx)[0].astype(int).tolist()
        records.append({
            'class_index': class_idx,
            'class_name': class_name(class_idx),
            'in_hungarian_competition': class_idx in mapping_class_indices,
            'assigned_endmembers': endmembers,
        })
    return records


def mapping_coverage_class_indices(mapping_class_indices,
                                   reference_counts=None):
    mapping_class_indices = np.asarray(mapping_class_indices, dtype=np.int64)
    if reference_counts is None:
        return mapping_class_indices

    reference_counts = np.asarray(reference_counts, dtype=np.float64)
    if reference_counts.shape[0] != mapping_class_indices.shape[0]:
        return mapping_class_indices
    return mapping_class_indices[reference_counts > 0]


def learn_validation_mapping(config, model, device, max_batches=None,
                             mapping_class_scope='all-interest'):
    data_loader_config = copy.deepcopy(config['data_loader'])
    data_loader_config['args']['num_workers'] = 0
    data_loader_config['args']['data_dir'] = \
        torchseg.test_unmixing.resolve_data_dir(
            data_loader_config['args']['data_dir'])

    data_loader = getattr(torchseg.data_loader, data_loader_config['type'])(
        **data_loader_config['args'])
    valid_loader = data_loader.split_validation()
    if valid_loader is None:
        raise RuntimeError(
            'Cannot learn spectral Hungarian mapping: config.data_loader has no '
            'validation split.')

    base_model = model.module if hasattr(model, 'module') else model
    estimated_endmembers = base_model.get_endmembers().detach().to(device)
    n_endmembers = estimated_endmembers.shape[0]

    spectral_sums = None
    pixel_counts = None
    class_indices_np = None
    batches_used = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, raw_data in enumerate(valid_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            target = raw_data['target_reflectance'].to(device)
            labels = raw_data['label'].to(device)
            batch_sums, batch_counts, class_indices = \
                torchseg.model.metric.odsi_db_unmixing_reference_sums(
                    target, labels, n_endmembers)
            class_indices_batch = class_indices.detach().cpu().numpy()

            if spectral_sums is None:
                class_indices_np = class_indices_batch
                spectral_sums = torch.zeros_like(batch_sums)
                pixel_counts = torch.zeros_like(batch_counts)
            elif not np.array_equal(class_indices_np, class_indices_batch):
                raise RuntimeError(
                    'Validation class subset changed between batches.')

            spectral_sums += batch_sums.detach()
            pixel_counts += batch_counts.detach()
            batches_used += 1

    if spectral_sums is None:
        raise RuntimeError(
            'Cannot learn spectral Hungarian mapping: no labelled pixels found.')

    reference_endmembers = spectral_sums / \
        pixel_counts.clamp_min(1.0).unsqueeze(1)
    reference_valid = torchseg.test_unmixing \
        .reference_valid_for_mapping_scope(pixel_counts, mapping_class_scope)
    mapping, cost_matrix = torchseg.model.metric \
        .odsi_db_unmixing_hungarian_mapping_from_reference_endmembers(
            estimated_endmembers, reference_endmembers,
            torch.as_tensor(class_indices_np, device=device),
            reference_valid=reference_valid,
            device='cpu',
            return_cost_matrix=True)
    return (
        mapping.cpu().numpy(),
        cost_matrix,
        class_indices_np,
        batches_used,
        reference_endmembers.detach().cpu().numpy(),
        pixel_counts.detach().cpu().numpy(),
    )


def save_target_rgb(target_chw, wavelengths, path):
    target_hwc = np.transpose(target_chw, (1, 2, 0))
    target_hwc = np.clip(target_hwc, 0.0, 1.0)
    rgb = torchseg.data_loader.OdsiDbDataLoader.LoadImage.hyper2rgb(
        target_hwc, wavelengths)
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.imshow(rgb, interpolation='nearest')
    ax.set_title('Target reflectance RGB')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def label_summary(label_chw):
    idx2class = torchseg.data_loader.OdsiDbDataLoader.OdsiDbDataset.classnames
    class_pixels = label_chw.reshape(label_chw.shape[0], -1).sum(axis=1)
    summary = {}
    for idx, count in enumerate(class_pixels.tolist()):
        if count > 0:
            summary[idx2class[idx]] = int(count)
    return summary


def label_map(label_chw):
    valid = label_chw.sum(axis=0) == 1
    labels = np.argmax(label_chw, axis=0).astype(np.float32)
    labels[~valid] = -1
    return labels


def class_indices(class_names, include_all_classes=False):
    idx2class = torchseg.data_loader.OdsiDbDataLoader.OdsiDbDataset.classnames
    if include_all_classes:
        return list(idx2class.keys())

    class2idx = {v: k for k, v in idx2class.items()}
    missing = [name for name in class_names if name not in class2idx]
    if missing:
        raise ValueError('Unknown ODSI-DB class names: {}'.format(missing))
    return [class2idx[name] for name in class_names]


def dataset_class_presence(dataset, selected_class_indices):
    if selected_class_indices is None:
        return None

    idx2class = torchseg.data_loader.OdsiDbDataLoader.OdsiDbDataset.classnames
    selected_class_indices = [int(idx) for idx in selected_class_indices]
    pixels = {idx: 0 for idx in selected_class_indices}
    images = {idx: 0 for idx in selected_class_indices}

    for item in dataset.data:
        label = torchseg.data_loader.OdsiDbDataLoader.LoadImage.read_label(
            item['label'])
        counts = label.reshape(label.shape[0], -1).sum(axis=1)
        for class_idx in selected_class_indices:
            count = int(counts[class_idx])
            pixels[class_idx] += count
            if count > 0:
                images[class_idx] += 1

    return [{
        'class_index': class_idx,
        'class_name': idx2class[class_idx],
        'images': int(images[class_idx]),
        'pixels': int(pixels[class_idx]),
    } for class_idx in selected_class_indices]


def try_labeled_crop(dataset, image_idx, target_class_idx, rng,
                     min_label_fraction, max_tries):
    item = dataset.data[image_idx]
    im_hyper, wl, label = dataset._load_image(image_idx)
    patch_size = dataset.patch_size
    h = im_hyper.shape[0]
    w = im_hyper.shape[1]
    min_pixels = int(np.ceil(min_label_fraction * patch_size * patch_size))

    target_mask = label[target_class_idx] > 0
    if target_mask.sum() == 0:
        return None
    ys, xs = np.where(target_mask)

    for _ in range(max_tries):
        center_idx = rng.randint(0, len(ys))
        row = int(ys[center_idx]) - rng.randint(0, patch_size)
        col = int(xs[center_idx]) - rng.randint(0, patch_size)
        row = int(np.clip(row, 0, h - patch_size))
        col = int(np.clip(col, 0, w - patch_size))

        patch_label = label[:, row:row + patch_size,
                            col:col + patch_size].copy()
        if patch_label[target_class_idx].sum() < min_pixels:
            continue

        patch_hyper = im_hyper[row:row + patch_size,
                               col:col + patch_size, :].copy()
        image, target = \
            torchseg.data_loader.OdsiDbDataLoader.LoadImage.read_hyper_image_pair(
                item['image'], dataset.mode, im_hyper=patch_hyper, wl=wl)
        return {
            'image': torch.from_numpy(image),
            'target_reflectance': torch.from_numpy(target),
            'label': torch.from_numpy(patch_label),
            'path': item['image'],
            'crop_row': row,
            'crop_col': col,
            'patch_size': patch_size,
            'target_class_idx': target_class_idx,
        }

    return None


def collect_samples(dataset, wanted_class_indices, num_samples, rng,
                    min_label_fraction, max_tries_per_image,
                    max_samples_per_image):
    idx2class = torchseg.data_loader.OdsiDbDataLoader.OdsiDbDataset.classnames
    n_images = len(dataset.data)
    per_image_count = {idx: 0 for idx in range(n_images)}
    samples = []

    class_order = []
    while len(class_order) < num_samples * max(1, len(wanted_class_indices)):
        class_order.extend(wanted_class_indices)

    for target_class_idx in class_order:
        if len(samples) >= num_samples:
            break

        image_order = rng.permutation(n_images).tolist()
        for image_idx in image_order:
            if per_image_count[image_idx] >= max_samples_per_image:
                continue
            sample = try_labeled_crop(dataset, image_idx, target_class_idx,
                                      rng, min_label_fraction,
                                      max_tries_per_image)
            if sample is None:
                continue
            sample['target_class_name'] = idx2class[target_class_idx]
            samples.append(sample)
            per_image_count[image_idx] += 1
            break

    if len(samples) < num_samples:
        raise RuntimeError(
            'Only found {} usable patches out of {} requested. Try lowering '
            '--min-label-fraction or increasing --max-tries-per-image.'.format(
                len(samples), num_samples))

    return samples


def select_full_images(dataset, wanted_class_indices, num_images, rng):
    idx2class = torchseg.data_loader.OdsiDbDataLoader.OdsiDbDataset.classnames
    if num_images <= 0:
        return []

    class_counts = []
    for item in dataset.data:
        label = torchseg.data_loader.OdsiDbDataLoader.LoadImage.read_label(
            item['label'])
        counts = label.reshape(label.shape[0], -1).sum(axis=1)
        class_counts.append(counts)

    selected = []
    used_images = set()
    class_order = []
    while len(class_order) < num_images * max(1, len(wanted_class_indices)):
        class_order.extend(wanted_class_indices)

    for target_class_idx in class_order:
        if len(selected) >= num_images:
            break

        candidates = [
            image_idx for image_idx in range(len(dataset.data))
            if image_idx not in used_images
            and class_counts[image_idx][target_class_idx] > 0
        ]
        if not candidates:
            continue

        # Pick a random image among those with substantial class presence.
        counts = np.array([class_counts[idx][target_class_idx]
                           for idx in candidates], dtype=np.float64)
        threshold = np.percentile(counts, 75)
        strong_candidates = [
            idx for idx in candidates
            if class_counts[idx][target_class_idx] >= threshold
        ]
        image_idx = strong_candidates[
            rng.randint(0, len(strong_candidates))]
        used_images.add(image_idx)
        selected.append({
            'image_idx': image_idx,
            'target_class_idx': target_class_idx,
            'target_class_name': idx2class[target_class_idx],
            'target_class_pixels': int(class_counts[image_idx][target_class_idx]),
        })

    if len(selected) < num_images:
        raise RuntimeError(
            'Only found {} whole images out of {} requested.'.format(
                len(selected), num_images))

    return selected


def model_context_margin(model):
    encoder_margin = model.encoder_conv1.kernel_size[0] // 2
    decoder_margin = model.decoder.kernel_size[0] // 2
    return encoder_margin + decoder_margin


def save_rgb_exact(rgb, path):
    plt.imsave(path, rgb)


def export_full_image(full_idx, image_info, dataset, model, device,
                      wavelengths, output_dir, tile_size, _top_abundances,
                      mapping=None, selected_class_indices=None,
                      class_names_by_index=None):
    item = dataset.data[image_info['image_idx']]
    im_hyper, wl, _, _ = torchseg.data_loader.read_stiff(
        item['image'], silent=True, rgb_only=False)
    label = torchseg.data_loader.OdsiDbDataLoader.LoadImage.read_label(
        item['label'])

    h, w, _ = im_hyper.shape
    n_endmembers = model.decoder.in_channels
    abundances = np.empty((n_endmembers, h, w), dtype=np.float32)
    sad_error = np.empty((h, w), dtype=np.float32)
    margin = model_context_margin(model)

    for row in range(0, h, tile_size):
        row_end = min(row + tile_size, h)
        for col in range(0, w, tile_size):
            col_end = min(col + tile_size, w)

            in_row = max(0, row - margin)
            in_col = max(0, col - margin)
            in_row_end = min(h, row_end + margin)
            in_col_end = min(w, col_end + margin)

            tile_hyper = im_hyper[in_row:in_row_end,
                                  in_col:in_col_end, :].copy()
            image, target = \
                torchseg.data_loader.OdsiDbDataLoader.LoadImage.read_hyper_image_pair(
                    item['image'], dataset.mode, im_hyper=tile_hyper, wl=wl)
            image = torch.from_numpy(image).unsqueeze(0).to(device)
            target = torch.from_numpy(target).unsqueeze(0).to(device)
            output = model(image)

            inner_row = row - in_row
            inner_col = col - in_col
            inner_row_end = inner_row + (row_end - row)
            inner_col_end = inner_col + (col_end - col)

            abundance_tile = output['abundances'][0, :,
                inner_row:inner_row_end, inner_col:inner_col_end]
            reconstruction_tile = output['reconstruction'][0, :,
                inner_row:inner_row_end, inner_col:inner_col_end]
            target_tile = target[0, :, inner_row:inner_row_end,
                                 inner_col:inner_col_end]
            sad_tile = spectral_angle_map(reconstruction_tile, target_tile)

            abundances[:, row:row_end, col:col_end] = \
                abundance_tile.detach().cpu().numpy()
            sad_error[row:row_end, col:col_end] = \
                sad_tile.detach().cpu().numpy()

    dominant = abundances.argmax(axis=0).astype(np.int16)
    mapped_class = apply_mapping_to_dominant(dominant, mapping)
    label_idx = label_map(label).astype(np.int16)
    mapped_class_selected = None
    label_idx_selected = None
    if selected_class_indices is not None:
        mapped_class_selected = filter_class_index_map(
            mapped_class, selected_class_indices)
        label_idx_selected = filter_class_index_map(
            label_idx, selected_class_indices)
    target_rgb = torchseg.data_loader.OdsiDbDataLoader.LoadImage.hyper2rgb(
        im_hyper, wl)

    prefix = output_dir / 'full_{:03d}'.format(full_idx)
    np.save(str(prefix) + '_abundances.npy', abundances)
    np.save(str(prefix) + '_dominant_endmember.npy', dominant)
    np.save(str(prefix) + '_mapped_class.npy', mapped_class)
    np.save(str(prefix) + '_sad_error.npy', sad_error)
    np.save(str(prefix) + '_label_index.npy', label_idx)
    if selected_class_indices is not None:
        np.save(str(prefix) + '_mapped_class_selected.npy',
                mapped_class_selected)
        np.save(str(prefix) + '_label_index_selected.npy',
                label_idx_selected)

    save_rgb_exact(target_rgb, str(prefix) + '_target_rgb.png')
    save_abundance_grid(
        abundances, str(prefix) + '_abundances.png',
        titles=abundance_grid_titles(
            mapping, n_endmembers, class_names_by_index),
        figure_title='Full-image abundance maps with spectral pairing')
    if mapping is not None:
        save_class_index_map(
            mapped_class, str(prefix) + '_dominant_endmember.png',
            'Dominant mapped class from spectral Hungarian',
            class_names_by_index=class_names_by_index)
        if selected_class_indices is not None:
            save_class_index_map(
                mapped_class_selected,
                str(prefix) + '_mapped_class_selected.png',
                'Mapped class selected from validation spectral Hungarian',
                negative_labels={-2: 'other mapped class', -1: 'unmapped'},
                class_names_by_index=class_names_by_index)
    else:
        save_map(dominant, str(prefix) + '_dominant_endmember.png',
                 'Dominant endmember', cmap='tab20', vmin=-0.5,
                 vmax=n_endmembers - 0.5)
    save_map(sad_error, str(prefix) + '_sad_error.png',
             'SAD reconstruction error (rad)', cmap='magma', vmin=0.0)
    save_class_index_map(label_idx, str(prefix) + '_label_map.png',
                         'ODSI annotation class index',
                         negative_labels={-1: 'unlabelled'})
    if selected_class_indices is not None:
        save_class_index_map(
            label_idx_selected, str(prefix) + '_label_map_selected.png',
            'ODSI annotation selected class index',
            negative_labels={-2: 'other label', -1: 'unlabelled'})
    files = {
        'abundances_npy': str(prefix) + '_abundances.npy',
        'abundances_png': str(prefix) + '_abundances.png',
        'dominant_endmember_npy': str(prefix) + '_dominant_endmember.npy',
        'dominant_endmember_png': str(prefix) + '_dominant_endmember.png',
        'mapped_class_npy': str(prefix) + '_mapped_class.npy',
        'sad_error_npy': str(prefix) + '_sad_error.npy',
        'sad_error_png': str(prefix) + '_sad_error.png',
        'label_index_npy': str(prefix) + '_label_index.npy',
        'label_map_png': str(prefix) + '_label_map.png',
        'target_rgb_png': str(prefix) + '_target_rgb.png',
    }
    if mapping is not None:
        files['mapped_class_png'] = str(prefix) + '_dominant_endmember.png'
    if selected_class_indices is not None:
        files['mapped_class_selected_npy'] = \
            str(prefix) + '_mapped_class_selected.npy'
        files['label_index_selected_npy'] = \
            str(prefix) + '_label_index_selected.npy'
        files['label_map_selected_png'] = \
            str(prefix) + '_label_map_selected.png'
        if mapping is not None:
            files['mapped_class_selected_png'] = \
                str(prefix) + '_mapped_class_selected.png'

    full_info = {
        'full_image': full_idx,
        'target_class': image_info['target_class_name'],
        'target_class_pixels': image_info['target_class_pixels'],
        'path': item['image'],
        'height': int(h),
        'width': int(w),
        'tile_size': int(tile_size),
        'context_margin': int(margin),
        'label_pixels': label_summary(label),
        'files': files,
    }
    if selected_class_indices is not None:
        full_info['selected_mapped_class_pixels'] = class_index_summary(
            mapped_class_selected,
            negative_labels={-2: 'other mapped class', -1: 'unmapped'},
            class_names_by_index=class_names_by_index)
        full_info['selected_label_pixels'] = class_index_summary(
            label_idx_selected,
            negative_labels={-2: 'other label', -1: 'unlabelled'})
    return full_info


def export_sample(sample_idx, sample, output, wavelengths, output_dir,
                  mapping=None, selected_class_indices=None,
                  class_names_by_index=None):
    abundances = output['abundances'][0].detach().cpu().numpy()
    reconstruction = output['reconstruction'][0]
    target_item = sample['target_reflectance']
    sad = spectral_angle_map(reconstruction, target_item)
    label_item = sample['label'].detach().cpu().numpy()

    prefix = output_dir / 'sample_{:03d}'.format(sample_idx)
    np.save(str(prefix) + '_abundances.npy', abundances)
    np.save(str(prefix) + '_sad_error.npy', sad.detach().cpu().numpy())
    np.save(str(prefix) + '_label.npy', label_item)

    save_abundance_grid(
        abundances, str(prefix) + '_abundances.png',
        titles=abundance_grid_titles(
            mapping, abundances.shape[0], class_names_by_index),
        figure_title='Patch abundance maps with spectral pairing')
    dominant = np.argmax(abundances, axis=0)
    mapped_class = apply_mapping_to_dominant(dominant, mapping)
    label_idx = label_map(label_item)
    mapped_class_selected = None
    label_idx_selected = None
    if selected_class_indices is not None:
        mapped_class_selected = filter_class_index_map(
            mapped_class, selected_class_indices)
        label_idx_selected = filter_class_index_map(
            label_idx, selected_class_indices)
    np.save(str(prefix) + '_dominant_endmember.npy', dominant.astype(np.int16))
    np.save(str(prefix) + '_mapped_class.npy', mapped_class)
    if selected_class_indices is not None:
        np.save(str(prefix) + '_mapped_class_selected.npy',
                mapped_class_selected)
        np.save(str(prefix) + '_label_index_selected.npy',
                label_idx_selected)
    if mapping is not None:
        save_class_index_map(
            mapped_class, str(prefix) + '_dominant_endmember.png',
            'Dominant mapped class from spectral Hungarian',
            class_names_by_index=class_names_by_index)
        if selected_class_indices is not None:
            save_class_index_map(
                mapped_class_selected,
                str(prefix) + '_mapped_class_selected.png',
                'Mapped class selected from validation spectral Hungarian',
                negative_labels={-2: 'other mapped class', -1: 'unmapped'},
                class_names_by_index=class_names_by_index)
    else:
        save_map(dominant, str(prefix) + '_dominant_endmember.png',
                 'Dominant endmember', cmap='tab20', vmin=-0.5,
                 vmax=abundances.shape[0] - 0.5)
    save_map(sad.detach().cpu().numpy(), str(prefix) + '_sad_error.png',
             'SAD reconstruction error (rad)', cmap='magma', vmin=0.0)
    save_class_index_map(label_idx, str(prefix) + '_label_map.png',
                         'ODSI annotation class index',
                         negative_labels={-1: 'unlabelled'})
    if selected_class_indices is not None:
        save_class_index_map(
            label_idx_selected, str(prefix) + '_label_map_selected.png',
            'ODSI annotation selected class index',
            negative_labels={-2: 'other label', -1: 'unlabelled'})
    save_target_rgb(target_item.detach().cpu().numpy(), wavelengths,
                    str(prefix) + '_target_rgb.png')

    path = sample['path']
    if isinstance(path, pathlib.Path):
        path = str(path)
    label_info = label_summary(label_item)

    result = {
        'sample': sample_idx,
        'target_class': sample['target_class_name'],
        'path': path,
        'crop_row': int(sample['crop_row']),
        'crop_col': int(sample['crop_col']),
        'patch_size': int(sample['patch_size']),
        'label_pixels': label_info,
        'files': {
            'abundances_npy': str(prefix) + '_abundances.npy',
            'abundances_png': str(prefix) + '_abundances.png',
            'dominant_endmember_npy': str(prefix) + '_dominant_endmember.npy',
            'dominant_endmember_png': str(prefix) + '_dominant_endmember.png',
            'mapped_class_npy': str(prefix) + '_mapped_class.npy',
            'sad_error_npy': str(prefix) + '_sad_error.npy',
            'sad_error_png': str(prefix) + '_sad_error.png',
            'label_npy': str(prefix) + '_label.npy',
            'label_map_png': str(prefix) + '_label_map.png',
            'target_rgb_png': str(prefix) + '_target_rgb.png',
        },
    }
    if mapping is not None:
        result['files']['mapped_class_png'] = \
            str(prefix) + '_dominant_endmember.png'
    if selected_class_indices is not None:
        result['files']['mapped_class_selected_npy'] = \
            str(prefix) + '_mapped_class_selected.npy'
        result['files']['label_index_selected_npy'] = \
            str(prefix) + '_label_index_selected.npy'
        result['files']['label_map_selected_png'] = \
            str(prefix) + '_label_map_selected.png'
        if mapping is not None:
            result['files']['mapped_class_selected_png'] = \
                str(prefix) + '_mapped_class_selected.png'
        result['selected_mapped_class_pixels'] = class_index_summary(
            mapped_class_selected,
            negative_labels={-2: 'other mapped class', -1: 'unmapped'},
            class_names_by_index=class_names_by_index)
        result['selected_label_pixels'] = class_index_summary(
            label_idx_selected,
            negative_labels={-2: 'other label', -1: 'unlabelled'})
    return result


def export_single_image_sample(sample_idx, sample, output, wavelengths,
                               output_dir, mapping=None,
                               class_names_by_index=None,
                               selected_class_indices=None):
    abundances = output['abundances'][0].detach().cpu().numpy()
    reconstruction = output['reconstruction'][0]
    target_item = sample['target_reflectance']
    sad = spectral_angle_map(reconstruction, target_item)
    dominant = np.argmax(abundances, axis=0).astype(np.int16)
    mapped_class = apply_mapping_to_dominant(dominant, mapping)
    mapped_class_selected = None
    if selected_class_indices is not None:
        mapped_class_selected = filter_class_index_map(
            mapped_class, selected_class_indices)

    prefix = output_dir / 'sample_{:03d}'.format(sample_idx)
    np.save(str(prefix) + '_abundances.npy', abundances)
    np.save(str(prefix) + '_dominant_endmember.npy', dominant)
    np.save(str(prefix) + '_mapped_class.npy', mapped_class)
    if selected_class_indices is not None:
        np.save(str(prefix) + '_mapped_class_selected.npy',
                mapped_class_selected)
    np.save(str(prefix) + '_sad_error.npy', sad.detach().cpu().numpy())
    np.save(str(prefix) + '_reconstruction.npy',
            reconstruction.detach().cpu().numpy())
    np.save(str(prefix) + '_target_reflectance.npy',
            target_item.detach().cpu().numpy())

    save_abundance_grid(
        abundances, str(prefix) + '_abundances.png',
        titles=abundance_grid_titles(
            mapping, abundances.shape[0], class_names_by_index),
        figure_title='Patch abundance maps')
    if mapping is not None:
        save_class_index_map(
            mapped_class, str(prefix) + '_dominant_endmember.png',
            'Dominant mapped class from spectral Hungarian',
            class_names_by_index=class_names_by_index)
        if selected_class_indices is not None:
            save_class_index_map(
                mapped_class_selected,
                str(prefix) + '_mapped_class_selected.png',
                'Mapped class selected for visualization',
                negative_labels={-2: 'other mapped class', -1: 'unmapped'},
                class_names_by_index=class_names_by_index)
    else:
        save_map(dominant, str(prefix) + '_dominant_endmember.png',
                 'Dominant endmember', cmap='tab20', vmin=-0.5,
                 vmax=abundances.shape[0] - 0.5)
    save_map(sad.detach().cpu().numpy(), str(prefix) + '_sad_error.png',
             'SAD reconstruction error (rad)', cmap='magma', vmin=0.0)
    save_target_rgb(target_item.detach().cpu().numpy(), wavelengths,
                    str(prefix) + '_target_rgb.png')
    save_target_rgb(reconstruction.detach().cpu().numpy(), wavelengths,
                    str(prefix) + '_reconstruction_rgb.png')

    path = sample['path']
    if isinstance(path, pathlib.Path):
        path = str(path)

    result = {
        'sample': sample_idx,
        'path': path,
        'crop_row': int(sample['crop_row']),
        'crop_col': int(sample['crop_col']),
        'patch_size': int(sample['patch_size']),
        'files': {
            'abundances_npy': str(prefix) + '_abundances.npy',
            'abundances_png': str(prefix) + '_abundances.png',
            'dominant_endmember_npy':
                str(prefix) + '_dominant_endmember.npy',
            'dominant_endmember_png':
                str(prefix) + '_dominant_endmember.png',
            'mapped_class_npy': str(prefix) + '_mapped_class.npy',
            'sad_error_npy': str(prefix) + '_sad_error.npy',
            'sad_error_png': str(prefix) + '_sad_error.png',
            'reconstruction_npy': str(prefix) + '_reconstruction.npy',
            'reconstruction_rgb_png':
                str(prefix) + '_reconstruction_rgb.png',
            'target_reflectance_npy':
                str(prefix) + '_target_reflectance.npy',
            'target_rgb_png': str(prefix) + '_target_rgb.png',
        },
    }
    if mapping is not None:
        result['files']['mapped_class_png'] = \
            str(prefix) + '_dominant_endmember.png'
        result['mapped_class_pixels'] = class_index_summary(
            mapped_class,
            negative_labels={-1: 'unmapped'},
            class_names_by_index=class_names_by_index)
    if selected_class_indices is not None:
        result['files']['mapped_class_selected_npy'] = \
            str(prefix) + '_mapped_class_selected.npy'
        if mapping is not None:
            result['files']['mapped_class_selected_png'] = \
                str(prefix) + '_mapped_class_selected.png'
        result['selected_mapped_class_pixels'] = class_index_summary(
            mapped_class_selected,
            negative_labels={-2: 'other mapped class', -1: 'unmapped'},
            class_names_by_index=class_names_by_index)
    return result


def export_single_full_image(dataset, model, device, wavelengths, output_dir,
                             tile_size, mapping=None,
                             class_names_by_index=None,
                             selected_class_indices=None,
                             label_path=None):
    im_hyper, wl = dataset.load_full_image()
    h, w, _ = im_hyper.shape
    n_endmembers = model.decoder.in_channels
    n_bands = model.decoder.out_channels
    abundances = np.empty((n_endmembers, h, w), dtype=np.float32)
    reconstruction = np.empty((n_bands, h, w), dtype=np.float32)
    sad_error = np.empty((h, w), dtype=np.float32)
    margin = model_context_margin(model)

    for row in range(0, h, tile_size):
        row_end = min(row + tile_size, h)
        for col in range(0, w, tile_size):
            col_end = min(col + tile_size, w)

            in_row = max(0, row - margin)
            in_col = max(0, col - margin)
            in_row_end = min(h, row_end + margin)
            in_col_end = min(w, col_end + margin)

            tile_hyper = im_hyper[in_row:in_row_end,
                                  in_col:in_col_end, :].copy()
            image, target = \
                torchseg.data_loader.OdsiDbDataLoader.LoadImage \
                .read_hyper_image_pair(
                    dataset.image_path, dataset.mode, im_hyper=tile_hyper,
                    wl=wl)
            image = torch.from_numpy(image).unsqueeze(0).to(device)
            target = torch.from_numpy(target).unsqueeze(0).to(device)
            output = model(image)

            inner_row = row - in_row
            inner_col = col - in_col
            inner_row_end = inner_row + (row_end - row)
            inner_col_end = inner_col + (col_end - col)

            abundance_tile = output['abundances'][0, :,
                inner_row:inner_row_end, inner_col:inner_col_end]
            reconstruction_tile = output['reconstruction'][0, :,
                inner_row:inner_row_end, inner_col:inner_col_end]
            target_tile = target[0, :, inner_row:inner_row_end,
                                 inner_col:inner_col_end]
            sad_tile = spectral_angle_map(reconstruction_tile, target_tile)

            abundances[:, row:row_end, col:col_end] = \
                abundance_tile.detach().cpu().numpy()
            reconstruction[:, row:row_end, col:col_end] = \
                reconstruction_tile.detach().cpu().numpy()
            sad_error[row:row_end, col:col_end] = \
                sad_tile.detach().cpu().numpy()

    dominant = abundances.argmax(axis=0).astype(np.int16)
    mapped_class = apply_mapping_to_dominant(dominant, mapping)
    mapped_class_selected = None
    if selected_class_indices is not None:
        mapped_class_selected = filter_class_index_map(
            mapped_class, selected_class_indices)
    target_reflectance = \
        torchseg.data_loader.OdsiDbDataLoader.LoadImage.read_hyper_reflectance(
            dataset.image_path, dataset.mode, im_hyper=im_hyper, wl=wl)
    target_reflectance = target_reflectance.transpose((2, 0, 1))
    target_rgb = torchseg.data_loader.OdsiDbDataLoader.LoadImage.hyper2rgb(
        im_hyper, wl)
    label_idx = None
    label_idx_selected = None
    label_info = None
    if label_path is not None:
        label = torchseg.data_loader.OdsiDbDataLoader.LoadImage.read_label(
            label_path)
        label_idx = label_map(label).astype(np.int16)
        label_info = label_summary(label)
        if selected_class_indices is not None:
            label_idx_selected = filter_class_index_map(
                label_idx, selected_class_indices)

    prefix = output_dir / 'full_000'
    np.save(str(prefix) + '_abundances.npy', abundances)
    np.save(str(prefix) + '_dominant_endmember.npy', dominant)
    np.save(str(prefix) + '_mapped_class.npy', mapped_class)
    if selected_class_indices is not None:
        np.save(str(prefix) + '_mapped_class_selected.npy',
                mapped_class_selected)
    if label_idx is not None:
        np.save(str(prefix) + '_label_index.npy', label_idx)
        if selected_class_indices is not None:
            np.save(str(prefix) + '_label_index_selected.npy',
                    label_idx_selected)
    np.save(str(prefix) + '_sad_error.npy', sad_error)
    np.save(str(prefix) + '_reconstruction.npy', reconstruction)
    np.save(str(prefix) + '_target_reflectance.npy', target_reflectance)

    save_rgb_exact(target_rgb, str(prefix) + '_target_rgb.png')
    save_target_rgb(reconstruction, wavelengths,
                    str(prefix) + '_reconstruction_rgb.png')
    save_abundance_grid(
        abundances, str(prefix) + '_abundances.png',
        titles=abundance_grid_titles(
            mapping, n_endmembers, class_names_by_index),
        figure_title='Full-image abundance maps')
    if mapping is not None:
        save_class_index_map(
            mapped_class, str(prefix) + '_dominant_endmember.png',
            'Dominant mapped class from spectral Hungarian',
            class_names_by_index=class_names_by_index)
        if selected_class_indices is not None:
            save_class_index_map(
                mapped_class_selected,
                str(prefix) + '_mapped_class_selected.png',
                'Mapped class selected for visualization',
                negative_labels={-2: 'other mapped class', -1: 'unmapped'},
                class_names_by_index=class_names_by_index)
    else:
        save_map(dominant, str(prefix) + '_dominant_endmember.png',
                 'Dominant endmember', cmap='tab20', vmin=-0.5,
                 vmax=n_endmembers - 0.5)
    save_map(sad_error, str(prefix) + '_sad_error.png',
             'SAD reconstruction error (rad)', cmap='magma', vmin=0.0)
    if label_idx is not None:
        save_class_index_map(
            label_idx, str(prefix) + '_label_map.png',
            'ODSI annotation class index',
            negative_labels={-1: 'unlabelled'})
        if selected_class_indices is not None:
            save_class_index_map(
                label_idx_selected, str(prefix) + '_label_map_selected.png',
                'ODSI annotation selected class index',
                negative_labels={-2: 'other label', -1: 'unlabelled'})

    result = {
        'full_image': 0,
        'path': dataset.image_path,
        'height': int(h),
        'width': int(w),
        'tile_size': int(tile_size),
        'context_margin': int(margin),
        'files': {
            'abundances_npy': str(prefix) + '_abundances.npy',
            'abundances_png': str(prefix) + '_abundances.png',
            'dominant_endmember_npy':
                str(prefix) + '_dominant_endmember.npy',
            'dominant_endmember_png':
                str(prefix) + '_dominant_endmember.png',
            'mapped_class_npy': str(prefix) + '_mapped_class.npy',
            'sad_error_npy': str(prefix) + '_sad_error.npy',
            'sad_error_png': str(prefix) + '_sad_error.png',
            'reconstruction_npy': str(prefix) + '_reconstruction.npy',
            'reconstruction_rgb_png':
                str(prefix) + '_reconstruction_rgb.png',
            'target_reflectance_npy':
                str(prefix) + '_target_reflectance.npy',
            'target_rgb_png': str(prefix) + '_target_rgb.png',
        },
    }
    if label_idx is not None:
        result['label_pixels'] = label_info
        result['files']['label_index_npy'] = str(prefix) + '_label_index.npy'
        result['files']['label_map_png'] = str(prefix) + '_label_map.png'
    if mapping is not None:
        result['files']['mapped_class_png'] = \
            str(prefix) + '_dominant_endmember.png'
        result['mapped_class_pixels'] = class_index_summary(
            mapped_class,
            negative_labels={-1: 'unmapped'},
            class_names_by_index=class_names_by_index)
    if selected_class_indices is not None:
        result['files']['mapped_class_selected_npy'] = \
            str(prefix) + '_mapped_class_selected.npy'
        if mapping is not None:
            result['files']['mapped_class_selected_png'] = \
                str(prefix) + '_mapped_class_selected.png'
        result['selected_mapped_class_pixels'] = class_index_summary(
            mapped_class_selected,
            negative_labels={-2: 'other mapped class', -1: 'unmapped'},
            class_names_by_index=class_names_by_index)
        if label_idx is not None:
            result['files']['label_index_selected_npy'] = \
                str(prefix) + '_label_index_selected.npy'
            result['files']['label_map_selected_png'] = \
                str(prefix) + '_label_map_selected.png'
            result['selected_label_pixels'] = class_index_summary(
                label_idx_selected,
                negative_labels={-2: 'other label', -1: 'unlabelled'})
    return result


def run_single_image_visualization(args, config, output_dir):
    if args.mapping_source == 'validation':
        raise RuntimeError(
            'Single-image unmixing visualizations do not have validation '
            'labels. Use --mapping-source auto, mask, reference, or none.')

    dataset_config = copy.deepcopy(config['data_loader'])
    dataset_config['args']['num_workers'] = 0
    mode = dataset_config['args']['mode']
    wavelengths = torchseg.data_loader.OdsiDbDataLoader.mode2wl[mode]
    image_path = dataset_config['args']['image_path']

    device, _ = torchseg.utils.setup_gpu_devices(config['n_gpu'])
    model, checkpoint = torchseg.test_unmixing.load_model(
        config, args.resume, device)
    save_endmembers(model, wavelengths, output_dir)

    mapping = None
    class_names_by_index = None
    hungarian_mapping = None
    mapping_class_indices = None
    mapping_reference_counts = None
    mapping_source = args.mapping_source
    label_path = torchseg.test_unmixing.resolve_mask_path(
        args.mask_path, image_path=image_path, required=False)
    if mapping_source == 'auto':
        mapping_source = 'mask' if label_path is not None else 'none'

    if mapping_source == 'mask':
        mapping, cost_matrix, mapping_class_indices, reference_endmembers, \
            reference_counts, mask_path = \
            torchseg.test_unmixing.compute_mask_spectral_mapping(
                model, device, image_path, mode, mask_path=label_path,
                max_sad=args.reference_max_sad,
                mapping_class_scope=args.mapping_class_scope)
        mapping = mapping.cpu().numpy()
        mapping_reference_counts = reference_counts
        hungarian_mapping = save_mapping(
            output_dir, mapping, cost_matrix, mapping_class_indices,
            'auto_mask' if args.mapping_source == 'auto' else 'mask',
            1,
            reference_endmembers=reference_endmembers,
            reference_counts=reference_counts,
            reference_path=mask_path,
            max_sad=args.reference_max_sad,
            mapping_class_scope=args.mapping_class_scope)
    elif mapping_source == 'reference':
        mapping, cost_matrix, mapping_class_indices, reference_endmembers, \
            class_names_by_index, reference_path = \
            torchseg.test_unmixing.compute_reference_spectral_mapping(
                model, device, args.reference_endmembers,
                reference_class_indices=args.reference_class_indices,
                reference_class_names=args.reference_class_names,
                max_sad=args.reference_max_sad)
        mapping = mapping.cpu().numpy()
        hungarian_mapping = save_mapping(
            output_dir, mapping, cost_matrix, mapping_class_indices,
            args.mapping_source, 0,
            reference_endmembers=reference_endmembers,
            class_names_by_index=class_names_by_index,
            reference_path=reference_path,
            max_sad=args.reference_max_sad)
    coverage_class_indices = None if mapping_class_indices is None else \
        mapping_coverage_class_indices(
            mapping_class_indices, mapping_reference_counts)

    data_loader = getattr(torchseg.data_loader, dataset_config['type'])(
        **dataset_config['args'])
    data_loader.training = False
    dataset = data_loader.dataset

    selected_class_indices = None
    selected_class_names = None
    if args.class_names is not None and not args.include_all_classes:
        selected_class_names = args.class_names
        selected_class_indices = class_indices(selected_class_names, False)

    manifest = {
        'checkpoint': args.resume,
        'checkpoint_epoch': checkpoint.get('epoch'),
        'mode': mode,
        'wavelengths_nm': wavelengths.tolist(),
        'image_path': dataset.image_path,
        'samples': [],
        'full_images': [],
        'semantic_mapping': hungarian_mapping if hungarian_mapping is not None
            else {
                'source': 'none',
                'method': None,
                'note': (
                    'Single-image CNNAEU flow has no labels. External '
                    'reference/mask mapping was not available or requested.')
            },
        'hungarian_mapping': hungarian_mapping,
        'selected_map_mapping_coverage': None if hungarian_mapping is None
            else selected_mapping_coverage(
                selected_class_indices, coverage_class_indices, mapping),
        'selection': {
            'class_names': selected_class_names,
            'include_all_classes': args.include_all_classes,
            'mapping_source': args.mapping_source,
            'effective_mapping_source': mapping_source,
            'mapping_class_scope': args.mapping_class_scope,
            'label_path': label_path,
            'selected_map_class_indices': selected_class_indices,
            'selected_map_class_names': selected_class_names,
        },
        'abundance_grid_definition': (
            'abundances.png is a single grid containing every abundance map.'),
        'selected_map_definition': (
            'selected maps are visualization-only filters over mapped_class. '
            'They do not restrict the spectral Hungarian mapping, which uses '
            'all configured relevant classes available in the reference.'),
    }

    sample_count = min(args.num_samples, len(dataset))
    with torch.no_grad():
        for sample_idx in range(sample_count):
            sample = dataset[sample_idx]
            data = sample['image'].unsqueeze(0).to(device)
            output = model(data)
            manifest['samples'].append(
                export_single_image_sample(
                    sample_idx, sample, output, wavelengths, output_dir,
                    mapping=mapping,
                    class_names_by_index=class_names_by_index,
                    selected_class_indices=selected_class_indices))

        if args.full_images != 0:
            manifest['full_images'].append(
                export_single_full_image(dataset, model, device, wavelengths,
                                         output_dir, args.tile_size,
                                         mapping=mapping,
                                         class_names_by_index=
                                         class_names_by_index,
                                         selected_class_indices=
                                         selected_class_indices,
                                         label_path=label_path))
        elif args.full_images == 0:
            manifest['full_images'].append(
                export_single_full_image(dataset, model, device, wavelengths,
                                         output_dir, args.tile_size,
                                         mapping=mapping,
                                         class_names_by_index=
                                         class_names_by_index,
                                         selected_class_indices=
                                         selected_class_indices,
                                         label_path=label_path))

    with (output_dir / 'manifest.json').open('w') as f:
        json.dump(manifest, f, indent=4)
    print('Saved unmixing visualizations to {}'.format(output_dir))


def main():
    args = parse_args()
    if args.device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = torchseg.test_unmixing.read_config(args)
    active_reference_class_names = \
        torchseg.model.metric.configure_odsi_db_unmixing_from_config(config)
    if torchseg.test_unmixing.is_single_image_config(config):
        run_single_image_visualization(args, config, output_dir)
        return

    selection_class_names = args.class_names
    if selection_class_names is None:
        selection_class_names = active_reference_class_names

    dataset_config = config['testing']['datasets'][args.dataset_index]
    dataset_config['args']['num_workers'] = 0

    mode = dataset_config['args']['mode']
    wavelengths = torchseg.data_loader.OdsiDbDataLoader.mode2wl[mode]

    device, _ = torchseg.utils.setup_gpu_devices(config['n_gpu'])
    model, checkpoint = torchseg.test_unmixing.load_model(
        config, args.resume, device)
    save_endmembers(model, wavelengths, output_dir)

    data_loader = getattr(torchseg.data_loader, dataset_config['type'])(
        **dataset_config['args'])
    data_loader.training = False
    dataset = data_loader.dataset
    wanted_class_indices = class_indices(selection_class_names,
                                         args.include_all_classes)
    selected_class_indices = None if args.include_all_classes \
        else wanted_class_indices
    idx2class = torchseg.data_loader.OdsiDbDataLoader.OdsiDbDataset.classnames
    selected_class_names = None if selected_class_indices is None else [
        idx2class[idx] for idx in selected_class_indices
    ]

    manifest = {
        'checkpoint': args.resume,
        'checkpoint_epoch': checkpoint.get('epoch'),
        'mode': mode,
        'wavelengths_nm': wavelengths.tolist(),
        'selection': {
            'class_names': selection_class_names,
            'include_all_classes': args.include_all_classes,
            'min_label_fraction': args.min_label_fraction,
            'max_samples_per_image': args.max_samples_per_image,
            'max_tries_per_image': args.max_tries_per_image,
            'full_images': args.full_images,
            'tile_size': args.tile_size,
            'full_top_abundances': args.full_top_abundances,
            'mapping_source': args.mapping_source,
            'effective_mapping_source':
                'validation' if args.mapping_source == 'auto'
                else args.mapping_source,
            'mapping_class_scope': args.mapping_class_scope,
            'mapping_max_batches': args.mapping_max_batches,
            'selected_map_class_indices': selected_class_indices,
            'selected_map_class_names': selected_class_names,
            'selected_class_presence': dataset_class_presence(
                dataset, selected_class_indices),
        },
        'active_reference_class_names': active_reference_class_names,
        'samples': [],
        'full_images': [],
        'hungarian_mapping': None,
        'selected_map_mapping_coverage': None,
        'mapped_class_definition': (
            'dominant_endmember.npy is the raw argmax over abundances. '
            'dominant_endmember.png and mapped_class apply a validation-set '
            'spectral Hungarian mapping from decoder endmember spectra to '
            'class-mean reference spectra. When mapping is available, the PNG '
            'shows mapped classes for every dominant endmember, not raw '
            'endmember IDs.'),
        'abundance_grid_definition': (
            'abundances.png is a single grid containing every abundance map. '
            'Each panel title shows the endmember ID and its spectral '
            'Hungarian class pairing when a mapping is available. Individual '
            'per-endmember abundance PNGs are intentionally not exported.'),
        'selected_map_definition': (
            'selected maps are visualization-only filters over mapped_class '
            'and label_map. They do not restrict the spectral Hungarian mapping; '
            'non-selected known classes are encoded as other.'),
    }

    rng = np.random.RandomState(args.seed)
    mapping = None
    class_names_by_index = None
    mapping_reference_counts = None
    mapping_source = 'validation' if args.mapping_source == 'auto' \
        else args.mapping_source
    if mapping_source == 'validation':
        mapping, cost_matrix, mapping_class_indices, batches_used, \
            reference_endmembers, reference_counts = \
            learn_validation_mapping(config, model, device,
                                     max_batches=args.mapping_max_batches,
                                     mapping_class_scope=
                                     args.mapping_class_scope)
        mapping_reference_counts = reference_counts
        manifest['hungarian_mapping'] = save_mapping(
            output_dir, mapping, cost_matrix, mapping_class_indices,
            'auto_validation' if args.mapping_source == 'auto'
            else args.mapping_source, batches_used,
            reference_endmembers=reference_endmembers,
            reference_counts=reference_counts,
            mapping_class_scope=args.mapping_class_scope)
        manifest['selected_map_mapping_coverage'] = \
            selected_mapping_coverage(selected_class_indices,
                                      mapping_coverage_class_indices(
                                          mapping_class_indices,
                                          mapping_reference_counts),
                                      mapping)
    elif mapping_source == 'reference':
        mapping, cost_matrix, mapping_class_indices, reference_endmembers, \
            class_names_by_index, reference_path = \
            torchseg.test_unmixing.compute_reference_spectral_mapping(
                model, device, args.reference_endmembers,
                reference_class_indices=args.reference_class_indices,
                reference_class_names=args.reference_class_names,
                max_sad=args.reference_max_sad)
        mapping = mapping.cpu().numpy()
        manifest['hungarian_mapping'] = save_mapping(
            output_dir, mapping, cost_matrix, mapping_class_indices,
            args.mapping_source, 0,
            reference_endmembers=reference_endmembers,
            class_names_by_index=class_names_by_index,
            reference_path=reference_path,
            max_sad=args.reference_max_sad)
        manifest['selected_map_mapping_coverage'] = \
            selected_mapping_coverage(selected_class_indices,
                                      mapping_class_indices, mapping)

    samples = collect_samples(dataset, wanted_class_indices, args.num_samples,
                              rng, args.min_label_fraction,
                              args.max_tries_per_image,
                              args.max_samples_per_image)
    with torch.no_grad():
        for sample_idx, sample in enumerate(samples):
            data = sample['image'].unsqueeze(0).to(device)
            sample['target_reflectance'] = \
                sample['target_reflectance'].to(device)
            output = model(data)
            sample_info = export_sample(sample_idx, sample, output,
                                        wavelengths, output_dir,
                                        mapping=mapping,
                                        selected_class_indices=
                                        selected_class_indices,
                                        class_names_by_index=
                                        class_names_by_index)
            manifest['samples'].append(sample_info)

        full_image_infos = select_full_images(dataset, wanted_class_indices,
                                              args.full_images, rng)
        for full_idx, image_info in enumerate(full_image_infos):
            full_info = export_full_image(full_idx, image_info, dataset, model,
                                          device, wavelengths, output_dir,
                                          args.tile_size,
                                          args.full_top_abundances,
                                          mapping=mapping,
                                          selected_class_indices=
                                          selected_class_indices,
                                          class_names_by_index=
                                          class_names_by_index)
            manifest['full_images'].append(full_info)

    with (output_dir / 'manifest.json').open('w') as f:
        json.dump(manifest, f, indent=4)
    print('Saved unmixing visualizations to {}'.format(output_dir))


if __name__ == '__main__':
    main()
