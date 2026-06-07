"""
@brief Generate weakly expanded ODSI-DB masks from CNNAEU unmixing outputs.

The output directory keeps the same layout expected by OdsiDbDataLoader:
HSI files are linked/copied from the source directory and masks are written as
ODSI multi-page TIFF files. Unlabelled pixels that do not pass all confidence
checks remain empty, so the existing segmentation loss ignores them.
"""

import argparse
import collections
import json
import os
import pathlib
import shutil

import numpy as np
import torch
import torch.nn.functional as F

import torchseg.data_loader
import torchseg.model
import torchseg.test_unmixing
import torchseg.utils


DEFAULT_MAX_ABUNDANCE = 0.85
DEFAULT_MIN_ABUNDANCE_MARGIN = 0.20
DEFAULT_MAX_REFERENCE_SAD = 0.15
DEFAULT_MAX_RECONSTRUCTION_SAD = 0.10
DEFAULT_MAX_PSEUDO_RATIO = 5.0


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate weak ODSI-DB masks from unmixing checkpoints.')
    parser.add_argument('--source-data-dir', required=True, type=str,
                        help='ODSI-DB train directory with original masks.')
    parser.add_argument('--output-data-dir', required=True, type=str,
                        help='Output directory compatible with OdsiDbDataLoader.')
    parser.add_argument('--unmixing-manifest', required=True, type=str,
                        help='JSON mapping each train image to CNNAEU config/checkpoint.')
    parser.add_argument('--mode', default='simage_170', type=str,
                        help='Hyperspectral preprocessing mode.')
    parser.add_argument('--tile-size', default=256, type=int,
                        help='Full-image unmixing tile size.')
    parser.add_argument('--device', default=None, type=str,
                        help='CUDA device ids, e.g. "0"; CPU is used if CUDA is unavailable.')
    parser.add_argument('--max-abundance', default=DEFAULT_MAX_ABUNDANCE,
                        type=float)
    parser.add_argument('--min-abundance-margin',
                        default=DEFAULT_MIN_ABUNDANCE_MARGIN, type=float)
    parser.add_argument('--max-reference-sad',
                        default=DEFAULT_MAX_REFERENCE_SAD, type=float)
    parser.add_argument('--max-reconstruction-sad',
                        default=DEFAULT_MAX_RECONSTRUCTION_SAD, type=float)
    parser.add_argument('--max-pseudo-ratio',
                        default=DEFAULT_MAX_PSEUDO_RATIO, type=float,
                        help='Maximum pseudo pixels per class as a multiple of '
                             'the original labelled pixels in that image.')
    parser.add_argument('--mapping-max-sad', default=None, type=float,
                        help='Optional maximum SAD for endmember-to-class mapping.')
    parser.add_argument('--copy-images', action='store_true',
                        help='Copy HSI files instead of creating symlinks.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite generated files inside output-data-dir.')
    return parser.parse_args()


def thresholds_from_args(args):
    return {
        'max_abundance': float(args.max_abundance),
        'min_abundance_margin': float(args.min_abundance_margin),
        'max_reference_sad': float(args.max_reference_sad),
        'max_reconstruction_sad': float(args.max_reconstruction_sad),
    }


def resolve_existing_path(path, description, base_dirs=None):
    path = pathlib.Path(path).expanduser()
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(pathlib.Path.cwd() / path)
        if base_dirs is not None:
            candidates.extend(pathlib.Path(base) / path for base in base_dirs)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    tried = ', '.join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        '{} not found. Tried: {}'.format(description, tried))


def read_unmixing_manifest(manifest_path, mode):
    manifest_path = resolve_existing_path(
        manifest_path, 'unmixing manifest')
    with manifest_path.open() as f:
        manifest = json.load(f)

    manifest_mode = manifest.get('mode')
    if manifest_mode is not None and manifest_mode != mode:
        raise ValueError(
            'Manifest mode {} does not match requested mode {}.'.format(
                manifest_mode, mode))

    items = manifest.get('items')
    if not items:
        raise ValueError('Manifest must contain a non-empty "items" list.')

    by_name = {}
    base_dirs = [manifest_path.parent]
    for pos, item in enumerate(items):
        for key in ['image_path', 'checkpoint', 'config']:
            if key not in item:
                raise ValueError(
                    'Manifest item {} is missing "{}".'.format(pos, key))

        resolved = dict(item)
        resolved['image_path'] = resolve_existing_path(
            item['image_path'], 'manifest image_path', base_dirs)
        resolved['checkpoint'] = resolve_existing_path(
            item['checkpoint'], 'manifest checkpoint', base_dirs)
        resolved['config'] = resolve_existing_path(
            item['config'], 'manifest config', base_dirs)

        name = resolved['image_path'].name
        if name in by_name:
            raise ValueError(
                'Manifest contains duplicate image basename: {}'.format(name))
        by_name[name] = resolved

    return manifest_path, by_name


def source_image_mask_pairs(source_data_dir):
    source_data_dir = resolve_existing_path(
        source_data_dir, 'source data directory')
    pairs = []
    for mask_path in sorted(source_data_dir.glob('*_masks.tif')):
        image_name = mask_path.name.replace('_masks.tif', '.tif')
        image_path = source_data_dir / image_name
        if not image_path.is_file():
            raise FileNotFoundError(
                'Expected image for mask {}: {}'.format(
                    mask_path, image_path))
        pairs.append((image_path.resolve(), mask_path.resolve()))
    if not pairs:
        raise RuntimeError(
            'No ODSI-DB mask files found in {}'.format(source_data_dir))
    return pairs


def ensure_image_available(source_image, output_image, overwrite=False,
                           copy_images=False):
    output_image.parent.mkdir(parents=True, exist_ok=True)
    if output_image.exists() or output_image.is_symlink():
        if overwrite:
            output_image.unlink()
        elif output_image.resolve() == source_image.resolve():
            return
        else:
            raise FileExistsError(
                'Output image already exists: {}'.format(output_image))

    if copy_images:
        shutil.copy2(str(source_image), str(output_image))
    else:
        rel_source = os.path.relpath(str(source_image), str(output_image.parent))
        os.symlink(rel_source, str(output_image))


def spectral_angle_numpy(a, b, eps=1e-8):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    numerator = np.sum(a * b, axis=-1)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    cosine = numerator / np.maximum(denom, eps)
    cosine = np.clip(cosine, -1.0 + eps, 1.0)
    cosine = np.where(cosine > 1.0 - 1e-6, 1.0, cosine)
    return np.arccos(cosine).astype(np.float32)


def spectral_angle_map_torch(reconstruction, target, eps=1e-8):
    reconstruction = reconstruction.to(dtype=torch.float32)
    target = target.to(device=reconstruction.device, dtype=torch.float32)
    rec_norm = F.normalize(reconstruction, p=2, dim=0, eps=eps)
    target_norm = F.normalize(target, p=2, dim=0, eps=eps)
    cosine = torch.sum(rec_norm * target_norm, dim=0)
    cosine = torch.clamp(cosine, min=-1.0 + eps, max=1.0)
    cosine = torch.where(cosine > 1.0 - 1e-6,
                         torch.ones_like(cosine), cosine)
    return torch.acos(cosine)


def compute_reference_spectra(target_reflectance_chw, label_chw):
    target = np.asarray(target_reflectance_chw, dtype=np.float32)
    label = np.asarray(label_chw) > 0
    n_classes = label.shape[0]
    n_bands = target.shape[0]
    references = np.zeros((n_classes, n_bands), dtype=np.float32)
    counts = label.reshape(n_classes, -1).sum(axis=1).astype(np.int64)

    for class_idx in np.where(counts > 0)[0].tolist():
        pixels = target[:, label[class_idx]].transpose((1, 0))
        references[class_idx] = pixels.mean(axis=0)

    return references, counts


def compute_semantic_mapping(model, reference_spectra, reference_counts,
                             device, max_sad=None):
    present = np.where(np.asarray(reference_counts) > 0)[0].astype(np.int64)
    if present.shape[0] == 0:
        raise RuntimeError('Cannot map endmembers: no labelled pixels found.')

    base_model = model.module if hasattr(model, 'module') else model
    estimated = base_model.get_endmembers().detach().to(device)
    references = torch.as_tensor(
        reference_spectra[present], device=device, dtype=torch.float32)
    class_indices = torch.as_tensor(present, device=device, dtype=torch.long)
    mapping, cost_matrix = torchseg.model.metric \
        .odsi_db_unmixing_hungarian_mapping_from_reference_endmembers(
            estimated, references, class_indices,
            max_sad=max_sad, device='cpu', return_cost_matrix=True)
    return mapping.cpu().numpy().astype(np.int64), cost_matrix, present


def model_context_margin(model):
    base_model = model.module if hasattr(model, 'module') else model
    margin = 0
    if hasattr(base_model, 'encoder_conv1'):
        margin += int(base_model.encoder_conv1.kernel_size[0]) // 2
    if hasattr(base_model, 'decoder'):
        margin += int(base_model.decoder.kernel_size[0]) // 2
    return margin


def run_tiled_unmixing(model, image_path, im_hyper, wl, mode, tile_size,
                       device):
    base_model = model.module if hasattr(model, 'module') else model
    n_endmembers = base_model.get_endmembers().shape[0]
    h, w, _ = im_hyper.shape
    abundances = np.empty((n_endmembers, h, w), dtype=np.float32)
    reconstruction_sad = np.empty((h, w), dtype=np.float32)
    margin = model_context_margin(model)

    model.eval()
    with torch.no_grad():
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
                image, target = torchseg.data_loader.OdsiDbDataLoader \
                    .LoadImage.read_hyper_image_pair(
                        str(image_path), mode, im_hyper=tile_hyper, wl=wl)
                image = torch.from_numpy(image).unsqueeze(0).to(device)
                target = torch.from_numpy(target).to(device)
                output = model(image)
                if not isinstance(output, dict) or 'abundances' not in output:
                    raise TypeError('Unmixing model must return abundances.')

                inner_row = row - in_row
                inner_col = col - in_col
                inner_row_end = inner_row + (row_end - row)
                inner_col_end = inner_col + (col_end - col)

                abundance_tile = output['abundances'][0, :,
                    inner_row:inner_row_end, inner_col:inner_col_end]
                reconstruction_tile = output['reconstruction'][0, :,
                    inner_row:inner_row_end, inner_col:inner_col_end]
                target_tile = target[:,
                    inner_row:inner_row_end, inner_col:inner_col_end]
                sad_tile = spectral_angle_map_torch(
                    reconstruction_tile, target_tile)

                abundances[:, row:row_end, col:col_end] = \
                    abundance_tile.detach().cpu().numpy()
                reconstruction_sad[row:row_end, col:col_end] = \
                    sad_tile.detach().cpu().numpy()

    return abundances, reconstruction_sad


def apply_mapping_to_dominant(dominant_endmember, semantic_mapping):
    mapped = np.full(dominant_endmember.shape, -1, dtype=np.int16)
    semantic_mapping = np.asarray(semantic_mapping, dtype=np.int64)
    for endmember_idx, class_idx in enumerate(semantic_mapping.tolist()):
        if class_idx >= 0:
            mapped[dominant_endmember == endmember_idx] = int(class_idx)
    return mapped


def reference_sad_for_mapped_pixels(target_reflectance_chw,
                                    reference_spectra,
                                    mapped_class):
    reference_sad = np.full(mapped_class.shape, np.inf, dtype=np.float32)
    for class_idx in sorted(np.unique(mapped_class).astype(int).tolist()):
        if class_idx < 0:
            continue
        class_pixels = mapped_class == class_idx
        spectra = target_reflectance_chw[:, class_pixels].transpose((1, 0))
        reference = reference_spectra[class_idx][None, :]
        reference_sad[class_pixels] = spectral_angle_numpy(spectra, reference)
    return reference_sad


def build_expanded_label(label_chw, abundances, semantic_mapping,
                         target_reflectance_chw, reference_spectra,
                         reconstruction_sad, thresholds,
                         max_pseudo_ratio=DEFAULT_MAX_PSEUDO_RATIO):
    label = np.asarray(label_chw) > 0
    expanded = label.copy()
    original_any = label.sum(axis=0) > 0
    original_counts = label.reshape(label.shape[0], -1).sum(axis=1)

    sorted_abundances = np.sort(np.asarray(abundances, dtype=np.float32),
                                axis=0)
    top1 = sorted_abundances[-1]
    top2 = sorted_abundances[-2] if sorted_abundances.shape[0] > 1 \
        else np.zeros_like(top1)
    margin = top1 - top2
    dominant = np.asarray(abundances).argmax(axis=0).astype(np.int16)
    mapped_class = apply_mapping_to_dominant(dominant, semantic_mapping)
    reference_sad = reference_sad_for_mapped_pixels(
        target_reflectance_chw, reference_spectra, mapped_class)

    eligible = np.logical_and.reduce([
        ~original_any,
        mapped_class >= 0,
        top1 >= float(thresholds['max_abundance']),
        margin >= float(thresholds['min_abundance_margin']),
        reference_sad <= float(thresholds['max_reference_sad']),
        reconstruction_sad <= float(thresholds['max_reconstruction_sad']),
        np.isfinite(reference_sad),
        np.isfinite(reconstruction_sad),
    ])

    accepted_counts = np.zeros((label.shape[0],), dtype=np.int64)
    candidate_counts = np.zeros((label.shape[0],), dtype=np.int64)
    confidence = top1 + margin - reference_sad - reconstruction_sad
    confidence = np.nan_to_num(confidence, nan=-np.inf, posinf=-np.inf,
                               neginf=-np.inf)

    for class_idx in sorted(np.unique(mapped_class[eligible]).astype(int).tolist()):
        if class_idx < 0:
            continue
        class_candidates = np.logical_and(eligible, mapped_class == class_idx)
        ys, xs = np.where(class_candidates)
        candidate_counts[class_idx] = int(len(ys))
        max_new_pixels = int(np.floor(
            float(max_pseudo_ratio) * int(original_counts[class_idx])))
        if max_new_pixels <= 0 or len(ys) == 0:
            continue

        if len(ys) > max_new_pixels:
            scores = confidence[ys, xs]
            keep_order = np.argsort(scores)[::-1][:max_new_pixels]
            ys = ys[keep_order]
            xs = xs[keep_order]

        expanded[class_idx, ys, xs] = True
        accepted_counts[class_idx] = int(len(ys))

    stats = {
        'original_counts': original_counts.astype(np.int64),
        'candidate_counts': candidate_counts,
        'accepted_counts': accepted_counts,
        'total_original_pixels': int(original_any.sum()),
        'total_candidate_pixels': int(candidate_counts.sum()),
        'total_accepted_pixels': int(accepted_counts.sum()),
    }
    return expanded.astype(np.float32), stats


def label_to_mask_dict(label_chw):
    idx2class = torchseg.data_loader.OdsiDbDataLoader.OdsiDbDataset.classnames
    masks = collections.OrderedDict()
    label = np.asarray(label_chw) > 0
    for class_idx in sorted(idx2class.keys()):
        if class_idx < label.shape[0] and np.any(label[class_idx]):
            masks[idx2class[class_idx]] = label[class_idx]
    return masks


def count_summary(counts):
    idx2class = torchseg.data_loader.OdsiDbDataLoader.OdsiDbDataset.classnames
    result = {}
    for class_idx, value in enumerate(np.asarray(counts).tolist()):
        if value:
            result[idx2class[class_idx]] = int(value)
    return result


def load_unmixing_model(config_path, checkpoint_path, device):
    config = torchseg.utils.read_json(config_path)
    torchseg.model.configure_odsi_db_unmixing_from_config(config)
    model, checkpoint = torchseg.test_unmixing.load_model(
        config, checkpoint_path, device)
    return model, checkpoint, config


def process_image(image_path, mask_path, manifest_item, output_data_dir,
                  mode, thresholds, tile_size, device, max_pseudo_ratio,
                  mapping_max_sad=None, overwrite=False, copy_images=False):
    output_image = output_data_dir / image_path.name
    output_mask = output_data_dir / mask_path.name
    ensure_image_available(
        image_path, output_image, overwrite=overwrite,
        copy_images=copy_images)
    if output_mask.exists() and not overwrite:
        raise FileExistsError(
            'Output mask already exists: {}'.format(output_mask))

    model, checkpoint, config = load_unmixing_model(
        manifest_item['config'], manifest_item['checkpoint'], device)

    im_hyper, wl, _, _ = torchseg.data_loader.read_stiff(
        str(image_path), silent=True, rgb_only=False)
    target_hwc = torchseg.data_loader.OdsiDbDataLoader.LoadImage \
        .read_hyper_reflectance(str(image_path), mode,
                                im_hyper=im_hyper, wl=wl)
    target_chw = target_hwc.transpose((2, 0, 1)).astype(np.float32)
    label = torchseg.data_loader.OdsiDbDataLoader.LoadImage.read_label(
        str(mask_path))

    reference_spectra, reference_counts = compute_reference_spectra(
        target_chw, label)
    mapping, cost_matrix, class_indices = compute_semantic_mapping(
        model, reference_spectra, reference_counts, device,
        max_sad=mapping_max_sad)
    abundances, reconstruction_sad = run_tiled_unmixing(
        model, image_path, im_hyper, wl, mode, tile_size, device)
    expanded_label, stats = build_expanded_label(
        label, abundances, mapping, target_chw, reference_spectra,
        reconstruction_sad, thresholds, max_pseudo_ratio=max_pseudo_ratio)

    masks = label_to_mask_dict(expanded_label)
    torchseg.data_loader.write_mtiff(str(output_mask), masks)

    idx2class = torchseg.data_loader.OdsiDbDataLoader.OdsiDbDataset.classnames
    return {
        'image_path': str(image_path),
        'mask_path': str(mask_path),
        'output_image_path': str(output_image),
        'output_mask_path': str(output_mask),
        'checkpoint': str(manifest_item['checkpoint']),
        'config': str(manifest_item['config']),
        'checkpoint_epoch': checkpoint.get('epoch'),
        'model_type': config['model']['type'],
        'mapping': [
            {
                'endmember': int(endmember_idx),
                'class_index': int(class_idx),
                'class_name': idx2class[int(class_idx)]
                    if int(class_idx) >= 0 else None,
            }
            for endmember_idx, class_idx in enumerate(mapping.tolist())
        ],
        'mapping_class_indices': class_indices.astype(int).tolist(),
        'mapping_cost_matrix_sad': np.asarray(cost_matrix).tolist(),
        'original_pixels_by_class': count_summary(stats['original_counts']),
        'candidate_pixels_by_class': count_summary(stats['candidate_counts']),
        'accepted_pixels_by_class': count_summary(stats['accepted_counts']),
        'total_original_pixels': stats['total_original_pixels'],
        'total_candidate_pixels': stats['total_candidate_pixels'],
        'total_accepted_pixels': stats['total_accepted_pixels'],
    }


def main():
    args = parse_args()
    if args.device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    source_pairs = source_image_mask_pairs(args.source_data_dir)
    manifest_path, manifest_items = read_unmixing_manifest(
        args.unmixing_manifest, args.mode)
    output_data_dir = pathlib.Path(args.output_data_dir).expanduser()
    output_data_dir.mkdir(parents=True, exist_ok=True)
    thresholds = thresholds_from_args(args)

    results = {
        'source_data_dir': str(resolve_existing_path(
            args.source_data_dir, 'source data directory')),
        'output_data_dir': str(output_data_dir.resolve()),
        'unmixing_manifest': str(manifest_path),
        'mode': args.mode,
        'thresholds': thresholds,
        'max_pseudo_ratio': float(args.max_pseudo_ratio),
        'mapping_max_sad': args.mapping_max_sad,
        'tile_size': int(args.tile_size),
        'items': [],
    }

    for image_path, mask_path in source_pairs:
        if image_path.name not in manifest_items:
            raise RuntimeError(
                'No unmixing manifest item found for {}'.format(image_path))
        item_result = process_image(
            image_path, mask_path, manifest_items[image_path.name],
            output_data_dir, args.mode, thresholds, int(args.tile_size),
            device, float(args.max_pseudo_ratio),
            mapping_max_sad=args.mapping_max_sad,
            overwrite=args.overwrite, copy_images=args.copy_images)
        print(json.dumps(item_result, indent=4))
        results['items'].append(item_result)

    summary_path = output_data_dir / 'weak_expansion_summary.json'
    torchseg.utils.write_json(results, summary_path)
    print('Wrote weak expansion summary: {}'.format(summary_path))


if __name__ == '__main__':
    main()
