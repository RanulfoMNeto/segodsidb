"""
@brief Utilities to keep the per-image CNNAEU unmixing manifest up to date.
"""

import datetime
import json
import math
import os
import pathlib
import re


def as_config_dict(config):
    return config.config if hasattr(config, 'config') else config


def monitor_info(config):
    cfg = as_config_dict(config)
    monitor = cfg.get('machine', {}).get('args', {}).get('monitor', 'off')
    if monitor == 'off':
        return monitor, None, None
    parts = monitor.split()
    if len(parts) != 2:
        return monitor, None, None
    return monitor, parts[0], parts[1]


def infer_manifest_path(image_path, default_dir='results'):
    image_path = pathlib.Path(str(image_path))
    fold_name = None
    for part in image_path.parts:
        if re.fullmatch(r'fold_\d+', part):
            fold_name = part.replace('_', '')
            break

    if fold_name is None:
        filename = 'unmixing_manifest.json'
    else:
        filename = '{}_unmixing_manifest.json'.format(fold_name)
    return pathlib.Path(default_dir) / filename


def resolve_manifest_path(config):
    cfg = as_config_dict(config)
    unmixing_cfg = cfg.get('unmixing', {})
    data_args = cfg.get('data_loader', {}).get('args', {})
    manifest_path = unmixing_cfg.get('manifest_path', 'auto')
    if manifest_path in [None, '', 'auto']:
        manifest_path = infer_manifest_path(data_args['image_path'])
    return pathlib.Path(manifest_path)


def portable_path(path, base_dir=None):
    path = pathlib.Path(path).resolve()
    if base_dir is None:
        base_dir = pathlib.Path.cwd().resolve()
    else:
        base_dir = pathlib.Path(base_dir).resolve()
    return os.path.relpath(str(path), str(base_dir))


def read_manifest(path):
    path = pathlib.Path(path)
    if not path.is_file():
        return None
    with path.open() as f:
        return json.load(f)


def write_manifest(path, manifest):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(manifest, f, indent=4)


def latest_checkpoint(run_dir):
    run_dir = pathlib.Path(run_dir)
    best = run_dir / 'model_best.pth'
    if best.is_file():
        return best

    checkpoints = sorted(
        run_dir.glob('checkpoint-epoch*.pth'),
        key=lambda path: path.stat().st_mtime)
    if checkpoints:
        return checkpoints[-1]
    raise FileNotFoundError(
        'No checkpoint found in run directory: {}'.format(run_dir))


def checkpoint_metadata(checkpoint=None):
    if checkpoint is None:
        return {}
    monitor_best = checkpoint.get('monitor_best')
    try:
        monitor_best = float(monitor_best)
    except (TypeError, ValueError):
        monitor_best = None
    if monitor_best is not None and not math.isfinite(monitor_best):
        monitor_best = None
    return {
        'epoch': checkpoint.get('epoch'),
        'monitor_best': monitor_best,
    }


def entry_from_training_run(config, run_dir, checkpoint_path,
                            checkpoint=None, base_dir=None):
    cfg = as_config_dict(config)
    data_args = cfg['data_loader']['args']
    monitor, monitor_mode, monitor_metric = monitor_info(cfg)
    metadata = checkpoint_metadata(checkpoint)
    mode = data_args.get('mode')

    entry = {
        'image_path': portable_path(data_args['image_path'], base_dir=base_dir),
        'image_basename': pathlib.Path(data_args['image_path']).name,
        'checkpoint': portable_path(checkpoint_path, base_dir=base_dir),
        'config': portable_path(pathlib.Path(run_dir) / 'config.json',
                                base_dir=base_dir),
        'run_dir': portable_path(run_dir, base_dir=base_dir),
        'mode': mode,
        'monitor': monitor,
        'monitor_mode': monitor_mode,
        'monitor_metric': monitor_metric,
        'monitor_best': metadata.get('monitor_best'),
        'epoch': metadata.get('epoch'),
        'updated_at': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    return entry


def entries_refer_to_same_image(left, right):
    return pathlib.Path(left['image_path']).name == \
        pathlib.Path(right['image_path']).name


def candidate_is_better(candidate, existing):
    if existing is None:
        return True, 'new_image'

    cand_mode = candidate.get('monitor_mode')
    cand_metric = candidate.get('monitor_metric')
    old_mode = existing.get('monitor_mode')
    old_metric = existing.get('monitor_metric')
    cand_score = candidate.get('monitor_best')
    old_score = existing.get('monitor_best')

    if cand_score is None:
        return False, 'candidate_has_no_monitor_score'
    if old_score is None:
        return True, 'existing_has_no_monitor_score'
    if cand_mode != old_mode or cand_metric != old_metric:
        return False, 'monitor_not_comparable'

    if cand_mode == 'min' and cand_score < old_score:
        return True, 'lower_monitor_score'
    if cand_mode == 'max' and cand_score > old_score:
        return True, 'higher_monitor_score'
    return False, 'existing_monitor_score_is_better'


def update_manifest(path, candidate_entry):
    path = pathlib.Path(path)
    manifest = read_manifest(path)
    if manifest is None:
        manifest = {
            'mode': candidate_entry.get('mode'),
            'selection_policy': 'best_monitor_score_per_image',
            'items': [],
        }

    manifest_mode = manifest.get('mode')
    candidate_mode = candidate_entry.get('mode')
    if manifest_mode is not None and candidate_mode is not None and \
            manifest_mode != candidate_mode:
        raise ValueError(
            'Manifest mode {} does not match candidate mode {}.'.format(
                manifest_mode, candidate_mode))
    if manifest_mode is None:
        manifest['mode'] = candidate_mode

    existing_index = None
    existing_entry = None
    for idx, entry in enumerate(manifest.get('items', [])):
        if entries_refer_to_same_image(entry, candidate_entry):
            existing_index = idx
            existing_entry = entry
            break

    should_update, reason = candidate_is_better(
        candidate_entry, existing_entry)
    if should_update:
        if existing_index is None:
            manifest.setdefault('items', []).append(candidate_entry)
        else:
            manifest['items'][existing_index] = candidate_entry
        manifest['items'] = sorted(
            manifest['items'], key=lambda item: item['image_basename'])
        write_manifest(path, manifest)

    return {
        'updated': should_update,
        'reason': reason,
        'manifest_path': str(path),
        'image_basename': candidate_entry['image_basename'],
    }
