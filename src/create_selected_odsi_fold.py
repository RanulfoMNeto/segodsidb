"""
@brief Create a small ODSI-DB fold from a CSV of selected images.
"""

import argparse
import csv
import json
import os
import pathlib


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create a symlinked ODSI-DB fold from selected images.')
    parser.add_argument('--csv', default='results/selected_unmixing_images.csv',
                        type=str, help='CSV with a path column.')
    parser.add_argument('--output-dir',
                        default='generated/odsi_db/folds/foldX',
                        type=str, help='Output fold directory.')
    parser.add_argument('--test-count', default=2, type=int,
                        help='Number of selected images reserved for test.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace existing symlinks/files in output.')
    return parser.parse_args()


def resolve_path(path):
    path = pathlib.Path(path).expanduser()
    if path.is_file():
        return path.resolve()
    sibling = pathlib.Path.cwd().parent / path
    if sibling.is_file():
        return sibling.resolve()
    raise FileNotFoundError('Selected image not found: {}'.format(path))


def mask_path_for_image(image_path):
    image_path = pathlib.Path(image_path)
    mask_path = image_path.with_name(
        image_path.stem + '_masks' + image_path.suffix)
    if not mask_path.is_file():
        raise FileNotFoundError(
            'Mask not found for {}: {}'.format(image_path, mask_path))
    return mask_path.resolve()


def link_or_replace(source, destination, overwrite=False):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            return
        destination.unlink()
    relative_source = os.path.relpath(str(source), str(destination.parent))
    os.symlink(relative_source, str(destination))


def read_selected(csv_path):
    with pathlib.Path(csv_path).open(newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError('Selection CSV is empty: {}'.format(csv_path))
    if 'path' not in rows[0]:
        raise RuntimeError('Selection CSV must contain a "path" column.')
    return rows


def main():
    args = parse_args()
    rows = read_selected(args.csv)
    if args.test_count <= 0 or args.test_count >= len(rows):
        raise ValueError(
            '--test-count must be between 1 and len(selection)-1.')

    output_dir = pathlib.Path(args.output_dir)
    train_rows = rows[:-args.test_count]
    test_rows = rows[-args.test_count:]
    manifest = {
        'source_csv': args.csv,
        'output_dir': str(output_dir),
        'split_policy': 'csv_order_train_then_last_{}_test'.format(
            args.test_count),
        'train': [],
        'test': [],
    }

    for split, split_rows in [('train', train_rows), ('test', test_rows)]:
        split_dir = output_dir / split
        for row in split_rows:
            image_path = resolve_path(row['path'])
            mask_path = mask_path_for_image(image_path)
            image_dst = split_dir / image_path.name
            mask_dst = split_dir / mask_path.name
            link_or_replace(image_path, image_dst, overwrite=args.overwrite)
            link_or_replace(mask_path, mask_dst, overwrite=args.overwrite)
            manifest[split].append({
                'image_path': str(image_path),
                'mask_path': str(mask_path),
                'image_link': str(image_dst),
                'mask_link': str(mask_dst),
                'filename': image_path.name,
            })

    manifest_path = output_dir / 'selected_fold_manifest.json'
    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open('w') as f:
        json.dump(manifest, f, indent=4)

    print('Created selected fold:', output_dir)
    print('Train images:', len(train_rows))
    print('Test images:', len(test_rows))
    print('Manifest:', manifest_path)


if __name__ == '__main__':
    main()
