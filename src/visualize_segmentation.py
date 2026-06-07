"""
@brief Export visualisations for trained ODSI-DB segmentation checkpoints.
"""

import argparse
import csv
import json
import pathlib

import cv2
import numpy as np
import torch

import torchseg.data_loader
import torchseg.model
import torchseg.utils


PALETTE = np.round(np.array([
    [0.0, 0.0, 0.0],
    [0.73910129, 0.54796227, 0.70659469],
    [0.07401779, 0.48485457, 0.2241555],
    [0.35201324, 0.9025658, 0.81062183],
    [0.08126211, 0.23986311, 0.54880697],
    [0.33267484, 0.6119932, 0.30272535],
    [0.45419585, 0.1818727, 0.5877175],
    [0.1239585, 0.17862775, 0.6892662],
    [0.20556493, 0.44462774, 0.38364081],
    [0.18754881, 0.1831789, 0.00863592],
    [0.37702173, 0.075744, 0.07170247],
    [0.9487006, 0.90159635, 0.26639963],
    [0.8954375, 0.58731839, 0.87918311],
    [0.83980577, 0.77131811, 0.02192928],
    [0.47681103, 0.72962211, 0.96439166],
    [0.44293943, 0.60166042, 0.5879358],
    [0.52419707, 0.18690438, 0.69027514],
    [0.34720014, 0.57450984, 0.96570434],
    [0.78380941, 0.2237716, 0.52199938],
    [0.98170786, 0.61735585, 0.73834123],
    [0.44000012, 0.06259595, 0.76726459],
    [0.47754739, 0.13137904, 0.04615173],
    [0.65486219, 0.24028978, 0.75424866],
    [0.79301129, 0.75970907, 0.06562084],
    [0.14864707, 0.55623561, 0.80328385],
    [0.54439947, 0.234355, 0.81248573],
    [0.24443958, 0.00697174, 0.59921356],
    [0.76808718, 0.56387681, 0.52199431],
    [0.69855907, 0.73646473, 0.8320837],
    [0.85436454, 0.86456808, 0.61494475],
    [0.34944949, 0.79188401, 0.8251793],
    [0.43554137, 0.18054355, 0.80210866],
    [0.76501493, 0.38795293, 0.49637574],
    [0.31552006, 0.3704537, 0.90083695],
    [0.26176471, 0.66781917, 0.65375891],
    [0.21141543, 0.16505171, 0.53799316],
]) * 255.0).astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize ODSI-DB segmentation predictions.')
    parser.add_argument('-r', '--resume', required=True, type=str,
                        help='Segmentation checkpoint, usually model_best.pth.')
    parser.add_argument('-c', '--conf', default=None, type=str,
                        help='Optional config override; defaults to checkpoint config.json.')
    parser.add_argument('--data-dir', default=None, type=str,
                        help='Directory with image/mask pairs. Defaults to test data_dir from config.')
    parser.add_argument('--image-path', default=None, type=str,
                        help='Specific HSI image to visualize.')
    parser.add_argument('--mask-path', default=None, type=str,
                        help='Optional mask path. Inferred from image path when omitted.')
    parser.add_argument('--basename', default=None, type=str,
                        help='Image basename to select inside --data-dir.')
    parser.add_argument('--index', default=0, type=int,
                        help='Image index inside --data-dir when --image-path/--basename are omitted.')
    parser.add_argument('--mode', default=None, type=str,
                        help='Input mode. Defaults to config testing/data_loader mode.')
    parser.add_argument('--output-dir', default=None, type=str,
                        help='Output directory. Defaults to results/segmentation_vis/<run>/<image>.')
    parser.add_argument('--tile-size', default=0, type=int,
                        help='Optional tiled inference size. 0 uses full image.')
    parser.add_argument('--overlay-alpha', default=0.55, type=float,
                        help='Prediction opacity in overlay images.')
    parser.add_argument('--legend-classes', default='present',
                        choices=['present', 'all', 'none'],
                        help='Classes shown in the shared legend.')
    parser.add_argument('--device', default=None, type=str,
                        help='Device string, e.g. cuda, cuda:0, or cpu.')
    return parser.parse_args()


def load_config(args):
    resume = pathlib.Path(args.resume)
    config_path = pathlib.Path(args.conf) if args.conf else resume.parent / 'config.json'
    if not config_path.is_file():
        raise FileNotFoundError('Config file not found: {}'.format(config_path))
    with config_path.open() as f:
        return json.load(f), config_path


def config_test_data_dir(config):
    datasets = config.get('testing', {}).get('datasets', [])
    if datasets:
        return datasets[0].get('args', {}).get('data_dir')
    return config.get('data_loader', {}).get('args', {}).get('data_dir')


def config_mode(config):
    datasets = config.get('testing', {}).get('datasets', [])
    if datasets:
        mode = datasets[0].get('args', {}).get('mode')
        if mode:
            return mode
    return config.get('data_loader', {}).get('args', {}).get('mode')


def image_mask_pairs(data_dir):
    data_dir = pathlib.Path(data_dir)
    mask_paths = sorted(data_dir.glob('*_masks.tif'))
    pairs = []
    for mask_path in mask_paths:
        image_path = mask_path.with_name(mask_path.name.replace('_masks.tif',
                                                               '.tif'))
        if image_path.is_file():
            pairs.append((image_path, mask_path))
    return pairs


def select_image(args, config):
    if args.image_path:
        image_path = pathlib.Path(args.image_path)
        mask_path = pathlib.Path(args.mask_path) if args.mask_path else None
        if mask_path is None:
            inferred = torchseg.data_loader.OdsiDbDataLoader.LoadImage \
                .infer_label_path(image_path)
            mask_path = pathlib.Path(inferred) if inferred else None
        return image_path, mask_path

    data_dir = args.data_dir or config_test_data_dir(config)
    if data_dir is None:
        raise ValueError('Provide --data-dir or --image-path.')

    pairs = image_mask_pairs(data_dir)
    if not pairs:
        raise RuntimeError('No image/mask pairs found in {}'.format(data_dir))

    if args.basename:
        for image_path, mask_path in pairs:
            if image_path.name == args.basename:
                return image_path, mask_path
        raise RuntimeError('Image basename not found in {}: {}'.format(
            data_dir, args.basename))

    if args.index < 0 or args.index >= len(pairs):
        raise IndexError('--index {} outside available range 0..{}'.format(
            args.index, len(pairs) - 1))
    return pairs[args.index]


def build_output_dir(args, config, config_path, image_path):
    if args.output_dir:
        output_dir = pathlib.Path(args.output_dir)
    else:
        run_id = config_path.parent.name
        experiment = config.get('name', 'segmentation')
        output_dir = pathlib.Path('results') / 'segmentation_vis' / \
            '{}_{}'.format(experiment, run_id) / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_model(config, checkpoint_path, device):
    model_cfg = config['model']
    model = getattr(torchseg.model, model_cfg['type'])(**model_cfg['args'])
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device,
                                weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint['state_dict']
    if state_dict and next(iter(state_dict)).startswith('module.'):
        state_dict = {
            key.replace('module.', '', 1): value
            for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, checkpoint


def predict_full(model, image_tensor, device):
    with torch.no_grad():
        output = model(image_tensor.to(device))
    return output.detach().cpu()


def predict_tiled(model, image_tensor, device, tile_size):
    _, _, height, width = image_tensor.shape
    out_channels = model.upconv1[-1].out_channels \
        if hasattr(model, 'upconv1') else None
    if out_channels is None:
        return predict_full(model, image_tensor, device)

    output = torch.empty((1, out_channels, height, width),
                         dtype=torch.float32)
    with torch.no_grad():
        for row in range(0, height, tile_size):
            row_end = min(row + tile_size, height)
            for col in range(0, width, tile_size):
                col_end = min(col + tile_size, width)
                patch = image_tensor[:, :, row:row_end, col:col_end]
                pred = model(patch.to(device)).detach().cpu()
                output[:, :, row:row_end, col:col_end] = pred
    return output


def read_rgb_preview(image_path):
    im_hyper, wl, rgb, _ = torchseg.data_loader.read_stiff(
        str(image_path), silent=True, rgb_only=False)
    if rgb is None:
        rgb = torchseg.data_loader.OdsiDbDataLoader.LoadImage.hyper2rgb(
            im_hyper, wl)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    return rgb[:, :, :3]


def class_map_to_rgb(class_map, valid_mask=None):
    colour_indices = class_map.astype(np.int64) + 1
    rgb = PALETTE[colour_indices]
    if valid_mask is not None:
        rgb = rgb.copy()
        rgb[~valid_mask] = 0
    return rgb.astype(np.uint8)


def label_to_class_map(label):
    valid = np.sum(label, axis=0) == 1
    class_map = np.argmax(label, axis=0).astype(np.int64)
    return class_map, valid


def overlay(rgb, label_rgb, mask=None, alpha=0.55):
    mixed = rgb.astype(np.float32).copy()
    label_rgb = label_rgb.astype(np.float32)
    if mask is None:
        mixed = (1.0 - alpha) * mixed + alpha * label_rgb
    else:
        mixed[mask] = (1.0 - alpha) * mixed[mask] + alpha * label_rgb[mask]
    return np.clip(mixed, 0, 255).astype(np.uint8)


def write_rgb(path, rgb):
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def render_legend(rows, width):
    if not rows:
        return None

    margin = 12
    title_h = 34
    row_h = 27
    swatch = 18
    min_col_w = 300
    ncols = max(1, int(width // min_col_w))
    ncols = min(ncols, max(1, len(rows)))
    rows_per_col = int(np.ceil(float(len(rows)) / ncols))
    col_w = int(width // ncols)
    height = title_h + rows_per_col * row_h + margin
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    cv2.putText(canvas, 'Shared legend', (margin, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2,
                cv2.LINE_AA)

    for idx, row in enumerate(rows):
        col = idx // rows_per_col
        line = idx % rows_per_col
        x = margin + col * col_w
        y = title_h + line * row_h + 4
        color = tuple(int(c) for c in row['color_rgb'])
        cv2.rectangle(canvas, (x, y), (x + swatch, y + swatch),
                      color, thickness=-1)
        cv2.rectangle(canvas, (x, y), (x + swatch, y + swatch),
                      (40, 40, 40), thickness=1)
        prefix = ''
        if row.get('class_index') not in [None, '']:
            prefix = '{}: '.format(row['class_index'])
        cv2.putText(canvas, prefix + row['class_name'],
                    (x + swatch + 8, y + swatch - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1,
                    cv2.LINE_AA)

    return canvas


def side_by_side(images, titles, legend_rows=None):
    height = max(image.shape[0] for image in images)
    width = sum(image.shape[1] for image in images)
    title_h = 34
    canvas = np.full((height + title_h, width, 3), 255, dtype=np.uint8)
    offset = 0
    for image, title in zip(images, titles):
        h, w = image.shape[:2]
        canvas[title_h:title_h + h, offset:offset + w] = image
        cv2.putText(canvas, title, (offset + 8, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2,
                    cv2.LINE_AA)
        offset += w
    legend = render_legend(legend_rows, width)
    if legend is not None:
        separator_h = 1
        out = np.full((canvas.shape[0] + separator_h + legend.shape[0],
                       width, 3), 255, dtype=np.uint8)
        out[:canvas.shape[0], :, :] = canvas
        out[canvas.shape[0]:canvas.shape[0] + separator_h, :, :] = 210
        out[canvas.shape[0] + separator_h:, :, :] = legend
        return out
    return canvas


def counts_by_class(class_map, valid_mask=None):
    if valid_mask is None:
        values, counts = np.unique(class_map, return_counts=True)
    else:
        values, counts = np.unique(class_map[valid_mask], return_counts=True)
    idx2class = torchseg.data_loader.OdsiDbDataLoader.OdsiDbDataset.classnames
    rows = []
    for class_idx, pixels in zip(values.tolist(), counts.tolist()):
        rows.append({
            'class_index': int(class_idx),
            'class_name': idx2class[int(class_idx)],
            'pixels': int(pixels),
            'color_rgb': PALETTE[int(class_idx) + 1].tolist(),
        })
    return rows


def write_counts_csv(path, rows):
    fieldnames = ['class_index', 'class_name', 'pixels', 'color_rgb']
    with pathlib.Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def shared_legend_rows(pred_summary, gt_summary, mode):
    if mode == 'none':
        return []

    idx2class = torchseg.data_loader.OdsiDbDataLoader.OdsiDbDataset.classnames
    if mode == 'all':
        rows = [{
            'class_index': int(class_idx),
            'class_name': class_name,
            'color_rgb': PALETTE[int(class_idx) + 1].tolist(),
        } for class_idx, class_name in idx2class.items()]
    else:
        by_class = {}
        for row in pred_summary + gt_summary:
            class_idx = int(row['class_index'])
            by_class[class_idx] = {
                'class_index': class_idx,
                'class_name': row['class_name'],
                'color_rgb': row['color_rgb'],
            }
        rows = [by_class[class_idx] for class_idx in sorted(by_class)]

    if gt_summary:
        rows.insert(0, {
            'class_index': '',
            'class_name': 'Ignored / unannotated',
            'color_rgb': [0, 0, 0],
        })
    return rows


def main():
    args = parse_args()
    config, config_path = load_config(args)
    mode = args.mode or config_mode(config)
    if mode is None:
        raise ValueError('Input mode is missing; pass --mode.')

    image_path, mask_path = select_image(args, config)
    output_dir = build_output_dir(args, config, config_path, image_path)
    device = torch.device(args.device or (
        'cuda' if torch.cuda.is_available() else 'cpu'))

    model, checkpoint = load_model(config, args.resume, device)
    image = torchseg.data_loader.OdsiDbDataLoader.LoadImage.read_image(
        str(image_path), mode)
    image_tensor = torch.from_numpy(image).unsqueeze(0)

    if args.tile_size and args.tile_size > 0:
        output = predict_tiled(model, image_tensor, device, args.tile_size)
    else:
        output = predict_full(model, image_tensor, device)

    prediction = output.argmax(dim=1)[0].numpy().astype(np.int64)
    confidence = torch.exp(output).max(dim=1)[0][0].numpy()
    rgb = read_rgb_preview(image_path)
    pred_rgb = class_map_to_rgb(prediction)

    write_rgb(output_dir / 'input_rgb.png', rgb)
    write_rgb(output_dir / 'prediction_full.png', pred_rgb)
    write_rgb(output_dir / 'overlay_full.png',
              overlay(rgb, pred_rgb, alpha=args.overlay_alpha))
    cv2.imwrite(str(output_dir / 'confidence.png'),
                np.clip(confidence * 255.0, 0, 255).astype(np.uint8))

    pred_summary = counts_by_class(prediction)
    write_counts_csv(output_dir / 'prediction_counts.csv', pred_summary)

    panels = [rgb, pred_rgb, overlay(rgb, pred_rgb, alpha=args.overlay_alpha)]
    titles = ['Input RGB', 'Prediction', 'Overlay']

    gt_summary = []
    annotated_pixels = None
    if mask_path is not None and pathlib.Path(mask_path).is_file():
        label = torchseg.data_loader.OdsiDbDataLoader.LoadImage.read_label(
            str(mask_path))
        gt_class_map, gt_valid = label_to_class_map(label)
        gt_rgb = class_map_to_rgb(gt_class_map, gt_valid)
        pred_annotated_rgb = class_map_to_rgb(prediction, gt_valid)
        write_rgb(output_dir / 'ground_truth.png', gt_rgb)
        write_rgb(output_dir / 'prediction_annotated.png', pred_annotated_rgb)
        write_rgb(output_dir / 'overlay_annotated.png',
                  overlay(rgb, pred_rgb, mask=gt_valid,
                          alpha=args.overlay_alpha))
        panels = [rgb, gt_rgb, pred_annotated_rgb,
                  overlay(rgb, pred_rgb, mask=gt_valid,
                          alpha=args.overlay_alpha)]
        titles = ['Input RGB', 'Ground truth', 'Prediction annotated',
                  'Overlay annotated']
        gt_summary = counts_by_class(gt_class_map, gt_valid)
        write_counts_csv(output_dir / 'ground_truth_counts.csv', gt_summary)
        annotated_pixels = int(np.count_nonzero(gt_valid))

    legend_rows = shared_legend_rows(pred_summary, gt_summary,
                                     args.legend_classes)
    legend = render_legend(
        legend_rows, sum(image.shape[1] for image in panels))
    if legend is not None:
        write_rgb(output_dir / 'legend.png', legend)
    write_rgb(output_dir / 'side_by_side.png',
              side_by_side(panels, titles, legend_rows))

    summary = {
        'checkpoint': str(args.resume),
        'checkpoint_epoch': checkpoint.get('epoch'),
        'config': str(config_path),
        'model': config.get('model'),
        'mode': mode,
        'image_path': str(image_path),
        'mask_path': str(mask_path) if mask_path else None,
        'output_dir': str(output_dir),
        'device': str(device),
        'tile_size': int(args.tile_size),
        'image_shape_chw': list(image.shape),
        'annotated_pixels': annotated_pixels,
        'prediction_counts': pred_summary,
        'ground_truth_counts': gt_summary,
        'legend_classes': args.legend_classes,
        'legend': legend_rows,
        'files': {
            'input_rgb': str(output_dir / 'input_rgb.png'),
            'prediction_full': str(output_dir / 'prediction_full.png'),
            'overlay_full': str(output_dir / 'overlay_full.png'),
            'confidence': str(output_dir / 'confidence.png'),
            'legend': str(output_dir / 'legend.png') if legend is not None
            else None,
            'side_by_side': str(output_dir / 'side_by_side.png'),
        },
    }
    with (output_dir / 'summary.json').open('w') as f:
        json.dump(summary, f, indent=4)

    print('Saved segmentation visualizations to {}'.format(output_dir))
    print('Side-by-side: {}'.format(output_dir / 'side_by_side.png'))


if __name__ == '__main__':
    main()
