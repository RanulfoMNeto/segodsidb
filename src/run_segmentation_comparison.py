"""
@brief Compare ODSI-DB segmentation JSON results from two U-Net runs.
"""

import argparse
import csv
import json
import math
import pathlib


DEFAULT_METRICS = [
    'accuracy',
    'balanced_accuracy',
    'sensitivity',
    'specificity',
]

ARTICLE_TISSUE_CLASSES = [
    'Skin',
    'Oral mucosa',
    'Enamel',
    'Tongue',
    'Lip',
    'Hard palate',
    'Attached gingiva',
    'Soft palate',
    'Hair',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare baseline U-Net and unmixing-assisted U-Net results.')
    parser.add_argument('--baseline-prefix', required=True, type=str,
                        help='Prefix used by src/test.py, e.g. results/fold0_simage_170.')
    parser.add_argument('--proposed-prefix', required=True, type=str,
                        help='Prefix used by src/test.py for the proposed run.')
    parser.add_argument('--output-prefix', required=True, type=str,
                        help='Prefix for comparison JSON/CSV files.')
    parser.add_argument('--baseline-name', default='unet_baseline', type=str)
    parser.add_argument('--proposed-name', default='unmixing_weak_unet',
                        type=str)
    parser.add_argument('--metrics', default=','.join(DEFAULT_METRICS),
                        type=str,
                        help='Comma-separated metric names to compare.')
    parser.add_argument('--baseline-config', default=None, type=str,
                        help='Optional baseline segmentation config.')
    parser.add_argument('--proposed-config', default=None, type=str,
                        help='Optional proposed segmentation config.')
    parser.add_argument('--baseline-checkpoint', default=None, type=str)
    parser.add_argument('--proposed-checkpoint', default=None, type=str)
    parser.add_argument('--class-set', default='all',
                        choices=['all', 'article_tissue', 'custom'],
                        help='Class subset used for global averages and per-class output.')
    parser.add_argument('--classes', default=None, type=str,
                        help='Comma-separated class names when --class-set custom is used.')
    parser.add_argument('--strict-config', action='store_true',
                        help='Fail if configs differ beyond the generated train data_dir/name/experiment.')
    return parser.parse_args()


def metric_path(prefix, metric):
    return pathlib.Path(str(prefix) + '_' + metric + '.json')


def read_metric_json(prefix, metric):
    path = metric_path(prefix, metric)
    if not path.is_file():
        raise FileNotFoundError('Metric JSON not found: {}'.format(path))
    with path.open() as f:
        return json.load(f)


def parse_metrics(value):
    metrics = [item.strip() for item in value.split(',') if item.strip()]
    if not metrics:
        raise ValueError('At least one metric must be provided.')
    return metrics


def parse_classes(value):
    if value is None:
        return []
    return [item.strip() for item in value.split(',') if item.strip()]


def resolve_class_filter(class_set, classes=None):
    if class_set == 'all':
        if classes:
            raise ValueError('--classes can only be used with --class-set custom.')
        return None
    if class_set == 'article_tissue':
        if classes:
            raise ValueError('--classes can only be used with --class-set custom.')
        return ARTICLE_TISSUE_CLASSES.copy()
    if class_set == 'custom':
        classes = parse_classes(classes)
        if not classes:
            raise ValueError('--class-set custom requires --classes.')
        if len(set(classes)) != len(classes):
            raise ValueError('Duplicate class names in --classes: {}'.format(
                classes))
        return classes
    raise ValueError('Unknown class set: {}'.format(class_set))


def finite_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def mean_finite(values):
    finite = [float(value) for value in values if value is not None
              and math.isfinite(float(value))]
    if not finite:
        return None
    return sum(finite) / len(finite)


def load_results(prefix, metrics):
    results = {}
    for metric in metrics:
        raw = read_metric_json(prefix, metric)
        results[metric] = {
            class_name: finite_float(value)
            for class_name, value in raw.items()
        }
    return results


def collect_class_names(*result_sets):
    names = set()
    for results in result_sets:
        for metric_values in results.values():
            names.update(metric_values.keys())
    return sorted(names)


def selected_class_names(baseline_results, proposed_results, class_filter=None):
    if class_filter is None:
        return collect_class_names(baseline_results, proposed_results)
    return list(class_filter)


def unknown_requested_classes(baseline_results, proposed_results, class_filter):
    if class_filter is None:
        return []
    available = set(collect_class_names(baseline_results, proposed_results))
    return [class_name for class_name in class_filter
            if class_name not in available]


def delta(proposed, baseline):
    if proposed is None or baseline is None:
        return None
    return proposed - baseline


def build_per_class_rows(baseline_results, proposed_results, metrics,
                         baseline_name, proposed_name, class_filter=None):
    rows = []
    for class_name in selected_class_names(
            baseline_results, proposed_results, class_filter):
        row = {'class_name': class_name}
        for metric in metrics:
            baseline = baseline_results.get(metric, {}).get(class_name)
            proposed = proposed_results.get(metric, {}).get(class_name)
            row['{}_{}'.format(baseline_name, metric)] = baseline
            row['{}_{}'.format(proposed_name, metric)] = proposed
            row['delta_{}'.format(metric)] = delta(proposed, baseline)
        rows.append(row)
    return rows


def build_global_rows(baseline_results, proposed_results, metrics,
                      baseline_name, proposed_name, class_filter=None):
    rows = []
    class_names = selected_class_names(
        baseline_results, proposed_results, class_filter)
    for metric in metrics:
        baseline_values = [
            baseline_results.get(metric, {}).get(class_name)
            for class_name in class_names
        ]
        proposed_values = [
            proposed_results.get(metric, {}).get(class_name)
            for class_name in class_names
        ]
        baseline = mean_finite(baseline_values)
        proposed = mean_finite(proposed_values)
        rows.append({
            'metric': metric,
            baseline_name: baseline,
            proposed_name: proposed,
            'delta': delta(proposed, baseline),
            'requested_classes': len(class_names),
            '{}_finite_classes'.format(baseline_name): len([
                value for value in baseline_values if value is not None
                and math.isfinite(float(value))
            ]),
            '{}_finite_classes'.format(proposed_name): len([
                value for value in proposed_values if value is not None
                and math.isfinite(float(value))
            ]),
        })
    return rows


def read_config(path):
    if path is None:
        return None
    with pathlib.Path(path).open() as f:
        return json.load(f)


def comparable_config_differences(baseline_config, proposed_config):
    ignored = {
        ('name',),
        ('experiment',),
        ('data_loader', 'args', 'data_dir'),
    }
    differences = []

    def walk(left, right, path=()):
        if path in ignored:
            return
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left.keys()) | set(right.keys())):
                walk(left.get(key), right.get(key), path + (key,))
            return
        if left != right:
            differences.append({
                'path': '.'.join(path),
                'baseline': left,
                'proposed': right,
            })

    walk(baseline_config, proposed_config)
    return differences


def write_csv(path, rows, fieldnames):
    with pathlib.Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    metrics = parse_metrics(args.metrics)
    baseline_results = load_results(args.baseline_prefix, metrics)
    proposed_results = load_results(args.proposed_prefix, metrics)
    class_filter = resolve_class_filter(args.class_set, args.classes)
    unknown_classes = unknown_requested_classes(
        baseline_results, proposed_results, class_filter)
    if unknown_classes:
        raise RuntimeError('Requested classes were not found in result JSONs: '
                           '{}'.format(unknown_classes))

    per_class_rows = build_per_class_rows(
        baseline_results, proposed_results, metrics,
        args.baseline_name, args.proposed_name, class_filter=class_filter)
    global_rows = build_global_rows(
        baseline_results, proposed_results, metrics,
        args.baseline_name, args.proposed_name, class_filter=class_filter)

    baseline_config = read_config(args.baseline_config)
    proposed_config = read_config(args.proposed_config)
    config_differences = []
    if baseline_config is not None and proposed_config is not None:
        config_differences = comparable_config_differences(
            baseline_config, proposed_config)
        if args.strict_config and config_differences:
            raise RuntimeError(
                'Configs are not directly comparable: {}'.format(
                    config_differences))

    output_prefix = pathlib.Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    per_class_path = pathlib.Path(str(output_prefix) + '_per_class.csv')
    global_path = pathlib.Path(str(output_prefix) + '_global.csv')
    json_path = pathlib.Path(str(output_prefix) + '.json')

    per_class_fields = ['class_name']
    for metric in metrics:
        per_class_fields.extend([
            '{}_{}'.format(args.baseline_name, metric),
            '{}_{}'.format(args.proposed_name, metric),
            'delta_{}'.format(metric),
        ])
    write_csv(per_class_path, per_class_rows, per_class_fields)
    write_csv(global_path, global_rows, [
        'metric', args.baseline_name, args.proposed_name, 'delta',
        'requested_classes',
        '{}_finite_classes'.format(args.baseline_name),
        '{}_finite_classes'.format(args.proposed_name)])

    summary = {
        'baseline': {
            'name': args.baseline_name,
            'prefix': args.baseline_prefix,
            'config': args.baseline_config,
            'checkpoint': args.baseline_checkpoint,
        },
        'proposed': {
            'name': args.proposed_name,
            'prefix': args.proposed_prefix,
            'config': args.proposed_config,
            'checkpoint': args.proposed_checkpoint,
        },
        'metrics': metrics,
        'class_set': args.class_set,
        'classes': selected_class_names(
            baseline_results, proposed_results, class_filter),
        'article_tissue_classes': ARTICLE_TISSUE_CLASSES,
        'config_differences_ignoring_expected_paths': config_differences,
        'global': global_rows,
        'per_class': per_class_rows,
        'files': {
            'global_csv': str(global_path),
            'per_class_csv': str(per_class_path),
        },
    }
    with json_path.open('w') as f:
        json.dump(summary, f, indent=4)
    print('Wrote comparison JSON: {}'.format(json_path))
    print('Wrote global CSV: {}'.format(global_path))
    print('Wrote per-class CSV: {}'.format(per_class_path))


if __name__ == '__main__':
    main()
