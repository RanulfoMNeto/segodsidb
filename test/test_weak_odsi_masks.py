"""
@brief Unit tests for weak ODSI-DB mask expansion.
"""

import pathlib
import tempfile
import unittest

import numpy as np

import torchseg.data_loader
try:
    import torchseg.generate_weak_odsi_masks as weak_masks
except ModuleNotFoundError:
    import generate_weak_odsi_masks as weak_masks

try:
    import torchseg.run_segmentation_comparison as comparison
except ModuleNotFoundError:
    import run_segmentation_comparison as comparison

try:
    import torchseg.unmixing_manifest as unmixing_manifest
except ModuleNotFoundError:
    import unmixing_manifest


class TestWeakOdsiMaskExpansion(unittest.TestCase):

    def _thresholds(self):
        return {
            'max_abundance': 0.85,
            'min_abundance_margin': 0.20,
            'max_reference_sad': 0.15,
            'max_reconstruction_sad': 0.10,
        }

    def test_original_labels_are_preserved_and_rejected_pixels_stay_ignored(self):
        label = np.zeros((3, 2, 4), dtype=np.float32)
        label[1, 0, 0] = 1
        label[2, 0, 1] = 1

        target = np.zeros((2, 2, 4), dtype=np.float32)
        target[:, 0, 0] = [1.0, 0.0]
        target[:, 0, 1] = [0.0, 1.0]
        target[:, 1, 0] = [1.0, 0.0]
        target[:, 1, 1] = [0.0, 1.0]
        target[:, 1, 2] = [1.0, 0.0]

        references, _ = weak_masks.compute_reference_spectra(target, label)
        abundances = np.zeros((2, 2, 4), dtype=np.float32)
        abundances[0, :, :] = 0.5
        abundances[1, :, :] = 0.5
        abundances[:, 1, 0] = [0.90, 0.10]
        abundances[:, 1, 1] = [0.10, 0.90]
        abundances[:, 1, 2] = [0.84, 0.16]
        reconstruction_sad = np.zeros((2, 4), dtype=np.float32)

        expanded, stats = weak_masks.build_expanded_label(
            label, abundances, np.array([1, 2]), target, references,
            reconstruction_sad, self._thresholds(), max_pseudo_ratio=5.0)

        self.assertEqual(expanded[1, 0, 0], 1)
        self.assertEqual(expanded[2, 0, 1], 1)
        self.assertEqual(expanded[1, 1, 0], 1)
        self.assertEqual(expanded[2, 1, 1], 1)
        self.assertEqual(expanded[:, 1, 2].sum(), 0)
        self.assertEqual(stats['total_accepted_pixels'], 2)

    def test_each_confidence_threshold_rejects_pixels(self):
        label = np.zeros((2, 1, 6), dtype=np.float32)
        label[1, 0, 0] = 1

        target = np.zeros((2, 1, 6), dtype=np.float32)
        target[:, 0, 0] = [1.0, 0.0]
        target[:, 0, 1] = [1.0, 0.0]
        target[:, 0, 2] = [1.0, 0.0]
        target[:, 0, 3] = [1.0, 0.0]
        target[:, 0, 4] = [0.0, 1.0]
        target[:, 0, 5] = [1.0, 0.0]

        references, _ = weak_masks.compute_reference_spectra(target, label)
        abundances = np.zeros((2, 1, 6), dtype=np.float32)
        abundances[:, 0, 1] = [0.90, 0.10]
        abundances[:, 0, 2] = [0.84, 0.16]
        abundances[:, 0, 3] = [0.85, 0.66]
        abundances[:, 0, 4] = [0.95, 0.05]
        abundances[:, 0, 5] = [0.95, 0.05]
        reconstruction_sad = np.zeros((1, 6), dtype=np.float32)
        reconstruction_sad[0, 5] = 0.11

        expanded, stats = weak_masks.build_expanded_label(
            label, abundances, np.array([1, -1]), target, references,
            reconstruction_sad, self._thresholds(), max_pseudo_ratio=10.0)

        self.assertEqual(expanded[1, 0, 1], 1)
        self.assertEqual(expanded[:, 0, 2].sum(), 0)
        self.assertEqual(expanded[:, 0, 3].sum(), 0)
        self.assertEqual(expanded[:, 0, 4].sum(), 0)
        self.assertEqual(expanded[:, 0, 5].sum(), 0)
        self.assertEqual(stats['total_accepted_pixels'], 1)

    def test_per_class_pseudo_label_cap_is_applied_by_confidence(self):
        label = np.zeros((2, 1, 6), dtype=np.float32)
        label[1, 0, 0] = 1
        target = np.zeros((2, 1, 6), dtype=np.float32)
        target[:, :, :] = np.array([1.0, 0.0], dtype=np.float32)[:, None, None]
        references, _ = weak_masks.compute_reference_spectra(target, label)

        abundances = np.zeros((2, 1, 6), dtype=np.float32)
        abundances[:, 0, 1] = [0.90, 0.10]
        abundances[:, 0, 2] = [0.99, 0.01]
        abundances[:, 0, 3] = [0.91, 0.09]
        abundances[:, 0, 4] = [0.98, 0.02]
        abundances[:, 0, 5] = [0.97, 0.03]
        reconstruction_sad = np.zeros((1, 6), dtype=np.float32)

        expanded, stats = weak_masks.build_expanded_label(
            label, abundances, np.array([1, -1]), target, references,
            reconstruction_sad, self._thresholds(), max_pseudo_ratio=2.0)

        self.assertEqual(stats['candidate_counts'][1], 5)
        self.assertEqual(stats['accepted_counts'][1], 2)
        self.assertEqual(expanded[1].sum(), 3)

    def test_mask_write_and_read_preserves_expanded_labels(self):
        label = np.zeros((35, 2, 2), dtype=np.float32)
        class2idx = torchseg.data_loader.OdsiDbDataLoader \
            .OdsiDbDataset.classnames_reverse
        label[class2idx['Enamel'], 0, 0] = 1
        label[class2idx['Skin'], 1, 1] = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / 'sample_masks.tif'
            torchseg.data_loader.write_mtiff(
                str(path), weak_masks.label_to_mask_dict(label))
            loaded = torchseg.data_loader.OdsiDbDataLoader \
                .LoadImage.read_label(str(path))

        self.assertEqual(loaded[class2idx['Enamel'], 0, 0], 1)
        self.assertEqual(loaded[class2idx['Skin'], 1, 1], 1)
        self.assertEqual(loaded[:, 0, 1].sum(), 0)


class TestSegmentationComparison(unittest.TestCase):

    def test_per_class_and_global_deltas(self):
        baseline = {
            'balanced_accuracy': {'A': 0.5, 'B': 0.75},
            'accuracy': {'A': 0.8, 'B': 0.9},
        }
        proposed = {
            'balanced_accuracy': {'A': 0.6, 'B': 0.70},
            'accuracy': {'A': 0.85, 'B': 0.95},
        }

        rows = comparison.build_per_class_rows(
            baseline, proposed, ['balanced_accuracy', 'accuracy'],
            'baseline', 'proposed')
        global_rows = comparison.build_global_rows(
            baseline, proposed, ['balanced_accuracy', 'accuracy'],
            'baseline', 'proposed')

        row_a = [row for row in rows if row['class_name'] == 'A'][0]
        self.assertAlmostEqual(row_a['delta_balanced_accuracy'], 0.1)
        self.assertAlmostEqual(global_rows[0]['baseline'], 0.625)
        self.assertAlmostEqual(global_rows[0]['delta'], 0.025)

    def test_article_class_filter_limits_global_average(self):
        baseline = {
            'balanced_accuracy': {
                'Skin': 0.5,
                'Enamel': 0.7,
                'Rare class': 0.1,
            },
        }
        proposed = {
            'balanced_accuracy': {
                'Skin': 0.6,
                'Enamel': 0.8,
                'Rare class': 0.9,
            },
        }
        class_filter = ['Skin', 'Enamel']

        rows = comparison.build_global_rows(
            baseline, proposed, ['balanced_accuracy'],
            'baseline', 'proposed', class_filter=class_filter)
        per_class = comparison.build_per_class_rows(
            baseline, proposed, ['balanced_accuracy'],
            'baseline', 'proposed', class_filter=class_filter)

        self.assertAlmostEqual(rows[0]['baseline'], 0.6)
        self.assertAlmostEqual(rows[0]['proposed'], 0.7)
        self.assertEqual(rows[0]['requested_classes'], 2)
        self.assertEqual([row['class_name'] for row in per_class],
                         class_filter)

    def test_article_class_set_is_available(self):
        class_filter = comparison.resolve_class_filter(
            'article_tissue', None)
        self.assertEqual(class_filter, comparison.ARTICLE_TISSUE_CLASSES)

    def test_custom_class_set_requires_explicit_classes(self):
        with self.assertRaises(ValueError):
            comparison.resolve_class_filter('custom', None)


class TestUnmixingManifest(unittest.TestCase):

    def _entry(self, image_name, score, run_dir):
        return {
            'image_path': '../odsi_db/folds/fold_0/train/{}'.format(
                image_name),
            'image_basename': image_name,
            'checkpoint': '{}/model_best.pth'.format(run_dir),
            'config': '{}/config.json'.format(run_dir),
            'run_dir': run_dir,
            'mode': 'simage_170',
            'monitor': 'min val_loss',
            'monitor_mode': 'min',
            'monitor_metric': 'val_loss',
            'monitor_best': score,
            'epoch': 10,
            'updated_at': '2026-06-06T00:00:00',
        }

    def test_manifest_keeps_best_single_image_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / 'fold0_unmixing_manifest.json'
            first = self._entry('sample.tif', 0.4, 'saved/run_a')
            worse = self._entry('sample.tif', 0.5, 'saved/run_b')
            better = self._entry('sample.tif', 0.3, 'saved/run_c')

            created = unmixing_manifest.update_manifest(path, first)
            rejected = unmixing_manifest.update_manifest(path, worse)
            updated = unmixing_manifest.update_manifest(path, better)
            manifest = unmixing_manifest.read_manifest(path)

        self.assertTrue(created['updated'])
        self.assertFalse(rejected['updated'])
        self.assertTrue(updated['updated'])
        self.assertEqual(len(manifest['items']), 1)
        self.assertEqual(manifest['items'][0]['run_dir'], 'saved/run_c')
        self.assertEqual(manifest['items'][0]['monitor_best'], 0.3)

    def test_manifest_path_is_inferred_from_fold_name(self):
        path = unmixing_manifest.infer_manifest_path(
            '../odsi_db/folds/fold_0/train/sample.tif')
        self.assertEqual(str(path), 'results/fold0_unmixing_manifest.json')

    def test_manifest_paths_are_relative_to_checkout(self):
        path = unmixing_manifest.portable_path(
            '/workspace/odsi_db/folds/fold_0/train/sample.tif',
            base_dir='/workspace/segodsidb')
        self.assertEqual(path, '../odsi_db/folds/fold_0/train/sample.tif')


if __name__ == '__main__':
    unittest.main()
