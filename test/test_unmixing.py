"""
@brief Unit tests for CNNAEU unmixing integration.
"""

import logging
import pathlib
import tempfile
import unittest

import numpy as np
import torch

import torchseg.data_loader
import torchseg.machine
import torchseg.model
import torchseg.test_unmixing
import torchseg.train
import torchseg.utils
import torchseg.visualize_unmixing


class TestCNNAEU(unittest.TestCase):

    def test_model_shapes_and_abundance_constraints(self):
        model = torchseg.model.CNNAEU(in_channels=170, num_endmembers=10,
                                      dropout=0.0)
        x = torch.randn(2, 170, 40, 40)
        output = model(x)

        self.assertEqual(output['reconstruction'].shape, (2, 170, 40, 40))
        self.assertEqual(output['abundances'].shape, (2, 10, 40, 40))
        self.assertEqual(output['endmembers'].shape, (10, 170))
        self.assertTrue(torch.all(output['abundances'] >= 0))
        self.assertTrue(torch.allclose(
            output['abundances'].sum(dim=1),
            torch.ones(2, 40, 40),
            atol=1e-5))

    def test_decoder_clamp_after_optimizer_step(self):
        model = torchseg.model.CNNAEU(in_channels=8, num_endmembers=3,
                                      encoder_filters=4,
                                      decoder_kernel_size=3, dropout=0.0)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        x = torch.randn(2, 8, 8, 8)
        target = torch.rand(2, 8, 8, 8)

        output = model(x)
        loss = torchseg.model.sad_reconstruction_loss(output, target)
        loss.backward()
        optimizer.step()
        model.decoder.weight.data -= 10.0
        model.clamp_decoder_weights()

        self.assertTrue(torch.all(model.decoder.weight >= 0))

    def test_decoder_uses_spatial_kernel_average_scale(self):
        model = torchseg.model.CNNAEU(in_channels=1, num_endmembers=1,
                                      encoder_filters=1,
                                      decoder_kernel_size=3, dropout=0.0)
        model.decoder.weight.data.fill_(1.0)
        abundances = torch.ones(1, 1, 5, 5)

        reconstruction = model.decode_abundances(abundances)

        self.assertAlmostEqual(reconstruction[0, 0, 2, 2].item(), 1.0)


class TestUnmixingLossesAndMetrics(unittest.TestCase):

    def test_relevant_unmixing_classes_include_prosthetics(self):
        class2idx = {
            v: k
            for k, v in torchseg.data_loader.OdsiDbDataLoader
            .OdsiDbDataset.classnames.items()
        }

        self.assertIn(
            'Prosthetics',
            torchseg.model.ODSI_DB_UNMIXING_RELEVANT_CLASS_NAMES)
        self.assertIn(
            'Prosthetics',
            torchseg.visualize_unmixing.DEFAULT_RELEVANT_CLASSES)

        labels = torch.zeros(1, 35, 1, 1)
        _, class_indices = torchseg.model.odsi_db_unmixing_label_subset(
            labels, n_endmembers=10)

        self.assertIn(class2idx['Prosthetics'], class_indices.tolist())
        self.assertEqual(
            len(torchseg.model.ODSI_DB_UNMIXING_RELEVANT_CLASS_NAMES),
            len(class_indices))

    def test_relevant_classes_are_independent_from_num_endmembers(self):
        default_classes = \
            torchseg.model.ODSI_DB_UNMIXING_RELEVANT_CLASS_NAMES.copy()
        class2idx = {
            v: k
            for k, v in torchseg.data_loader.OdsiDbDataLoader
            .OdsiDbDataset.classnames.items()
        }

        try:
            active = torchseg.model \
                .set_odsi_db_unmixing_relevant_class_names(
                    ['Enamel', 'Prosthetics'])
            labels = torch.zeros(1, 35, 1, 1)

            _, class_indices = torchseg.model.odsi_db_unmixing_label_subset(
                labels, n_endmembers=10)

            self.assertEqual(active, ['Enamel', 'Prosthetics'])
            self.assertEqual(class_indices.tolist(), [
                class2idx['Enamel'],
                class2idx['Prosthetics'],
            ])
        finally:
            torchseg.model.set_odsi_db_unmixing_relevant_class_names(
                default_classes)

    def test_unmixing_reference_classes_can_be_configured_from_config(self):
        default_classes = \
            torchseg.model.ODSI_DB_UNMIXING_RELEVANT_CLASS_NAMES.copy()
        try:
            active = torchseg.model.configure_odsi_db_unmixing_from_config({
                'unmixing': {
                    'reference_class_names': ['Enamel', 'Prosthetics']
                }
            })

            self.assertEqual(active, ['Enamel', 'Prosthetics'])
            self.assertEqual(
                torchseg.model.ODSI_DB_UNMIXING_RELEVANT_CLASS_NAMES,
                ['Enamel', 'Prosthetics'])
        finally:
            torchseg.model.set_odsi_db_unmixing_relevant_class_names(
                default_classes)

    def test_sad_loss_is_near_zero_for_identical_inputs(self):
        x = torch.rand(2, 5, 4, 4)
        loss = torchseg.model.sad_reconstruction_loss(x, x)
        self.assertLess(loss.item(), 1e-4)

    def test_sad_loss_is_finite_for_zero_inputs(self):
        pred = torch.zeros(2, 5, 4, 4)
        gt = torch.zeros(2, 5, 4, 4)
        loss = torchseg.model.sad_reconstruction_loss(pred, gt)
        self.assertTrue(torch.isfinite(loss))

    def test_sad_mse_loss_penalizes_reflectance_scale(self):
        pred = torch.ones(1, 1, 2, 2) * 2.0
        gt = torch.ones(1, 1, 2, 2)

        loss = torchseg.model.sad_mse_reconstruction_loss(
            {'reconstruction': pred}, gt, mse_weight=0.5)

        self.assertAlmostEqual(loss.item(), 0.5, places=5)

    def test_configured_loss_accepts_arguments(self):
        pred = torch.ones(1, 1, 2, 2) * 2.0
        gt = torch.ones(1, 1, 2, 2)
        loss_fn = torchseg.model.loss.get_loss_function({
            'type': 'sad_mse_reconstruction_loss',
            'args': {
                'mse_weight': 0.25,
            },
        })

        loss = loss_fn(pred, gt)

        self.assertEqual(loss_fn.__name__, 'sad_mse_reconstruction_loss')
        self.assertAlmostEqual(loss.item(), 0.25, places=5)

    def test_hungarian_balanced_accuracy_requires_spectral_mapping(self):
        abundances = torch.tensor([[
            [[0.1, 0.1], [0.9, 0.9]],
            [[0.9, 0.9], [0.1, 0.1]],
        ]])
        labels = torch.tensor([[
            [[1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 1.0]],
        ]])
        output = {'abundances': abundances, 'label': labels}

        score = torchseg.model.odsi_db_unmixing_hungarian_balanced_accuracy(
            output)

        self.assertAlmostEqual(score, 0.0)

    def test_reference_endmembers_are_class_mean_reflectance(self):
        target = torch.tensor([[
            [[1.0, 0.0]],
            [[0.0, 2.0]],
        ]])
        labels = torch.tensor([[
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ]])

        refs, counts, class_indices = \
            torchseg.model.odsi_db_unmixing_reference_endmembers(
                target, labels, n_endmembers=2)

        self.assertTrue(torch.allclose(refs, torch.tensor([
            [1.0, 0.0],
            [0.0, 2.0],
        ])))
        self.assertTrue(torch.allclose(counts, torch.tensor([1.0, 1.0])))
        self.assertEqual(class_indices.tolist(), [0, 1])

    def test_spectral_hungarian_mapping_matches_by_sad(self):
        estimated = torch.tensor([
            [0.0, 1.0],
            [1.0, 0.0],
        ])
        reference = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        class_indices = torch.tensor([5, 26])

        mapping = torchseg.model \
            .odsi_db_unmixing_hungarian_mapping_from_reference_endmembers(
                estimated, reference, class_indices)

        self.assertEqual(mapping.tolist(), [26, 5])

    def test_reference_endmember_csv_loader_accepts_exported_layout(self):
        reference = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        wavelengths = np.array([450.0, 500.0, 550.0], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / 'endmembers.csv'
            table = np.column_stack([wavelengths, reference.T])
            np.savetxt(
                path, table, delimiter=',',
                header='wavelength_nm,endmember_0,endmember_1',
                comments='')

            loaded, resolved = torchseg.test_unmixing.load_reference_endmembers(
                path, expected_bands=3)

        self.assertEqual(resolved, str(path))
        self.assertTrue(np.allclose(loaded, reference))

    def test_reference_hungarian_mapping_matches_external_endmembers(self):
        model = torchseg.model.CNNAEU(
            in_channels=3, num_endmembers=2, encoder_filters=2,
            decoder_kernel_size=1, dropout=0.0)
        with torch.no_grad():
            model.decoder.weight.zero_()
            model.decoder.weight[:, 0, 0, 0] = torch.tensor([0.0, 1.0, 0.0])
            model.decoder.weight[:, 1, 0, 0] = torch.tensor([1.0, 0.0, 0.0])

        reference = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = pathlib.Path(tmpdir) / 'refs.npy'
            np.save(ref_path, reference)

            mapping, info = \
                torchseg.test_unmixing.learn_reference_spectral_mapping(
                    model, torch.device('cpu'), ref_path,
                    reference_class_indices='5,26')

        self.assertEqual(mapping.tolist(), [26, 5])
        self.assertEqual(info['source'], 'reference')
        self.assertEqual(info['class_indices'], [5, 26])
        self.assertEqual(info['endmember_to_class'][0]['class_name'],
                         'Prosthetics')

    def test_reference_mapping_uses_generic_names_without_metadata(self):
        model = torchseg.model.CNNAEU(
            in_channels=3, num_endmembers=2, encoder_filters=2,
            decoder_kernel_size=1, dropout=0.0)
        with torch.no_grad():
            model.decoder.weight.zero_()
            model.decoder.weight[:, 0, 0, 0] = torch.tensor([1.0, 0.0, 0.0])
            model.decoder.weight[:, 1, 0, 0] = torch.tensor([0.0, 1.0, 0.0])

        reference = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = pathlib.Path(tmpdir) / 'refs.npy'
            np.save(ref_path, reference)

            _, info = torchseg.test_unmixing.learn_reference_spectral_mapping(
                model, torch.device('cpu'), ref_path)

        self.assertEqual(info['class_indices'], [0, 1])
        self.assertEqual(info['class_names'], ['reference_0', 'reference_1'])

    def test_mask_mapping_uses_configured_relevant_classes(self):
        default_classes = \
            torchseg.model.ODSI_DB_UNMIXING_RELEVANT_CLASS_NAMES.copy()
        original_read_stiff = torchseg.data_loader.read_stiff
        original_read_reflectance = \
            torchseg.data_loader.OdsiDbDataLoader.LoadImage \
            .read_hyper_reflectance
        original_read_label = \
            torchseg.data_loader.OdsiDbDataLoader.LoadImage.read_label

        model = torchseg.model.CNNAEU(
            in_channels=3, num_endmembers=2, encoder_filters=2,
            decoder_kernel_size=1, dropout=0.0)
        with torch.no_grad():
            model.decoder.weight.zero_()
            model.decoder.weight[:, 0, 0, 0] = torch.tensor([0.0, 1.0, 0.0])
            model.decoder.weight[:, 1, 0, 0] = torch.tensor([1.0, 0.0, 0.0])

        target = np.array([[[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0]]], dtype=np.float32)
        labels = np.zeros((35, 1, 2), dtype=np.float32)
        labels[5, 0, 0] = 1.0
        labels[26, 0, 1] = 1.0

        def fake_read_stiff(filename, silent=False, rgb_only=False):
            return target.copy(), np.array([450.0, 500.0, 550.0]), None, ''

        def fake_read_reflectance(path, mode, im_hyper=None, wl=None):
            return target.copy()

        def fake_read_label(path):
            return labels.copy()

        torchseg.data_loader.read_stiff = fake_read_stiff
        torchseg.data_loader.OdsiDbDataLoader.LoadImage \
            .read_hyper_reflectance = staticmethod(fake_read_reflectance)
        torchseg.data_loader.OdsiDbDataLoader.LoadImage.read_label = \
            staticmethod(fake_read_label)

        try:
            torchseg.model.set_odsi_db_unmixing_relevant_class_names(
                ['Enamel', 'Prosthetics'])
            with tempfile.TemporaryDirectory() as tmpdir:
                image_path = pathlib.Path(tmpdir) / 'image.tif'
                mask_path = pathlib.Path(tmpdir) / 'image_masks.tif'
                image_path.touch()
                mask_path.touch()

                mapping, info = \
                    torchseg.test_unmixing.learn_mask_spectral_mapping(
                        model, torch.device('cpu'), image_path, 'simage_170')
        finally:
            torchseg.model.set_odsi_db_unmixing_relevant_class_names(
                default_classes)
            torchseg.data_loader.read_stiff = original_read_stiff
            torchseg.data_loader.OdsiDbDataLoader.LoadImage \
                .read_hyper_reflectance = original_read_reflectance
            torchseg.data_loader.OdsiDbDataLoader.LoadImage.read_label = \
                original_read_label

        self.assertEqual(mapping.tolist(), [26, 5])
        self.assertEqual(info['source'], 'mask')
        self.assertEqual(info['class_indices'], [5, 26])
        self.assertEqual(info['class_names'], ['Enamel', 'Prosthetics'])

    def test_mask_mapping_keeps_absent_relevant_classes_eligible(self):
        default_classes = \
            torchseg.model.ODSI_DB_UNMIXING_RELEVANT_CLASS_NAMES.copy()
        original_read_stiff = torchseg.data_loader.read_stiff
        original_read_reflectance = \
            torchseg.data_loader.OdsiDbDataLoader.LoadImage \
            .read_hyper_reflectance
        original_read_label = \
            torchseg.data_loader.OdsiDbDataLoader.LoadImage.read_label

        model = torchseg.model.CNNAEU(
            in_channels=3, num_endmembers=2, encoder_filters=2,
            decoder_kernel_size=1, dropout=0.0)
        with torch.no_grad():
            model.decoder.weight.zero_()
            model.decoder.weight[:, 0, 0, 0] = torch.tensor([1.0, 0.0, 0.0])
            model.decoder.weight[:, 1, 0, 0] = torch.tensor([0.0, 1.0, 0.0])

        target = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
        labels = np.zeros((35, 1, 1), dtype=np.float32)
        labels[5, 0, 0] = 1.0

        def fake_read_stiff(filename, silent=False, rgb_only=False):
            return target.copy(), np.array([450.0, 500.0, 550.0]), None, ''

        def fake_read_reflectance(path, mode, im_hyper=None, wl=None):
            return target.copy()

        def fake_read_label(path):
            return labels.copy()

        torchseg.data_loader.read_stiff = fake_read_stiff
        torchseg.data_loader.OdsiDbDataLoader.LoadImage \
            .read_hyper_reflectance = staticmethod(fake_read_reflectance)
        torchseg.data_loader.OdsiDbDataLoader.LoadImage.read_label = \
            staticmethod(fake_read_label)

        try:
            torchseg.model.set_odsi_db_unmixing_relevant_class_names(
                ['Enamel', 'Prosthetics'])
            with tempfile.TemporaryDirectory() as tmpdir:
                image_path = pathlib.Path(tmpdir) / 'image.tif'
                mask_path = pathlib.Path(tmpdir) / 'image_masks.tif'
                image_path.touch()
                mask_path.touch()

                mapping, info = \
                    torchseg.test_unmixing.learn_mask_spectral_mapping(
                        model, torch.device('cpu'), image_path, 'simage_170')
        finally:
            torchseg.model.set_odsi_db_unmixing_relevant_class_names(
                default_classes)
            torchseg.data_loader.read_stiff = original_read_stiff
            torchseg.data_loader.OdsiDbDataLoader.LoadImage \
                .read_hyper_reflectance = original_read_reflectance
            torchseg.data_loader.OdsiDbDataLoader.LoadImage.read_label = \
                original_read_label

        self.assertEqual(mapping.tolist(), [5, 26])
        self.assertEqual(info['class_indices'], [5, 26])
        self.assertEqual(info['reference_pixel_counts'], [1.0, 0.0])
        self.assertEqual(info['endmember_to_class'][1]['class_name'],
                         'Prosthetics')

    def test_present_only_mapping_scope_filters_absent_classes(self):
        estimated = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        reference = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        class_indices = torch.tensor([5, 26])
        pixel_counts = torch.tensor([1.0, 0.0])

        reference_valid = torchseg.test_unmixing \
            .reference_valid_for_mapping_scope(pixel_counts, 'present-only')
        mapping = torchseg.model \
            .odsi_db_unmixing_hungarian_mapping_from_reference_endmembers(
                estimated, reference, class_indices,
                reference_valid=reference_valid, device='cpu')

        self.assertEqual(mapping.tolist(), [5, -1])

    def test_present_label_class_info_reports_known_classes(self):
        original_read_label = \
            torchseg.data_loader.OdsiDbDataLoader.LoadImage.read_label
        labels = np.zeros((35, 2, 2), dtype=np.float32)
        labels[5, 0, 0] = 1.0
        labels[26, 1, 1] = 1.0

        def fake_read_label(path):
            return labels.copy()

        torchseg.data_loader.OdsiDbDataLoader.LoadImage.read_label = \
            staticmethod(fake_read_label)
        try:
            info = torchseg.data_loader.OdsiDbDataLoader.LoadImage \
                .present_label_class_info('dummy_masks.tif')
        finally:
            torchseg.data_loader.OdsiDbDataLoader.LoadImage.read_label = \
                original_read_label

        self.assertEqual(
            [(item['class_index'], item['class_name'], item['pixels'])
             for item in info],
            [(5, 'Enamel', 1), (26, 'Prosthetics', 1)])

    def test_train_auto_config_sets_r_and_reference_classes(self):
        original_read_label = \
            torchseg.data_loader.OdsiDbDataLoader.LoadImage.read_label
        labels = np.zeros((35, 2, 2), dtype=np.float32)
        labels[5, 0, :] = 1.0
        labels[26, 1, :] = 1.0

        def fake_read_label(path):
            return labels.copy()

        class FakeConfig:
            def __init__(self, save_dir, image_path):
                self.save_dir = pathlib.Path(save_dir)
                self.config = {
                    'name': 'fake',
                    'model': {
                        'type': 'CNNAEU',
                        'args': {
                            'num_endmembers': 99,
                        },
                    },
                    'sweep': {
                        'num_endmembers': [9, 10],
                    },
                    'data_loader': {
                        'type': 'OdsiDbSingleImageUnmixingDataLoader',
                        'args': {
                            'image_path': str(image_path),
                        },
                    },
                    'unmixing': {
                        'auto_from_training_mask': True,
                    },
                }

        torchseg.data_loader.OdsiDbDataLoader.LoadImage.read_label = \
            staticmethod(fake_read_label)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                image_path = pathlib.Path(tmpdir) / 'image.tif'
                mask_path = pathlib.Path(tmpdir) / 'image_masks.tif'
                image_path.touch()
                mask_path.touch()
                cfg = FakeConfig(tmpdir, image_path)

                info = torchseg.train.auto_configure_single_image_unmixing(
                    cfg)

                saved = torchseg.utils.read_json(
                    pathlib.Path(tmpdir) / 'config.json')
        finally:
            torchseg.data_loader.OdsiDbDataLoader.LoadImage.read_label = \
                original_read_label

        self.assertEqual(info['num_endmembers'], 2)
        self.assertEqual(
            cfg.config['model']['args']['num_endmembers'], 2)
        self.assertEqual(
            cfg.config['unmixing']['reference_class_names'],
            ['Enamel', 'Prosthetics'])
        self.assertNotIn('sweep', cfg.config)
        self.assertEqual(saved['model']['args']['num_endmembers'], 2)
        self.assertNotIn('sweep', saved)

    def test_hungarian_balanced_accuracy_uses_frozen_semantic_mapping(self):
        abundances = torch.tensor([[
            [[0.1, 0.1], [0.9, 0.9]],
            [[0.9, 0.9], [0.1, 0.1]],
        ]])
        labels = torch.tensor([[
            [[1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 1.0]],
        ]])
        output = {
            'abundances': abundances,
            'label': labels,
            'semantic_mapping': torch.tensor([1, 0]),
        }

        score = torchseg.model.odsi_db_unmixing_hungarian_balanced_accuracy(
            output)

        self.assertAlmostEqual(score, 1.0)

    def test_reconstruction_and_target_range_metrics(self):
        output = {
            'reconstruction': torch.tensor([[[[0.1, 0.3]]]])
        }
        target = torch.tensor([[[[0.2, 0.4]]]])

        self.assertAlmostEqual(torchseg.model.reconstruction_min(output), 0.1)
        self.assertAlmostEqual(torchseg.model.reconstruction_max(output), 0.3)
        self.assertAlmostEqual(torchseg.model.reconstruction_mean(output), 0.2)
        self.assertAlmostEqual(torchseg.model.target_reflectance_min(output, target),
                               0.2)
        self.assertAlmostEqual(torchseg.model.target_reflectance_max(output, target),
                               0.4)
        self.assertAlmostEqual(torchseg.model.target_reflectance_mean(output, target),
                               0.3)


class TestUnmixingVisualizationHelpers(unittest.TestCase):

    def test_apply_mapping_to_dominant_preserves_unmapped_endmembers(self):
        dominant = np.array([[0, 1], [2, 1]], dtype=np.int16)
        mapping = np.array([11, -1, 2], dtype=np.int64)

        mapped = torchseg.visualize_unmixing.apply_mapping_to_dominant(
            dominant, mapping)

        expected = np.array([[11, -1], [2, -1]], dtype=np.int16)
        self.assertTrue(np.array_equal(mapped, expected))

    def test_save_class_index_map_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / 'mapped_class.png'
            class_map = np.array([[2, 11], [-1, 34]], dtype=np.int16)

            torchseg.visualize_unmixing.save_class_index_map(
                class_map, path, 'Mapped class')

            self.assertTrue(path.is_file())

    def test_filter_class_index_map_keeps_only_selected_classes(self):
        class_map = np.array([[5, 26], [11, -1]], dtype=np.int16)

        selected = torchseg.visualize_unmixing.filter_class_index_map(
            class_map, selected_class_indices=[5, 26])

        expected = np.array([[5, 26], [-2, -1]], dtype=np.int16)
        self.assertTrue(np.array_equal(selected, expected))

    def test_class_index_summary_uses_class_and_negative_names(self):
        class_map = np.array([[5, 26], [-2, -1]], dtype=np.int16)

        summary = torchseg.visualize_unmixing.class_index_summary(
            class_map,
            negative_labels={-2: 'other mapped class', -1: 'unmapped'})

        self.assertEqual(summary['Enamel'], 1)
        self.assertEqual(summary['Prosthetics'], 1)
        self.assertEqual(summary['other mapped class'], 1)
        self.assertEqual(summary['unmapped'], 1)

    def test_selected_mapping_coverage_reports_classes_outside_hungarian(self):
        coverage = torchseg.visualize_unmixing.selected_mapping_coverage(
            selected_class_indices=[5, 26],
            mapping_class_indices=np.array([5, 21]),
            mapping=np.array([21, 5]))

        self.assertEqual(coverage[0]['class_name'], 'Enamel')
        self.assertTrue(coverage[0]['in_hungarian_competition'])
        self.assertEqual(coverage[0]['assigned_endmembers'], [1])
        self.assertEqual(coverage[1]['class_name'], 'Prosthetics')
        self.assertFalse(coverage[1]['in_hungarian_competition'])
        self.assertEqual(coverage[1]['assigned_endmembers'], [])

    def test_mapping_coverage_class_indices_drop_empty_references(self):
        class_indices = torchseg.visualize_unmixing \
            .mapping_coverage_class_indices(
                np.array([5, 26]), reference_counts=np.array([10.0, 0.0]))

        self.assertEqual(class_indices.tolist(), [5])

    def test_abundance_grid_titles_include_spectral_pairing(self):
        titles = torchseg.visualize_unmixing.abundance_grid_titles(
            mapping=np.array([26, 5, -1]),
            n_endmembers=3)

        self.assertEqual(titles[0], 'E0 ->\nProsthetics')
        self.assertEqual(titles[1], 'E1 ->\nEnamel')
        self.assertEqual(titles[2], 'E2 ->\nunmapped')

    def test_dataset_class_presence_counts_pixels_and_images(self):
        class FakeDataset:
            data = [{'label': 'a.png'}, {'label': 'b.png'}]

        labels = {}
        labels['a.png'] = np.zeros((35, 2, 2), dtype=np.float32)
        labels['a.png'][5, 0, 0] = 1.0
        labels['a.png'][26, 0, 1] = 1.0
        labels['b.png'] = np.zeros((35, 2, 2), dtype=np.float32)
        labels['b.png'][5, :, :] = 1.0

        original_read_label = \
            torchseg.data_loader.OdsiDbDataLoader.LoadImage.read_label

        def fake_read_label(path):
            return labels[path]

        torchseg.data_loader.OdsiDbDataLoader.LoadImage.read_label = \
            fake_read_label
        try:
            presence = torchseg.visualize_unmixing.dataset_class_presence(
                FakeDataset(), [5, 26])
        finally:
            torchseg.data_loader.OdsiDbDataLoader.LoadImage.read_label = \
                original_read_label

        self.assertEqual(presence[0]['class_name'], 'Enamel')
        self.assertEqual(presence[0]['images'], 2)
        self.assertEqual(presence[0]['pixels'], 5)
        self.assertEqual(presence[1]['class_name'], 'Prosthetics')
        self.assertEqual(presence[1]['images'], 1)
        self.assertEqual(presence[1]['pixels'], 1)


class TestOdsiDbUnmixingDataLoader(unittest.TestCase):

    def test_image_patch_sampler_groups_indices_by_image(self):
        sampler = torchseg.data_loader.OdsiDbUnmixingDataLoader.ImagePatchSampler(
            [2, 0], num_patches=3, shuffle=False)

        self.assertEqual(list(iter(sampler)), [6, 7, 8, 0, 1, 2])
        self.assertEqual(len(sampler), 6)

    def test_hyper_image_pair_matches_simage_170_preprocessing(self):
        raw = np.linspace(0.0, 1.0, 5 * 6 * 51, dtype=np.float32)
        raw = raw.reshape(5, 6, 51)
        wavelengths = np.linspace(450, 950, 51)
        original_read_stiff = torchseg.data_loader.read_stiff

        def fake_read_stiff(filename, silent=False, rgb_only=False):
            return raw.copy(), wavelengths.copy(), None, ''

        torchseg.data_loader.read_stiff = fake_read_stiff
        try:
            image = torchseg.data_loader.OdsiDbDataLoader.LoadImage.read_image(
                'dummy.tif', 'simage_170')
            pair_image, target = \
                torchseg.data_loader.OdsiDbDataLoader.LoadImage.read_hyper_image_pair(
                    'dummy.tif', 'simage_170')
        finally:
            torchseg.data_loader.read_stiff = original_read_stiff

        self.assertTrue(np.allclose(image, pair_image))
        self.assertEqual(target.shape, (170, 5, 6))
        self.assertGreaterEqual(target.min(), 0.0)
        self.assertLessEqual(target.max(), 1.0)

    def test_single_image_loader_requires_existing_file(self):
        with self.assertRaises(FileNotFoundError):
            torchseg.data_loader.OdsiDbSingleImageUnmixingDataLoader(
                '/tmp/does_not_exist_odsi_single_image.tif',
                batch_size=1,
                mode='simage_51')

    def test_single_image_loader_returns_fixed_unlabelled_patches(self):
        raw = np.linspace(0.0, 1.0, 8 * 9 * 51, dtype=np.float32)
        raw = raw.reshape(8, 9, 51)
        wavelengths = np.linspace(450, 950, 51)
        original_read_stiff = torchseg.data_loader.read_stiff

        def fake_read_stiff(filename, silent=False, rgb_only=False):
            return raw.copy(), wavelengths.copy(), None, ''

        torchseg.data_loader.read_stiff = fake_read_stiff
        try:
            with tempfile.NamedTemporaryFile(suffix='.tif') as tmp:
                loader = \
                    torchseg.data_loader \
                    .OdsiDbSingleImageUnmixingDataLoader(
                        tmp.name,
                        batch_size=2,
                        mode='simage_51',
                        patch_size=4,
                        num_patches=6,
                        validation_split=0.0,
                        seed=7,
                        num_workers=0)
                item = loader.dataset[0]
        finally:
            torchseg.data_loader.read_stiff = original_read_stiff

        self.assertEqual(item['image'].shape, torch.Size([51, 4, 4]))
        self.assertEqual(item['target_reflectance'].shape,
                         torch.Size([51, 4, 4]))
        self.assertNotIn('label', item)
        self.assertEqual(item['patch_size'], 4)
        self.assertGreaterEqual(item['target_reflectance'].min().item(), 0.0)
        self.assertLessEqual(item['target_reflectance'].max().item(), 1.0)

    def test_single_image_loader_splits_patch_indices(self):
        raw = np.linspace(0.0, 1.0, 8 * 9 * 51, dtype=np.float32)
        raw = raw.reshape(8, 9, 51)
        wavelengths = np.linspace(450, 950, 51)
        original_read_stiff = torchseg.data_loader.read_stiff

        def fake_read_stiff(filename, silent=False, rgb_only=False):
            return raw.copy(), wavelengths.copy(), None, ''

        torchseg.data_loader.read_stiff = fake_read_stiff
        try:
            with tempfile.NamedTemporaryFile(suffix='.tif') as tmp:
                loader = \
                    torchseg.data_loader \
                    .OdsiDbSingleImageUnmixingDataLoader(
                        tmp.name,
                        batch_size=2,
                        mode='simage_51',
                        patch_size=4,
                        num_patches=10,
                        validation_split=0.2,
                        seed=11,
                        num_workers=0)
                valid_loader = loader.split_validation()
                train_indices = set(loader.sampler.indices)
                valid_indices = set(valid_loader.sampler.indices)
        finally:
            torchseg.data_loader.read_stiff = original_read_stiff

        self.assertEqual(len(train_indices), 8)
        self.assertEqual(len(valid_indices), 2)
        self.assertFalse(train_indices.intersection(valid_indices))


class TestUnmixingMachine(unittest.TestCase):

    class FakeConfig(dict):
        def __init__(self, root):
            super().__init__({
                'machine': {
                    'args': {
                        'epochs': 1,
                        'save_period': 1,
                        'verbosity': 0,
                        'monitor': 'off',
                        'tensorboard': False,
                    }
                }
            })
            self.save_dir = pathlib.Path(root)
            self.log_dir = pathlib.Path(root)
            self.resume = None

        def get_logger(self, name, verbosity=2):
            logger = logging.getLogger(name)
            logger.addHandler(logging.NullHandler())
            return logger

    class SyntheticDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            image = torch.randn(4, 8, 8)
            target = torch.rand(4, 8, 8)
            label = torch.zeros(2, 8, 8)
            return {
                'image': image,
                'target_reflectance': target,
                'label': label,
            }

    class SyntheticDatasetWithoutLabels(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            image = torch.randn(4, 8, 8)
            target = torch.rand(4, 8, 8)
            return {
                'image': image,
                'target_reflectance': target,
            }

    def test_train_epoch_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = torch.utils.data.DataLoader(
                TestUnmixingMachine.SyntheticDataset(), batch_size=2)
            model = torchseg.model.CNNAEU(in_channels=4, num_endmembers=2,
                                          encoder_filters=3,
                                          decoder_kernel_size=3, dropout=0.0)
            optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
            machine = torchseg.machine.UnmixingMachine(
                model,
                torchseg.model.sad_reconstruction_loss,
                [torchseg.model.asc_error],
                optimizer,
                config=TestUnmixingMachine.FakeConfig(tmpdir),
                device=torch.device('cpu'),
                data_loader=loader,
                valid_data_loader=None)

            log = machine._train_epoch(1)

        self.assertIn('loss', log)
        self.assertIn('asc_error', log)
        self.assertTrue(np.isfinite(log['loss']))

    def test_train_epoch_smoke_without_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = torch.utils.data.DataLoader(
                TestUnmixingMachine.SyntheticDatasetWithoutLabels(),
                batch_size=2)
            model = torchseg.model.CNNAEU(in_channels=4, num_endmembers=2,
                                          encoder_filters=3,
                                          decoder_kernel_size=3, dropout=0.0)
            optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
            machine = torchseg.machine.UnmixingMachine(
                model,
                torchseg.model.sad_reconstruction_loss,
                [torchseg.model.asc_error],
                optimizer,
                config=TestUnmixingMachine.FakeConfig(tmpdir),
                device=torch.device('cpu'),
                data_loader=loader,
                valid_data_loader=None)

            log = machine._train_epoch(1)

        self.assertIn('loss', log)
        self.assertIn('asc_error', log)
        self.assertTrue(np.isfinite(log['loss']))


if __name__ == '__main__':
    unittest.main()
