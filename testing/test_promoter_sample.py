#!/usr/bin/env python3
"""
Minimal test suite for Promoter sampling script.

Tests different conditioning scenarios:
- Random conditioning labels
- Provided expression targets
- Different target_dim values
- Unconditional sampling
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import torch
from omegaconf import OmegaConf

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from model_zoo.promoter.sample import PromoterSampler


class TestPromoterSampler(unittest.TestCase):
    """Test PromoterSampler class methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.sampler = PromoterSampler()
        self.device = self.sampler.device

    def test_get_sequence_length_default(self):
        """Test default sequence length."""
        config = OmegaConf.create({})
        seq_len = self.sampler.get_sequence_length(config)
        self.assertEqual(seq_len, 1024, "Default Promoter sequence length should be 1024")

    def test_get_sequence_length_from_config(self):
        """Test sequence length from config."""
        config = OmegaConf.create({
            'model': {'length': 512}
        })
        seq_len = self.sampler.get_sequence_length(config)
        self.assertEqual(seq_len, 512, "Should use config-specified sequence length")

    def test_generate_conditioning_labels_global(self):
        """Test conditioning label generation with global conditioning."""
        config = OmegaConf.create({
            'dataset': {
                'signal_dim': 1
            },
            'model': {
                'length': 1024,
                'use_global_conditioning': True
            }
        })

        num_samples = 10
        labels = self.sampler.generate_conditioning_labels(num_samples, config)

        # Check shape: (num_samples, signal_dim) for global conditioning
        self.assertEqual(labels.shape, (num_samples, 1),
                        f"Labels shape should be ({num_samples}, 1), got {labels.shape}")
        expected_device = torch.device(self.device) if isinstance(self.device, str) else self.device
        self.assertEqual(labels.device.type, expected_device.type,
                        "Labels should be on correct device")
        self.assertTrue(torch.is_tensor(labels), "Labels should be a torch tensor")

    def test_generate_conditioning_labels_per_position_default(self):
        """Test conditioning label generation with default per-position mode."""
        config = OmegaConf.create({
            'dataset': {
                'signal_dim': 1
            },
            'model': {'length': 1024}
        })

        num_samples = 5
        labels = self.sampler.generate_conditioning_labels(num_samples, config)

        # Should default to per-position conditioning with signal_dim=1
        self.assertEqual(labels.shape, (num_samples, 1024, 1),
                        f"Default labels shape should be ({num_samples}, 1024, 1), got {labels.shape}")

    def test_generate_conditioning_labels_per_position_multidim(self):
        """Test conditioning label generation with per-position multi-dimensional signals."""
        seq_length = 1024
        signal_dim = 4  # Multi-dimensional regulatory signal per position
        config = OmegaConf.create({
            'dataset': {
                'signal_dim': signal_dim
            },
            'model': {
                'length': seq_length,
                'use_global_conditioning': False
            }
        })

        num_samples = 8
        labels = self.sampler.generate_conditioning_labels(num_samples, config)

        # Check shape: (num_samples, seq_length, signal_dim) for per-position
        self.assertEqual(labels.shape, (num_samples, seq_length, signal_dim),
                        f"Per-position labels shape should be ({num_samples}, {seq_length}, {signal_dim}), got {labels.shape}")

    def test_generate_conditioning_labels_distribution(self):
        """Test that generated labels have reasonable distribution."""
        config = OmegaConf.create({
            'dataset': {
                'signal_dim': 1
            },
            'model': {
                'length': 1024,
                'use_global_conditioning': False
            }
        })

        num_samples = 1000
        labels = self.sampler.generate_conditioning_labels(num_samples, config)

        # Labels are generated as randn * 2.0, so should roughly have std ~2.0
        std = labels.std().item()
        mean = labels.mean().item()

        # Check roughly Gaussian with std ~2.0
        self.assertLess(abs(mean), 0.5, "Mean should be close to 0")
        self.assertGreater(std, 1.5, "Std should be roughly 2.0")
        self.assertLess(std, 2.5, "Std should be roughly 2.0")


class TestPromoterSamplingConditioning(unittest.TestCase):
    """Test different conditioning scenarios in the sampling pipeline."""

    @patch('model_zoo.promoter.sample.PromoterSampler.sample_and_save')
    @patch('model_zoo.promoter.sample.OmegaConf.load')
    def test_random_conditioning(self, mock_config_load, mock_sample_and_save):
        """Test sampling with random conditioning (default behavior)."""
        # Mock config
        mock_config = OmegaConf.create({
            'dataset': {
                'signal_dim': 1
            },
            'model': {
                'length': 1024,
                'use_global_conditioning': False
            }
        })
        mock_config_load.return_value = mock_config

        # Mock sample_and_save to capture arguments
        mock_sample_and_save.return_value = {
            'num_samples': 10,
            'output_path': 'test_output.npz'
        }

        # Simulate command-line args
        test_args = [
            'sample.py',
            '--checkpoint', 'test.ckpt',
            '--architecture', 'transformer',
            '--num_samples', '10',
            '--config', 'test_config.yaml'
        ]

        with patch('sys.argv', test_args):
            from model_zoo.promoter.sample import main
            main()

        # Verify sample_and_save was called
        mock_sample_and_save.assert_called_once()
        call_kwargs = mock_sample_and_save.call_args[1]

        # Check that conditioning_labels were generated
        self.assertIsNotNone(call_kwargs['conditioning_labels'],
                            "Should have conditioning labels for default random mode")
        self.assertEqual(call_kwargs['num_samples'], 10,
                        "Should use specified num_samples")

    @patch('model_zoo.promoter.sample.PromoterSampler.sample_and_save')
    @patch('model_zoo.promoter.sample.OmegaConf.load')
    def test_provided_expression_target(self, mock_config_load, mock_sample_and_save):
        """Test sampling with user-provided expression target."""
        mock_config = OmegaConf.create({
            'dataset': {
                'signal_dim': 1
            },
            'model': {
                'length': 1024,
                'use_global_conditioning': False
            }
        })
        mock_config_load.return_value = mock_config

        mock_sample_and_save.return_value = {'num_samples': 5}

        test_args = [
            'sample.py',
            '--checkpoint', 'test.ckpt',
            '--architecture', 'transformer',
            '--num_samples', '5',
            '--config', 'test_config.yaml',
            '--expression_target', '3.5'
        ]

        with patch('sys.argv', test_args):
            from model_zoo.promoter.sample import main
            main()

        call_kwargs = mock_sample_and_save.call_args[1]
        labels = call_kwargs['conditioning_labels']

        # Verify all labels are set to the target value
        self.assertIsNotNone(labels, "Should have conditioning labels")
        expected_value = 3.5
        # Check first sample's label value
        actual_value = labels[0, 0].item()
        self.assertAlmostEqual(actual_value, expected_value, places=5,
                              msg=f"Label should be set to {expected_value}")

    @patch('model_zoo.promoter.sample.PromoterSampler.sample_and_save')
    @patch('model_zoo.promoter.sample.OmegaConf.load')
    def test_unconditional_sampling(self, mock_config_load, mock_sample_and_save):
        """Test unconditional sampling (no labels)."""
        mock_config = OmegaConf.create({
            'model': {'length': 1024}
        })
        mock_config_load.return_value = mock_config

        mock_sample_and_save.return_value = {'num_samples': 3}

        test_args = [
            'sample.py',
            '--checkpoint', 'test.ckpt',
            '--architecture', 'transformer',
            '--num_samples', '3',
            '--config', 'test_config.yaml',
            '--unconditional'
        ]

        with patch('sys.argv', test_args):
            from model_zoo.promoter.sample import main
            main()

        call_kwargs = mock_sample_and_save.call_args[1]

        # Verify no conditioning labels were provided
        self.assertIsNone(call_kwargs['conditioning_labels'],
                         "Unconditional sampling should have None for conditioning_labels")

    @patch('model_zoo.promoter.data.PromoterDataset')
    @patch('model_zoo.promoter.sample.PromoterSampler.sample_and_save')
    @patch('model_zoo.promoter.sample.OmegaConf.load')
    def test_test_set_conditioning(self, mock_config_load, mock_sample_and_save, mock_dataset_class):
        """Test sampling with test set labels."""
        mock_config = OmegaConf.create({
            'model': {'length': 1024}
        })
        mock_config_load.return_value = mock_config

        # Mock dataset with fake labels
        mock_dataset = Mock()
        num_test_samples = 100
        # Create fake labels: (N, 1024, 1) as expected
        fake_labels = torch.randn(num_test_samples, 1024, 1)
        mock_dataset.y = fake_labels
        mock_dataset.__len__ = Mock(return_value=num_test_samples)
        mock_dataset_class.return_value = mock_dataset

        mock_sample_and_save.return_value = {'num_samples': num_test_samples}

        test_args = [
            'sample.py',
            '--checkpoint', 'test.ckpt',
            '--architecture', 'transformer',
            '--num_samples', '10',  # Should be overridden
            '--config', 'test_config.yaml',
            '--use_test_set',
            '--data_path', 'test_data.h5'
        ]

        with patch('sys.argv', test_args):
            from model_zoo.promoter.sample import main
            main()

        call_kwargs = mock_sample_and_save.call_args[1]

        # Verify test set labels were used
        self.assertIsNotNone(call_kwargs['conditioning_labels'],
                            "Should have conditioning labels from test set")
        # Verify num_samples was overridden to test set size
        self.assertEqual(call_kwargs['num_samples'], num_test_samples,
                        f"Should use test set size ({num_test_samples}) for num_samples")


class TestLabelShapes(unittest.TestCase):
    """Test that label shapes are correct for different signal_dim configurations."""

    def test_label_shape_consistency_global(self):
        """Test label shape consistency with global conditioning."""
        sampler = PromoterSampler()
        config = OmegaConf.create({
            'dataset': {
                'signal_dim': 1
            },
            'model': {
                'length': 1024,
                'use_global_conditioning': True
            }
        })

        for num_samples in [1, 5, 100]:
            with self.subTest(num_samples=num_samples):
                labels = sampler.generate_conditioning_labels(num_samples, config)
                self.assertEqual(labels.shape[0], num_samples,
                               f"First dimension should match num_samples={num_samples}")
                self.assertEqual(labels.shape[1], 1,
                               "Second dimension should be 1 for signal_dim=1 with global conditioning")

    def test_label_shape_consistency_per_position_multidim(self):
        """Test label shape consistency with per-position multi-dimensional signals."""
        sampler = PromoterSampler()

        for signal_dim in [2, 3, 5]:
            with self.subTest(signal_dim=signal_dim):
                config = OmegaConf.create({
                    'dataset': {
                        'signal_dim': signal_dim
                    },
                    'model': {
                        'length': 1024,
                        'use_global_conditioning': False
                    }
                })

                labels = sampler.generate_conditioning_labels(10, config)
                # Per-position targets: (num_samples, seq_length, signal_dim)
                self.assertEqual(labels.shape, (10, 1024, signal_dim),
                               f"Shape should be (10, 1024, {signal_dim}) for signal_dim={signal_dim}")


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestPromoterSampler))
    suite.addTests(loader.loadTestsFromTestCase(TestPromoterSamplingConditioning))
    suite.addTests(loader.loadTestsFromTestCase(TestLabelShapes))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
