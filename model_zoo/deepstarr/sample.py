#!/usr/bin/env python3
"""
DeepSTARR Sampling Script. Inherits from base sampling framework while using DeepSTARR-specific models directly.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from typing import Optional
import numpy as np
import h5py

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import base framework and DeepSTARR-specific components
from scripts.sample import BaseSampler, parse_base_args, main_sample
from model_zoo.deepstarr.data import get_deepstarr_datasets
from model_zoo.deepstarr.deepstarr import PL_DeepSTARR
from utils.inpainting import create_inpainting_manager


class DeepSTARRSampler(BaseSampler):
    """DeepSTARR-specific sampler that inherits from base framework."""
    
    def __init__(self):
        super().__init__("DeepSTARR")
    
    def load_model(self, checkpoint_path: str, config: OmegaConf, architecture: str = 'transformer'):
        from model_zoo.deepstarr.models import load_trained_model

        return load_trained_model(checkpoint_path, config, architecture, self.device)
    
    def get_sequence_length(self, config: OmegaConf) -> int:
        return 249  # DeepSTARR fixed sequence length
    
    def generate_conditioning_labels(self, num_samples: int, config: OmegaConf) -> torch.Tensor:
        # DeepSTARR has 2 activities: Dev and HK enhancer activities
        labels = torch.randn(num_samples, 2, device=self.device)
        return labels
    def create_dataloader(self, config: OmegaConf, split: str = 'test', batch_size: Optional[int] = None):
        train_ds, val_ds, test_ds = get_deepstarr_datasets(config.paths.data_file)

        if split == 'train':
            dataset = train_ds
        elif split == 'val':
            dataset = val_ds
        elif split == 'test':
            dataset = test_ds
        else:
            raise ValueError(f"Unknown split: {split}")

        if batch_size is None:
            batch_size = getattr(config, 'batch_size', 32)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )


def load_default_config():
    config_file = Path(__file__).parent / 'configs' / 'transformer.yaml'
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    return OmegaConf.load(config_file)


def main():

    parser = parse_base_args()
    # Add DeepSTARR-specific conditioning arguments
    parser.add_argument('--dev_activity', type=float, help='Dev enhancer activity value (if not provided, uses random)')
    parser.add_argument('--hk_activity', type=float, help='HK enhancer activity value (if not provided, uses random)')
    parser.add_argument('--unconditional', action='store_true', help='Sample unconditionally (ignoring any labels)')
    args = parser.parse_args()

    config, _ = BaseSampler.load_config_with_fallback(
        args.config, Path(__file__).parent, 'transformer.yaml'
    )
    sampler = DeepSTARRSampler()

    # Create inpainting manager if inpainting mode is specified
    inpainting_mgr = create_inpainting_manager(
        mode=args.inpainting_mode if args.inpainting_mode != 'none' else None,
        data_file=args.inpainting_data,
        device=sampler.device,
        seed=args.inpainting_seed,
        expected_signal_dim=config.dataset.signal_dim  # 2 for DeepSTARR
    )

    if args.save_rep:
        print(f"Saving representation of the model to {args.data_path}")
        results = sampler.save_representation(
            checkpoint_path=args.checkpoint,
            config=config,
            split=args.split,
            save_rep_timestamp=args.save_rep_timestamp,
            batch_size=args.batch_size,
            architecture=args.architecture,
            output_path=args.output,
            format=args.format
        )

        print(f"\n{sampler.dataset_name} Representation Results:")
        print("=" * 40)
        for key, value in results.items():
            print(f"{key}: {value}")

        print(f"\n✓ {sampler.dataset_name} saving representation completed successfully!")
        sys.exit(0)

    # Determine num_samples and conditioning labels based on inpainting mode
    if inpainting_mgr:
        num_samples_to_generate = inpainting_mgr.num_samples
        all_labels = inpainting_mgr.Y_target  # (658, 2) - use real DeepSTARR values
        print(f"Inpainting mode: {args.inpainting_mode}")
        print(f"Generating {num_samples_to_generate} sequences with inpainting constraints")
        print(f"Using Y_target values from inpainting data file")
    else:
        num_samples_to_generate = args.num_samples
        # Use regular conditioning logic
        if not args.unconditional:
            if args.dev_activity is not None and args.hk_activity is not None:
                all_labels = torch.tensor([[args.dev_activity, args.hk_activity]], device=sampler.device).expand(num_samples_to_generate, -1)
                print(f"Using specified activities: Dev={args.dev_activity}, HK={args.hk_activity}")
            else:
                all_labels = sampler.generate_conditioning_labels(num_samples_to_generate, config)
                print("Using random activities")
        else:
            all_labels = None
            print("Sampling unconditionally (no conditioning labels)")

    # Setup wandb if enabled
    if args.use_wandb:
        sampler.setup_wandb(args, config)

    steps = args.steps
    if steps is None:
        steps = sampler.get_sequence_length(config)

    print(f"Loading DeepSTARR {args.architecture} model from {args.checkpoint}")

    # TODO: reduce this code, can just do an if else to def the proj_fun and batch_labels

    # For inpainting, we need to batch and create proj_fun for each batch
    if inpainting_mgr:
        batch_size = args.batch_size or 256
        num_iterations = args.inpainting_iterations

        # Store sequences per iteration: list of [iteration][sequences]
        iterations_sequences = []

        print(f"Processing {num_samples_to_generate} samples in batches of {batch_size}")
        print(f"Running {num_iterations} iteration(s) per sample")

        for iteration in range(num_iterations):
            print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")
            iteration_sequences = []

            for start_idx in range(0, num_samples_to_generate, batch_size):
                end_idx = min(start_idx + batch_size, num_samples_to_generate)
                current_batch_size = end_idx - start_idx
                batch_indices = np.arange(start_idx, end_idx)

                # Get projection function and labels for this batch
                proj_fun = inpainting_mgr.get_projection_fn(batch_indices)
                batch_labels = all_labels[batch_indices]

                print(f"  Batch {start_idx}-{end_idx} ({current_batch_size} samples)")

                # Sample with inpainting constraints
                batch_result = sampler.sample_sequences_with_pc_sampler(
                    checkpoint_path=args.checkpoint,
                    config=config,
                    num_samples=current_batch_size,
                    steps=steps,
                    architecture=args.architecture,
                    conditioning_labels=batch_labels,
                    save_elements_list=None,  # Don't save elements for iterations
                    start_at_timestep=args.start_at_timestep,
                    proj_fun=proj_fun
                )

                iteration_sequences.append(batch_result)

            # Concatenate batches for this iteration
            iteration_seqs = torch.cat(iteration_sequences, dim=0)
            iterations_sequences.append(iteration_seqs)

        # Stack iterations: (num_samples, num_iterations, seq_len)
        sequences = torch.stack(iterations_sequences, dim=1)
        saved_elements = None
        result = sequences
        conditioning_labels = all_labels  # For wandb logging

        # Custom save for iterated inpainting results
        if args.output:
            output_path = args.output
            print(f"\nSaving iterated inpainting results to {output_path}")
            with h5py.File(output_path, 'w') as f:
                # Save sequences with shape (num_samples, num_iterations, seq_len)
                f.create_dataset('sequences', data=sequences.cpu().numpy())
                # Save metadata from inpainting manager
                f.create_dataset('Y_target', data=inpainting_mgr.Y_target.cpu().numpy())
                f.create_dataset('X_original', data=inpainting_mgr.X.cpu().numpy())
                # Save position data
                f.create_dataset('start_dev', data=inpainting_mgr.start_dev)
                f.create_dataset('end_dev', data=inpainting_mgr.end_dev)
                f.create_dataset('start_hk', data=inpainting_mgr.start_hk)
                f.create_dataset('end_hk', data=inpainting_mgr.end_hk)
                # Save sampling config as attributes
                f.attrs['num_samples'] = num_samples_to_generate
                f.attrs['num_iterations'] = num_iterations
                f.attrs['steps'] = steps
                f.attrs['inpainting_mode'] = args.inpainting_mode
            print(f"Saved: sequences shape {sequences.shape} (samples, iterations, seq_len)")

            # Skip the normal handle_sample_result since we already saved
            results = {
                'num_sequences': num_samples_to_generate,
                'num_iterations': num_iterations,
                'sequence_length': sequences.shape[-1],
                'output_file': output_path,
                'encoding': 'index'
            }

            # Log to wandb if enabled
            if sampler.wandb_enabled:
                try:
                    # Reshape for wandb: flatten iterations
                    flat_seqs = sequences.reshape(-1, sequences.shape[-1])
                    sampler.log_to_wandb(
                        sequences=flat_seqs,
                        conditioning_labels=conditioning_labels.repeat_interleave(num_iterations, dim=0) if conditioning_labels is not None else None,
                        saved_elements=None
                    )
                except Exception as e:
                    print(f"Warning: Error logging to wandb: {e}")
                finally:
                    sampler.cleanup_wandb()

            print(f"\nDeepSTARR Sampling Results:")
            print("=" * 40)
            for key, value in results.items():
                print(f"{key}: {value}")

            print(f"\n✓ DeepSTARR inpainting sampling completed successfully!")
            return 0
    else:
        # Regular sampling without inpainting
        result = sampler.sample_sequences_with_pc_sampler(
            checkpoint_path=args.checkpoint,
            config=config,
            num_samples=num_samples_to_generate,
            steps=steps,
            architecture=args.architecture,
            conditioning_labels=all_labels,
            save_elements_list=args.save_elements,
            start_at_timestep=args.start_at_timestep
        )
        conditioning_labels = all_labels

    sequences, saved_elements, results = sampler.handle_sample_result(
        result, args.output, args.format, args.sequence_encoding
    )

    results.update(sampler.handle_saved_elements(saved_elements, args.output, 'deepstarr_samples'))

    # Log to wandb if enabled
    if sampler.wandb_enabled:
        try:
            sampler.log_to_wandb(
                sequences=sequences,
                conditioning_labels=conditioning_labels,
                saved_elements=saved_elements
            )
        except Exception as e:
            print(f"Warning: Error logging to wandb: {e}")
        finally:
            sampler.cleanup_wandb()

    print(f"\nDeepSTARR Sampling Results:")
    print("=" * 40)
    for key, value in results.items():
        print(f"{key}: {value}")

    print(f"\n✓ DeepSTARR sampling completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())