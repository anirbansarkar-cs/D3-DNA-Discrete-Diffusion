#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPRA LegNet: Single file implementation for training and inference.
Combines model architecture, data loading, training, and prediction functionality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torchmetrics import PearsonCorrCoef
from torch.utils.data import Dataset, DataLoader
import json
import math
import random
import numpy as np
import pandas as pd
import h5py
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Union, Tuple
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import tqdm


# ============================================================================
# DNA Utilities
# ============================================================================

# DNA encoding constants
CODES = {"A": 0, "T": 3, "G": 1, "C": 2, 'N': 4}
INV_CODES = {value: key for key, value in CODES.items()}
COMPL = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}


def n2id(n):
    """Convert nucleotide to integer ID."""
    return CODES[n.upper()]


def n2compl(n):
    """Get complement of nucleotide."""
    return COMPL[n.upper()]


def reverse_complement(seq, mapping={"A": "T", "G": "C", "T": "A", "C": "G", 'N': 'N'}):
    """Get reverse complement of DNA sequence."""
    return "".join(mapping[s] for s in reversed(seq))


class Seq2Tensor(nn.Module):
    """Convert DNA sequence to one-hot encoded tensor."""
    
    def __init__(self):
        super().__init__()

    def forward(self, seq):
        """
        Convert DNA sequence to tensor.
        
        Args:
            seq: DNA sequence string or existing tensor
            
        Returns:
            Tensor of shape (4, sequence_length) with one-hot encoding
        """
        if isinstance(seq, torch.FloatTensor):
            return seq
            
        seq = [n2id(x) for x in seq]
        code = torch.from_numpy(np.array(seq))
        code = F.one_hot(code, num_classes=5)
        code[code[:, 4] == 1] = 0.25  # encode Ns with .25
        code = code[:, :4].float()
        return code.transpose(0, 1)


# ============================================================================
# Model Utilities
# ============================================================================

def parameter_count(model):
    """Count total number of trainable parameters in model."""
    return sum(torch.prod(torch.tensor(p.shape)) for _, p in model.named_parameters())


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def initialize_weights(m):
    """Initialize model weights using appropriate strategies for each layer type."""
    if isinstance(m, nn.Conv1d):
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2 / n))
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


def check_h5_dataset(file_path):
    """
    Check and display the structure of an HDF5 dataset.
    
    Args:
        file_path: Path to HDF5 file
    """
    try:
        with h5py.File(file_path, 'r') as f:
            print("Keys in the HDF5 file:")
            for key in f.keys():
                print(f" - {key}")
            print("\nDataset structure:")
            def print_structure(name, obj):
                if hasattr(obj, 'shape'):
                    print(f"{name}: {obj.shape} {obj.dtype}")
            f.visititems(print_structure)
    except Exception as e:
        print(f"An error occurred: {e}")


# ============================================================================
# Dataset Classes
# ============================================================================

class HDF5Dataset(Dataset):
    """Dataset for HDF5 format MPRA data."""

    def __init__(self, h5_file, split='train', use_reverse=False, use_shift=False,
                 use_reverse_channel=False, max_shift=None, seqsize=230, training=True):
        """
        Initialize HDF5 dataset.
        
        Args:
            h5_file: Path to HDF5 file
            split: Data split ('train', 'valid', 'test')
            use_reverse: Whether to apply reverse complement augmentation
            use_shift: Whether to apply shift augmentation
            use_reverse_channel: Whether to add reverse complement indicator channel
            max_shift: Maximum shift range (tuple)
            seqsize: Expected sequence size
            training: Whether in training mode (affects augmentation)
        """
        self.h5_file = h5_file
        self.split = split
        self.use_reverse = use_reverse
        self.use_shift = use_shift
        self.use_reverse_channel = use_reverse_channel
        self.seqsize = seqsize
        self.training = training

        # Load data from HDF5
        with h5py.File(h5_file, 'r') as f:
            # Load one-hot encoded sequences
            self.sequences = f[f'onehot_{split}'][:]  # shape: (n, 230, 4)
            self.targets = f[f'y_{split}'][:].squeeze()  # shape: (n,)

        # Convert to proper format: (n, 4, 230)
        self.sequences = np.transpose(self.sequences, (0, 2, 1))

        # Shift parameters
        self.forward_side = "GGCCCGCTCTAGACCTGCAGG"
        self.reverse_side = "CACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGT"
        if max_shift is None:
            self.max_shift = (0, len(self.forward_side))
        else:
            self.max_shift = max_shift

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """Get a single sample."""
        # Get sequence as tensor (4, 230)
        seq = torch.from_numpy(self.sequences[idx]).float()
        target = self.targets[idx]

        # Apply augmentations during training
        if self.training:
            # Shift augmentation (simplified for one-hot data)
            if self.use_shift and self.max_shift[1] > 0:
                shift = torch.randint(low=-self.max_shift[0], high=self.max_shift[1] + 1, size=(1,)).item()
                if shift != 0:
                    # Simple circular shift for demonstration
                    seq = torch.roll(seq, shift, dims=-1)

            # Reverse complement augmentation
            if self.use_reverse:
                if torch.rand(1).item() > 0.5:
                    # Reverse complement: flip both sequence and nucleotide order
                    seq = torch.flip(seq, dims=(-1, -2))
                    rev = 1.0
                else:
                    rev = 0.0
            else:
                rev = 0.0
        else:
            rev = 0.0

        # Add reverse channel if needed
        to_concat = [seq]
        if self.use_reverse_channel:
            rev_channel = torch.full((1, self.seqsize), rev, dtype=torch.float32)
            to_concat.append(rev_channel)

        # Create final tensor
        if len(to_concat) > 1:
            X = torch.concat(to_concat, dim=0)
        else:
            X = seq

        return X, target.astype(np.float32)


class TSVDataset(Dataset):
    """Dataset for TSV format data (legacy support)."""

    def __init__(self, df, use_reverse=False, use_shift=False, use_reverse_channel=False,
                 max_shift=None, seqsize=230, training=True):
        """
        Initialize TSV dataset.
        
        Args:
            df: Pandas DataFrame with sequence data
            use_reverse: Whether to apply reverse complement augmentation
            use_shift: Whether to apply shift augmentation
            use_reverse_channel: Whether to add reverse complement indicator channel
            max_shift: Maximum shift range (tuple)
            seqsize: Expected sequence size
            training: Whether in training mode (affects augmentation)
        """
        self.df = df
        self.totensor = Seq2Tensor()
        self.use_reverse = use_reverse
        self.use_shift = use_shift
        self.use_reverse_channel = use_reverse_channel
        self.seqsize = seqsize
        self.training = training

        self.forward_side = "GGCCCGCTCTAGACCTGCAGG"
        self.reverse_side = "CACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGT"

        if max_shift is None:
            self.max_shift = (0, len(self.forward_side))
        else:
            self.max_shift = max_shift

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Get a single sample."""
        seq = self.df.seq.iloc[idx]
        target = self.df.mean_value.iloc[idx]

        # Apply shift augmentation
        if self.training and self.use_shift:
            shift = torch.randint(low=-self.max_shift[0], high=self.max_shift[1] + 1, size=(1,)).item()
            if shift < 0:
                seq = seq[:shift]
                seq = self.forward_side[shift:] + seq
            elif shift > 0:
                seq = seq[shift:]
                seq = seq + self.reverse_side[:shift]

        # Apply reverse complement
        if self.training and self.use_reverse:
            if torch.rand(1).item() > 0.5:
                seq = reverse_complement(seq)
                rev = 1.0
            else:
                rev = 0.0
        else:
            rev = 0.0

        # Convert to tensor
        seq = self.totensor(seq)
        to_concat = [seq]

        # Add reverse channel
        if self.use_reverse_channel:
            rev_channel = torch.full((1, self.seqsize), rev, dtype=torch.float32)
            to_concat.append(rev_channel)

        if len(to_concat) > 1:
            X = torch.concat(to_concat, dim=0)
        else:
            X = seq

        return X, target.astype(np.float32)


# ============================================================================
# Data Loading Functions
# ============================================================================

def create_dataloaders(config, data_file: str, data_format: str = 'h5'):
    """
    Create train/val dataloaders from either HDF5 or TSV format.
    
    Args:
        config: TrainingConfig object
        data_file: Path to data file
        data_format: Data format ('h5' or 'tsv')
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    if data_format == 'h5':
        train_ds = HDF5Dataset(
            data_file,
            split='train',
            use_reverse=config.reverse_augment,
            use_shift=config.use_shift,
            use_reverse_channel=config.use_reverse_channel,
            max_shift=config.max_shift,
            training=True
        )

        val_ds = HDF5Dataset(
            data_file,
            split='valid',
            use_reverse_channel=config.use_reverse_channel,
            training=False
        )

    elif data_format == 'tsv':
        # Legacy TSV support
        df = pd.read_csv(data_file, sep='\t')
        df.columns = ['seq_id', 'seq', 'mean_value', 'fold_num', 'rev'][:len(df.columns)]

        if "rev" in df.columns:
            df = df[df.rev == 0]

        # Simple train/val split for demonstration
        train_df = df.sample(frac=0.8, random_state=config.seed)
        val_df = df.drop(train_df.index)

        train_ds = TSVDataset(
            train_df.reset_index(drop=True),
            use_reverse=config.reverse_augment,
            use_shift=config.use_shift,
            use_reverse_channel=config.use_reverse_channel,
            max_shift=config.max_shift,
            training=True
        )

        val_ds = TSVDataset(
            val_df.reset_index(drop=True),
            use_reverse_channel=config.use_reverse_channel,
            training=False
        )
    else:
        raise ValueError(f"Unsupported data format: {data_format}")

    train_dl = DataLoader(
        train_ds,
        batch_size=config.train_batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        pin_memory=True
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=config.valid_batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True
    )

    return train_dl, val_dl


def create_test_dataloader(config, data_file: str, data_format: str = 'h5', split: str = 'test'):
    """
    Create test dataloader.
    
    Args:
        config: TrainingConfig object
        data_file: Path to data file
        data_format: Data format ('h5' or 'tsv')
        split: Data split name
        
    Returns:
        Test dataloader
    """
    if data_format == 'h5':
        test_ds = HDF5Dataset(
            data_file,
            split=split,
            use_reverse_channel=config.use_reverse_channel,
            training=False
        )
    else:
        raise NotImplementedError("TSV test dataloader not implemented")

    test_dl = DataLoader(
        test_ds,
        batch_size=config.valid_batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True
    )

    return test_dl


# ============================================================================
# Model Architecture Components
# ============================================================================

class SELayer(nn.Module):
    """Squeeze-and-Excitation layer for channel attention."""
    
    def __init__(self, inp, reduction=4):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(inp, int(inp // reduction)),
            nn.SiLU(),
            nn.Linear(int(inp // reduction), inp),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y


class EffBlock(nn.Module):
    """EfficientNet-style inverted residual block."""
    
    def __init__(self, in_ch, ks, resize_factor, activation, out_ch=None, se_reduction=None):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = self.in_ch if out_ch is None else out_ch
        self.resize_factor = resize_factor
        self.se_reduction = resize_factor if se_reduction is None else se_reduction
        self.ks = ks
        self.inner_dim = self.in_ch * self.resize_factor

        self.block = nn.Sequential(
            nn.Conv1d(in_channels=self.in_ch, out_channels=self.inner_dim, kernel_size=1, padding='same', bias=False),
            nn.BatchNorm1d(self.inner_dim),
            activation(),
            nn.Conv1d(in_channels=self.inner_dim, out_channels=self.inner_dim, kernel_size=ks, groups=self.inner_dim, padding='same', bias=False),
            nn.BatchNorm1d(self.inner_dim),
            activation(),
            SELayer(self.inner_dim, reduction=self.se_reduction),
            nn.Conv1d(in_channels=self.inner_dim, out_channels=self.in_ch, kernel_size=1, padding='same', bias=False),
            nn.BatchNorm1d(self.in_ch),
            activation(),
        )

    def forward(self, x):
        return self.block(x)


class LocalBlock(nn.Module):
    """Local convolution block."""
    
    def __init__(self, in_ch, ks, activation, out_ch=None):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = self.in_ch if out_ch is None else out_ch
        self.ks = ks

        self.block = nn.Sequential(
            nn.Conv1d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=self.ks, padding='same', bias=False),
            nn.BatchNorm1d(self.out_ch),
            activation()
        )

    def forward(self, x):
        return self.block(x)


class ResidualConcat(nn.Module):
    """Apply function and concatenate with input."""
    
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return torch.concat([self.fn(x, **kwargs), x], dim=1)


class MapperBlock(nn.Module):
    """Channel mapping block."""
    
    def __init__(self, in_features, out_features, activation=nn.SiLU):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class LegNet(nn.Module):
    """
    LegNet: Deep learning model for MPRA regulatory activity prediction.
    
    Architecture combines EfficientNet-style blocks with sequence-specific components
    for DNA regulatory element analysis.
    """
    
    def __init__(self, in_ch, stem_ch, stem_ks, ef_ks, ef_block_sizes, pool_sizes, resize_factor, activation=nn.SiLU):
        super().__init__()
        assert len(pool_sizes) == len(ef_block_sizes)

        self.in_ch = in_ch
        self.stem = LocalBlock(in_ch=in_ch, out_ch=stem_ch, ks=stem_ks, activation=activation)

        blocks = []
        in_ch = stem_ch
        out_ch = stem_ch

        for pool_sz, out_ch in zip(pool_sizes, ef_block_sizes):
            blc = nn.Sequential(
                ResidualConcat(
                    EffBlock(in_ch=in_ch, out_ch=in_ch, ks=ef_ks, resize_factor=resize_factor, activation=activation)
                ),
                LocalBlock(in_ch=in_ch * 2, out_ch=out_ch, ks=ef_ks, activation=activation),
                nn.MaxPool1d(pool_sz) if pool_sz != 1 else nn.Identity()
            )
            in_ch = out_ch
            blocks.append(blc)

        self.main = nn.Sequential(*blocks)
        self.mapper = MapperBlock(in_features=out_ch, out_features=out_ch * 2)
        self.head = nn.Sequential(
            nn.Linear(out_ch * 2, out_ch * 2),
            nn.BatchNorm1d(out_ch * 2),
            activation(),
            nn.Linear(out_ch * 2, 1)
        )

    def forward(self, x):
        """
        Forward pass through LegNet.
        
        Args:
            x: Input tensor of shape (batch_size, channels, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size,) with predicted activity scores
        """
        x = self.stem(x)
        x = self.main(x)
        x = self.mapper(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(-1)
        x = self.head(x)
        x = x.squeeze(-1)
        return x


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration class for training MPRA LegNet models."""
    
    # Model architecture parameters
    stem_ch: int = 64
    stem_ks: int = 11
    ef_ks: int = 9
    ef_block_sizes: List[int] = None
    resize_factor: int = 4
    pool_sizes: List[int] = None
    
    # Data augmentation parameters
    reverse_augment: bool = True
    use_reverse_channel: bool = False
    use_shift: bool = True
    max_shift: Optional[tuple] = None
    
    # Training parameters
    max_lr: float = 0.01
    weight_decay: float = 0.1
    epoch_num: int = 25
    train_batch_size: int = 1024
    valid_batch_size: int = 1024
    
    # System parameters
    model_dir: str = "./models/default_model"
    data_path: str = "../datasets/lenti_MPRA_K562_data.h5"
    device: int = 0
    seed: int = 777
    num_workers: int = 8

    def __post_init__(self):
        """Post-initialization setup with defaults."""
        if self.ef_block_sizes is None:
            self.ef_block_sizes = [80, 96, 112, 128]
        if self.pool_sizes is None:
            self.pool_sizes = [2, 2, 2, 2]

    @property
    def in_ch(self) -> int:
        """Calculate number of input channels."""
        return 4 + self.use_reverse_channel

    def get_model(self) -> nn.Module:
        """Create and return model instance."""
        return LegNet(
            in_ch=self.in_ch,
            stem_ch=self.stem_ch,
            stem_ks=self.stem_ks,
            ef_ks=self.ef_ks,
            ef_block_sizes=self.ef_block_sizes,
            resize_factor=self.resize_factor,
            pool_sizes=self.pool_sizes
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_json(self, path: Union[str, Path]):
        """Save configuration to JSON file."""
        dt = self.to_dict()
        with open(path, 'w') as out:
            json.dump(dt, out, indent=4)

    @classmethod
    def from_json(cls, path: Union[Path, str]) -> 'TrainingConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as inp:
            dt = json.load(inp)
        return cls(**dt)


def get_default_config() -> TrainingConfig:
    """Get default training configuration."""
    return TrainingConfig()


# ============================================================================
# Lightning Module
# ============================================================================

class LitModel(pl.LightningModule):
    """PyTorch Lightning module for LegNet training."""
    
    def __init__(self, tr_cfg):
        """
        Initialize Lightning module.
        
        Args:
            tr_cfg: TrainingConfig object with model and training parameters
        """
        super().__init__()
        self.tr_cfg = tr_cfg
        self.model = self.tr_cfg.get_model()
        self.model.apply(initialize_weights)
        self.loss = nn.MSELoss()
        self.val_pearson = PearsonCorrCoef()
        
        # Save hyperparameters
        self.save_hyperparameters({"config": self.tr_cfg.to_dict()})

    def training_step(self, batch, batch_idx):
        """Training step."""
        X, y = batch
        y_hat = self.model(X)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.val_pearson(y_hat, y)
        self.log("val_pearson", self.val_pearson, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return {'test_loss': loss, 'predictions': y_hat, 'targets': y}

    def predict_step(self, batch, batch_idx):
        """Prediction step."""
        if isinstance(batch, (tuple, list)):
            x, _ = batch
        else:
            x = batch
        y_hat = self.model(x)
        return y_hat

    def predict(self, X, batch_size=None, keepgrad=False):
        """
        Make predictions on input data (similar to predict_custom from DeepSTARR).
        
        Args:
            X: Input tensor or dataset
            batch_size: Batch size for prediction (uses config default if None)
            keepgrad: Whether to keep gradients
            
        Returns:
            Tensor of predictions
        """
        self.model.eval()
        if batch_size is None:
            batch_size = self.tr_cfg.valid_batch_size
            
        # Handle different input types
        if isinstance(X, torch.Tensor):
            dataloader = torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=False)
        else:
            # Assume it's already a DataLoader
            dataloader = X
            
        preds = torch.empty(0)
        if keepgrad:
            preds = preds.to(self.device)
        else:
            preds = preds.cpu()
            
        for x in tqdm.tqdm(dataloader, total=len(dataloader)):
            if isinstance(x, (tuple, list)):
                x = x[0]  # Take only the features, ignore labels
            x = x.to(self.device)
            
            with torch.set_grad_enabled(keepgrad):
                pred = self.model(x)
                if not keepgrad:
                    pred = pred.detach().cpu()
                preds = torch.cat((preds, pred), axis=0)
                
        return preds

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.tr_cfg.max_lr / 25,
            weight_decay=self.tr_cfg.weight_decay
        )
        
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.tr_cfg.max_lr,
            three_phase=False,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,
            cycle_momentum=False
        )

        return [optimizer], [{
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "name": "cycle_lr"
        }]
    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        # Log learning rate
        if self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('lr', current_lr, on_epoch=True)


# ============================================================================
# Training Function
# ============================================================================

def train_model(config, train_dataloader=None, val_dataloader=None, data_file=None, data_format='h5', verbose=False):
    """
    Train MPRA LegNet model.
    
    Args:
        config: TrainingConfig object
        train_dataloader: Training data loader (optional, will be created if None)
        val_dataloader: Validation data loader (optional, will be created if None)
        data_file: Path to data file (used if dataloaders not provided)
        data_format: Data format ('h5' or 'tsv')
        verbose: Whether to print verbose output
        
    Returns:
        Tuple of (model, trainer, best_model_path)
    """
    # Set random seed
    set_global_seed(config.seed)
    
    # Set PyTorch precision
    torch.set_float32_matmul_precision('medium')
    
    # Create dataloaders if not provided
    if train_dataloader is None or val_dataloader is None:
        if data_file is None:
            data_file = config.data_path
        train_dataloader, val_dataloader = create_dataloaders(config, data_file, data_format)
    
    # Create model
    model = LitModel(config)
    if verbose:
        print(f"Model parameters: {parameter_count(model.model).item():,}")
        print(f"Training samples: {len(train_dataloader.dataset):,}")
        print(f"Validation samples: {len(val_dataloader.dataset):,}")
    
    # Setup callbacks
    model_dir = Path(config.model_dir)
    model_dir.mkdir(exist_ok=True, parents=True)
    
    callbacks = [
        ModelCheckpoint(
            dirpath=model_dir,
            save_top_k=1,
            monitor="val_pearson",
            mode="max",
            filename="best_model-{epoch:02d}-{val_pearson:.3f}",
            save_last=True,
            save_weights_only=False
        ),
        EarlyStopping(
            monitor="val_pearson",
            mode="max",
            patience=10,
            verbose=verbose
        ),
        LearningRateMonitor(
            logging_interval='step'
        )
    ]
    
    # Create trainer
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=[config.device] if torch.cuda.is_available() else 1,
        precision='16-mixed' if torch.cuda.is_available() else 32,
        max_epochs=config.epoch_num,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        default_root_dir=config.model_dir,
        log_every_n_steps=50,
        val_check_interval=0.5,  # Validate twice per epoch
        enable_progress_bar=verbose,
        enable_model_summary=verbose
    )
    
    if verbose:
        print(f"\nTraining Configuration:")
        print(f"  Model directory: {config.model_dir}")
        print(f"  Epochs: {config.epoch_num}")
        print(f"  Batch size: {config.train_batch_size}")
        print(f"  Learning rate: {config.max_lr}")
        print(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'} {config.device}")
        print(f"  Reverse augmentation: {config.reverse_augment}")
        print(f"  Shift augmentation: {config.use_shift}")
        print()
    
    # Train model
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # Get best model path
    best_model_path = callbacks[0].best_model_path
    if verbose:
        print(f"\nTraining completed!")
        print(f"Best model saved at: {best_model_path}")
    
    return model, trainer, best_model_path


def load_model(checkpoint_path, config_path=None):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Optional path to config file (will try to load from checkpoint if not provided)
        
    Returns:
        Tuple of (model, config)
    """
    if config_path:
        config = TrainingConfig.from_json(config_path)
        model = LitModel.load_from_checkpoint(checkpoint_path, tr_cfg=config)
    else:
        # Try to load from checkpoint
        model = LitModel.load_from_checkpoint(checkpoint_path)
        config = model.tr_cfg
    
    model.eval()
    return model, config


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == '__main__':
    # Example configuration - you can modify these parameters
    config = get_default_config()
    
    # Override default parameters if needed
    config.data_path = "lenti_MPRA_K562_data.h5"  # Change to your data path
    config.model_dir = "oracle_models"
    config.epoch_num = 50
    config.train_batch_size = 512
    config.max_lr = 0.005
    
    print("MPRA LegNet Training Example")
    print("=" * 40)
    print("Configuration:")
    print(f"  Data path: {config.data_path}")
    print(f"  Model directory: {config.model_dir}")
    print(f"  Epochs: {config.epoch_num}")
    print(f"  Batch size: {config.train_batch_size}")
    print(f"  Learning rate: {config.max_lr}")
    
    # Check if data file exists
    if Path(config.data_path).exists():
        print(f"\nData file found: {config.data_path}")
        
        # Check data structure
        print("\nChecking data structure...")
        check_h5_dataset(config.data_path)
        
        # Train model (this will automatically create dataloaders)
        # model, trainer, best_model_path = train_model(
        #     config, 
        #     data_file=config.data_path, 
        #     data_format='h5', 
        #     verbose=True
        # )
        model, config = load_model(
            "oracle_models/best_model-epoch=24-val_pearson=0.814.ckpt",
            "oracle_models/config.json"
        )
        
        # Example prediction
        test_dl = create_test_dataloader(config, config.data_path)
        predictions = model.predict(test_dl)
        # Calculate Pearson correlation coefficient
        predictions_np = predictions.cpu().numpy().flatten()
        targets_np = test_dl.dataset.targets.flatten()
        pearson_corr = np.corrcoef(predictions_np, targets_np)[0, 1]
        print(f"Pearson correlation coefficient: {pearson_corr}")
        print(f"\nGenerated {len(predictions)} predictions")
        
    else:
        print(f"\nData file not found: {config.data_path}")
        print("\nTo use this script:")
        print("1. Set config.data_path to your HDF5 data file")
        print("2. Run: model, trainer, best_path = train_model(config, data_file='path/to/data.h5')")
        print("3. Make predictions: predictions = model.predict(test_data)")
        print("4. Or load saved model: model, config = load_model('path/to/checkpoint.ckpt')")