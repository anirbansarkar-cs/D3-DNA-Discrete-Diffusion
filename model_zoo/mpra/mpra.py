"""
MPRA Oracle Model for MPRA Dataset

This module contains the MPRA (Massively Parallel Reporter Assay) model architecture.
This is the oracle model used for evaluation against D3 generated sequences for MPRA data.

The model predicts regulatory activity from DNA sequences using a convolutional neural network
with dilated residual blocks for capturing long-range dependencies.
"""

import torch
from torch import nn
import pytorch_lightning as pl
import torch.optim as optim

import os
import h5py
import numpy as np
import tqdm
from scipy import stats
import random
from typing import Dict
from pytorch_lightning import loggers as pl_loggers
from pathlib import Path

class DilatedResidual(pl.LightningModule):
    """Dilated residual block for capturing long-range dependencies.
    
    Args:
        filter_num: Number of filters
        kernel_size: Size of convolutional kernels
        dilation_rate: List of dilation rates for each layer
        dropout: Dropout probability
    """
    
    def __init__(self, filter_num: int, kernel_size: int, 
                 dilation_rate: list, dropout: float):
        super().__init__()
        layers = []
        layers.append(nn.Conv1d(filter_num, filter_num, kernel_size, padding='same'))
        layers.append(nn.BatchNorm1d(filter_num))
        
        for dilation in dilation_rate:
            layers.extend([
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(filter_num, filter_num, kernel_size, stride=1,
                         padding='same', dilation=dilation, bias=False),
                nn.BatchNorm1d(filter_num)
            ])
            
        self.block = nn.Sequential(*layers)
        self.output_act = nn.ReLU()

    def forward(self, x):
        """Forward pass with residual connection."""
        out = self.block(x)
        residual = torch.add(out, x)
        output = self.output_act(residual)
        return output

class MPRA(nn.Module):
    """MPRA model architecture for regulatory activity prediction.
    
    This model uses convolutional layers with dilated residual blocks
    to predict regulatory activity from DNA sequences.
    
    Args:
        output_dim: Number of output dimensions (typically 1 for MPRA)
    """
    
    def __init__(self, output_dim: int):
        super().__init__()
        
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv1d(4, 196, 15, padding='same'),
            nn.BatchNorm1d(196),
            nn.ELU(),
            nn.Dropout(0.1)
        )
        
        # First residual block
        self.res1 = nn.Sequential(
            DilatedResidual(196, 3, [1, 2, 4, 8], 0.1),
            nn.MaxPool1d(4),
            nn.Dropout(0.2)
        )
        
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv1d(196, 256, 5, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Second residual block
        self.res2 = nn.Sequential(
            DilatedResidual(256, 3, [1, 2, 4], 0.1),
            nn.MaxPool1d(4),
            nn.Dropout(0.2)
        )
        
        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, 3, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Third residual block
        self.res3 = nn.Sequential(
            DilatedResidual(256, 3, [1], 0.1),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        
        # Fully connected layers
        self.activation = nn.ReLU()
        self.dropout4 = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        
        self.fc4 = nn.LazyLinear(512, bias=True)
        self.batchnorm4 = nn.BatchNorm1d(512)
        
        self.fc5 = nn.LazyLinear(256, bias=True)
        self.batchnorm5 = nn.BatchNorm1d(256)
        
        # Output layer
        self.fc6 = nn.Linear(256, output_dim)

    def forward(self, x):
        """Forward pass through the MPRA model.
        
        Args:
            x: Input tensor of shape (batch_size, 4, sequence_length)
            
        Returns:
            Predicted regulatory activity scores
        """
        # Convolutional layers with residual blocks
        x = self.conv1(x)
        x = self.res1(x)
        
        x = self.conv2(x)
        x = self.res2(x)
        
        x = self.conv3(x)
        x = self.res3(x)
        
        # Fully connected layers
        x = self.flatten(x)
        
        x = self.fc4(x)
        x = self.batchnorm4(x)
        x = self.activation(x)
        x = self.dropout4(x)
        
        x = self.fc5(x)
        x = self.batchnorm5(x)
        x = self.activation(x)
        x = self.dropout4(x)
        
        # Output
        y_pred = self.fc6(x)
        return y_pred


# =============================================================================
# PyTorch Lightning Integration
# =============================================================================

def get_github_main_directory(reponame: str = 'DALdna') -> str:
    """Get the main directory path for the given repository name."""
    currdir = os.getcwd()
    dir_path = ''
    for dirname in currdir.split('/'):
        dir_path += dirname + '/'
        if dirname == reponame:
            break
    return dir_path


def key_with_low(key_list: list, low: str) -> str:
    """Find key in list that matches lowercase string."""
    the_key = ''
    for key in key_list:
        if key.lower() == low:
            the_key = key
    return the_key


class PL_MPRA(pl.LightningModule):
    """PyTorch Lightning wrapper for MPRA model.
    
    This class provides training, validation, and testing functionality
    for the MPRA oracle model used in D3 evaluations.
    """
    
    def __init__(self,
                 batch_size: int = 128,
                 train_max_epochs: int = 100,
                 patience: int = 10,
                 min_delta: float = 0.001,
                 input_h5_file: str = 'mpra_tewhey.h5',
                 lr: float = 0.002,
                 initial_ds: bool = True,
                 weight_decay: float = 1e-6,
                 min_lr: float = 0.0,
                 lr_patience: int = 10,
                 decay_factor: float = 0.1,
                 scale: float = 0.005,
                 initialization: str = 'kaiming_uniform',
                 initialize_dense: bool = False):
        super().__init__()
        self.save_hyperparameters()
        
        # Model configuration
        self.scale = scale
        self.model = MPRA(output_dim=3)  # 3 outputs for MPRA signals
        self.name = 'mpra'
        self.metric_names = ['PCC', 'Spearman']
        self.initial_ds = initial_ds
        
        # Training configuration
        self.batch_size = batch_size
        self.train_max_epochs = train_max_epochs
        self.patience = patience
        self.lr = lr
        self.min_delta = min_delta
        self.weight_decay = weight_decay
        self.min_lr = min_lr
        self.lr_patience = lr_patience
        self.decay_factor = decay_factor
        
        # Data configuration
        self.input_h5_file = input_h5_file
        # Load data
        data = h5py.File(input_h5_file, 'r')
        if initial_ds:
            # Load and preprocess training data
            self.X_train = torch.tensor(
                np.array(data['x_train']).astype(np.float32)
            ).permute(0, 2, 1)
            self.y_train = torch.tensor(
                np.array(data['y_train']).astype(np.float32)
            )
            
            # Load and preprocess test data
            self.X_test = torch.tensor(
                np.array(data['x_test']).astype(np.float32)
            ).permute(0, 2, 1)
            self.y_test = torch.tensor(
                np.array(data['y_test']).astype(np.float32)
            )
            
            # Load and preprocess validation data
            self.X_valid = torch.tensor(
                np.array(data['x_valid']).astype(np.float32)
            ).permute(0, 2, 1)
            self.y_valid = torch.tensor(
                np.array(data['y_valid']).astype(np.float32)
            )
            
            self.X_test2 = self.X_test
            self.y_test2 = self.y_test
        else:
            # Use raw data without preprocessing
            self.X_train = data['x_train']
            self.y_train = data['y_train']
            self.X_test = data['x_test']
            self.y_test = data['y_test']
            self.X_test2 = data['x_test2']
            self.y_test2 = data['y_test2']
            self.X_valid = data['x_valid']
            self.y_valid = data['y_valid']

    def training_step(self, batch, batch_idx):
        """Training step for one batch."""
        self.model.train()
        inputs, labels = batch
        loss_fn = nn.MSELoss()
        outputs = self.model(inputs)
        loss = loss_fn(outputs, labels)
        
        self.log("train_loss", loss, on_step=False, on_epoch=True,
                prog_bar=True, logger=True, sync_dist=True)
        
        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=self.lr_patience,
            min_lr=self.min_lr,
            factor=self.decay_factor
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

    def validation_step(self, batch, batch_idx):
        """Validation step for one batch."""
        self.model.eval()
        inputs, labels = batch
        loss_fn = nn.MSELoss()
        outputs = self.model(inputs)
        loss = loss_fn(outputs, labels)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True,
                prog_bar=True, logger=True, sync_dist=True)
        
        # Calculate and log PCC metric
        out_cpu = outputs.detach().cpu()
        lab_cpu = labels.detach().cpu()
        pcc = torch.tensor(self.metrics(out_cpu, lab_cpu)['PCC'].mean())
        self.log("val_pcc", pcc, on_step=False, on_epoch=True,
                prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        """Test step for one batch."""
        self.model.eval()
        inputs, labels = batch
        loss_fn = nn.MSELoss()
        outputs = self.model(inputs)
        loss = loss_fn(outputs, labels)
        
        self.log("test_loss", loss, on_step=False, on_epoch=True,
                prog_bar=True, logger=True)

    def metrics(self, y_score, y_true):
        """Calculate Pearson and Spearman correlation metrics."""
        # Spearman correlation
        spearman_vals = []
        for output_index in range(y_score.shape[1]):
            spearman_vals.append(
                stats.spearmanr(y_true[:, output_index], y_score[:, output_index])[0]
            )
        spearmanr_vals = np.array(spearman_vals)
        
        # Pearson correlation
        pearson_vals = []
        for output_index in range(y_score.shape[-1]):
            pearson_vals.append(
                stats.pearsonr(y_true[:, output_index], y_score[:, output_index])[0]
            )
        pearsonr_vals = np.array(pearson_vals)
        
        return {'Spearman': spearmanr_vals, 'PCC': pearsonr_vals}

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

    def predict_custom(self, X, keepgrad=False):
        """Custom prediction function with batch processing."""
        self.model.eval()
        
        device = next(self.model.parameters()).device
        
        if X.device != device:
            X = X.to(device)
        
        dataloader = torch.utils.data.DataLoader(
            X, batch_size=self.batch_size, shuffle=False, pin_memory=False
        )
        
        preds = []
        
        for x in tqdm.tqdm(dataloader, total=len(dataloader)):
            with torch.no_grad():
                pred = self.model(x)
                if not keepgrad:
                    pred = pred.detach().cpu()
                preds.append(pred)
                
        return torch.cat(preds, axis=0)

    def predict_custom_mcdropout(self, X, seed=41, keepgrad=False):
        """Prediction with Monte Carlo dropout for uncertainty estimation."""
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        self.model.eval()
        # Enable dropout during inference for MC dropout
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
        
        dataloader = torch.utils.data.DataLoader(
            X, batch_size=self.batch_size, shuffle=False
        )
        preds = torch.empty(0)
        
        if keepgrad:
            preds = preds.to(self.device)
        else:
            preds = preds.cpu()
        
        for x in tqdm.tqdm(dataloader, total=len(dataloader)):
            pred = self.model(x.to(self.device))
            if not keepgrad:
                pred = pred.detach().cpu()
            preds = torch.cat((preds, pred), axis=0)
        
        return preds


# =============================================================================
# Training Functions
# =============================================================================


def training_with_PL(chosen_model: str, chosen_dataset: str,
                     initial_test: bool = False,
                     mcdropout_test: bool = False,
                     verbose: bool = False,
                     wanted_wandb: bool = False) -> Dict[str, np.ndarray]:
    """Train MPRA model using PyTorch Lightning.
    
    Args:
        chosen_model: Name of the model ('mpra')
        chosen_dataset: Name of the dataset ('mpra_data')
        initial_test: Whether to test before training
        mcdropout_test: Whether to test with Monte Carlo dropout
        verbose: Whether to print verbose output
        wanted_wandb: Whether to use Weights & Biases logging
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Setup logging
    if wanted_wandb:
        import wandb
        from pytorch_lightning.loggers import WandbLogger
        wandb_logger = WandbLogger(log_model="all")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device=} {torch.cuda.is_available()=}")
    
    # Setup directories and logging
    currdir = os.popen('pwd').read().replace("\n", "")
    outdir = "../outputs/"
    log_dir = outdir + "lightning_logs_" + chosen_model + "/"
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir)

    logger_of_choice = wandb_logger if wanted_wandb else tb_logger
    
    # Initialize model - note: use PL_MPRA class name
    if chosen_model.lower() == 'mpra':
        model = PL_MPRA(input_h5_file=f'./{chosen_dataset}.h5', initial_ds=True)
    else:
        raise ValueError(f"Unknown model: {chosen_model}")

    # Setup data loaders
    if verbose:
        os.system('date')
        print(f"Training data shape: {model.X_train.shape}")
        print(f"Training labels shape: {model.y_train.shape}")
    
    train_dataloader = torch.utils.data.DataLoader(
        list(zip(model.X_train, model.y_train)),
        batch_size=model.batch_size,
        shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        list(zip(model.X_valid, model.y_valid)),
        batch_size=model.batch_size,
        shuffle=False
    )
    test_dataloader = torch.utils.data.DataLoader(
        list(zip(model.X_test, model.y_test)),
        batch_size=model.batch_size,
        shuffle=False
    )

    # Setup callbacks
    ckptfile = f"oracle_{model.name}_{chosen_dataset}"
    to_monitor = 'val_loss'
    
    callback_ckpt = pl.callbacks.ModelCheckpoint(
        monitor=to_monitor,
        mode='min',
        save_top_k=1,
        save_weights_only=True,
        dirpath="./",
        filename=ckptfile,
    )
    
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor=to_monitor,
        min_delta=model.min_delta,
        patience=model.patience,
        verbose=False,
        mode='min'
    )

    # Initial testing before training
    if initial_test:
        print('Running initial test...')
        y_score = model.predict_custom(model.X_test)
        print(f'y_score.shape: {y_score.shape}')
        metrics_pretrain = model.metrics(y_score, model.y_test)
        print(f"Pre-training metrics: {metrics_pretrain}")
        print(f"Sample predictions: {model(model.X_test[0:10])}")

    # Monte Carlo dropout testing
    if mcdropout_test:
        print('Running Monte Carlo dropout test...')
        n_mc = 5
        preds_mc = torch.zeros((n_mc, len(model.X_test)))
        for i in range(n_mc):
            preds_mc[i] = model.predict_custom_mcdropout(
                model.X_test, seed=41+i
            ).squeeze(axis=1).unsqueeze(axis=0)
        print(f'MC dropout predictions shape: {preds_mc.shape}')
        metrics_pretrain = model.metrics(y_score, model.y_test)
        print(f"MC dropout metrics: {metrics_pretrain}")
        print(f"Sample MC predictions: {model(model.X_test[0:10])}")

    # Training
    print(f"Model device: {model.device}")
    trainer = pl.Trainer(
        accelerator='cuda',
        devices=-1,
        max_epochs=model.train_max_epochs,
        logger=logger_of_choice,
        callbacks=[callback_ckpt, early_stop_callback],
        deterministic=True
    )
    
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
    
    # Rename checkpoint file
    os.system(f'mv ./{ckptfile}-v1.ckpt ./{ckptfile}.ckpt')
    
    # Final evaluation
    if verbose:
        os.system('date')
    y_score = model.predict_custom(model.X_test)
    if verbose:
        os.system('date')
    metrics = model.metrics(y_score, model.y_test)
    print(f"Final metrics: {metrics}")
    
    # Log to wandb if enabled
    if wanted_wandb:
        import wandb
        wandb.log(metrics)
    
    print(f"Model checkpoint saved as: {ckptfile}")
    return metrics


# =============================================================================
# Main Execution
# =============================================================================


if __name__ == '__main__':
    """Main execution for MPRA oracle model inference."""
    
    # Configuration
    chosen_model = 'mpra'
    chosen_dataset = 'mpra_data'
    data_path = f'./{chosen_dataset}.h5'
    checkpoint_path = 'oracle_models/oracle_mpra_mpra_data.ckpt'
    
    print("MPRA Oracle Model Inference")
    print("=" * 40)
    print("Configuration:")
    print(f"  Model: {chosen_model}")
    print(f"  Dataset: {chosen_dataset}")
    print(f"  Data path: {data_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    
    # Check if data file exists
    if Path(data_path).exists():
        print(f"\nData file found: {data_path}")
        
        # Suppress Lightning logs
        import logging
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        
        # Check if checkpoint exists
        if Path(checkpoint_path).exists():
            print(f"Checkpoint found: {checkpoint_path}")
            
            # Load pre-trained model
            try:
                model = PL_MPRA(input_h5_file=data_path, initial_ds=True)
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                model.eval()
                print("✓ Loaded pre-trained model from checkpoint")
                
                # Get test data
                print(f"\nTest data shape: {model.X_test.shape}")
                print(f"Test labels shape: {model.y_test.shape}")
                
                # Make predictions
                print("Making predictions...")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
                
                # Move test data to the same device as the model
                X_test_device = model.X_test.to(device)
                
                y_score = model.predict_custom(X_test_device)
                y_true = model.y_test
                
                print(f"Predictions shape: {y_score.shape}")
                print(f"Targets shape: {y_true.shape}")
                
                # Calculate metrics using the model's built-in metrics function
                metrics_dict = model.metrics(y_score.cpu(), y_true.cpu())
                
                # Extract correlations
                pearson_vals = metrics_dict['PCC'] 
                spearman_vals = metrics_dict['Spearman']
                
                print(f"\n" + "="*50)
                print("INFERENCE RESULTS")
                print("="*50)
                print(f"Pearson correlations: {pearson_vals}")
                print(f"Mean Pearson r: {pearson_vals.mean():.4f}")
                print(f"Spearman correlations: {spearman_vals}")
                print(f"Mean Spearman rho: {spearman_vals.mean():.4f}")
                
                # Print individual signal correlations (MPRA has 3 signals)
                signal_names = ['K562', 'HepG2', 'SK-N-SH']
                print(f"\nPer-cell line correlations:")
                for i, signal in enumerate(signal_names):
                    print(f"  {signal}:")
                    print(f"    Pearson r: {pearson_vals[i]:.4f}")
                    print(f"    Spearman rho: {spearman_vals[i]:.4f}")
                print("="*50)
                
            except Exception as e:
                print(f"Error loading model or making predictions: {e}")
                print("\nTo train a model first, uncomment the training section below:")
                print("# metrics = training_with_PL(chosen_model, chosen_dataset)")
        else:
            print(f"\nCheckpoint not found: {checkpoint_path}")
            print("Training a new model...")
            
            # Commented out training - uncomment to train
            # metrics = training_with_PL(
            #     chosen_model, 
            #     chosen_dataset, 
            #     initial_test=True, 
            #     mcdropout_test=False, 
            #     verbose=True, 
            #     wanted_wandb=False
            # )
            # print(f"Training completed. Final metrics: {metrics}")
            
            print("Training is commented out. To train:")
            print("1. Uncomment the training_with_PL call above")
            print("2. Run the script to generate the checkpoint")
            print("3. Re-run for inference")
    else:
        print(f"\nData file not found: {data_path}")
        print("\nTo use this script:")
        print("1. Ensure mpra_data.h5 is in the current directory")
        print("2. Either train a model first or provide a pre-trained checkpoint")
        print("3. Run inference to get Pearson and Spearman correlations")
