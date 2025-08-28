"""
DeepSTARR Oracle Model for DeepSTARR Dataset

This module contains the original DeepSTARR model architecture from
de Almeida et al., 2022 (https://www.nature.com/articles/s41588-022-01048-5).

This is the oracle model used for evaluation against D3 generated sequences.
It predicts enhancer activity for both developmental and housekeeping promoters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
import random
import h5py
import os
from scipy import stats
from pytorch_lightning import loggers as pl_loggers
import tqdm
from filelock import FileLock
from typing import Any, Dict, Optional
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path

class DeepSTARR(nn.Module):
    """DeepSTARR model from de Almeida et al., 2022; 
        see <https://www.nature.com/articles/s41588-022-01048-5>
    """
    def __init__(self, output_dim, d=256,
                 conv1_filters=None, learn_conv1_filters=True,
                 conv2_filters=None, learn_conv2_filters=True,
                 conv3_filters=None, learn_conv3_filters=True,
                 conv4_filters=None, learn_conv4_filters=True):
        super().__init__()
        
        if d != 256:
            print("NB: number of first-layer convolutional filters in original DeepSTARR model is 256; current number of first-layer convolutional filters is not set to 256")
        
        self.activation = nn.ReLU()
        self.dropout4 = nn.Dropout(0.4)
        self.flatten = nn.Flatten()
        
        self.init_conv1_filters = conv1_filters
        self.init_conv2_filters = conv2_filters
        self.init_conv3_filters = conv3_filters
        self.init_conv4_filters = conv4_filters
        
        assert (not (conv1_filters is None and not learn_conv1_filters)), "initial conv1_filters cannot be set to None while learn_conv1_filters is set to False"
        assert (not (conv2_filters is None and not learn_conv2_filters)), "initial conv2_filters cannot be set to None while learn_conv2_filters is set to False"
        assert (not (conv3_filters is None and not learn_conv3_filters)), "initial conv3_filters cannot be set to None while learn_conv3_filters is set to False"
        assert (not (conv4_filters is None and not learn_conv4_filters)), "initial conv4_filters cannot be set to None while learn_conv4_filters is set to False"
        
        # Layer 1 (convolutional), constituent parts
        if conv1_filters is not None:
            if learn_conv1_filters: # continue modifying existing conv1_filters through learning
                self.conv1_filters = nn.Parameter( torch.Tensor(conv1_filters) )
            else:
                self.register_buffer("conv1_filters", torch.Tensor(conv1_filters))
        else:
            self.conv1_filters = nn.Parameter(torch.zeros(d, 4, 7))
            nn.init.kaiming_normal_(self.conv1_filters)
        self.batchnorm1 = nn.BatchNorm1d(d)
        self.activation1 = nn.ReLU() # name the first-layer activation function for hook purposes
        self.maxpool1 = nn.MaxPool1d(2)
        
        # Layer 2 (convolutional), constituent parts
        if conv2_filters is not None:
            if learn_conv2_filters: # continue modifying existing conv2_filters through learning
                self.conv2_filters = nn.Parameter( torch.Tensor(conv2_filters) )
            else:
                self.register_buffer("conv2_filters", torch.Tensor(conv2_filters))
        else:
            self.conv2_filters = nn.Parameter(torch.zeros(60, d, 3))
            nn.init.kaiming_normal_(self.conv2_filters)
        self.batchnorm2 = nn.BatchNorm1d(60)
        self.maxpool2 = nn.MaxPool1d(2)
        
        # Layer 3 (convolutional), constituent parts
        if conv3_filters is not None:
            if learn_conv3_filters: # continue modifying existing conv3_filters through learning
                self.conv3_filters = nn.Parameter( torch.Tensor(conv3_filters) )
            else:
                self.register_buffer("conv3_filters", torch.Tensor(conv3_filters))
        else:
            self.conv3_filters = nn.Parameter(torch.zeros(60, 60, 5))
            nn.init.kaiming_normal_(self.conv3_filters)
        self.batchnorm3 = nn.BatchNorm1d(60)
        self.maxpool3 = nn.MaxPool1d(2)
        
        # Layer 4 (convolutional), constituent parts
        if conv4_filters is not None:
            if learn_conv4_filters: # continue modifying existing conv4_filters through learning
                self.conv4_filters = nn.Parameter( torch.Tensor(conv4_filters) )
            else:
                self.register_buffer("conv4_filters", torch.Tensor(conv4_filters))
        else:
            self.conv4_filters = nn.Parameter(torch.zeros(120, 60, 3))
            nn.init.kaiming_normal_(self.conv4_filters)
        self.batchnorm4 = nn.BatchNorm1d(120)
        self.maxpool4 = nn.MaxPool1d(2)
        
        # Layer 5 (fully connected), constituent parts
        self.fc5 = nn.LazyLinear(256, bias=True)
        self.batchnorm5 = nn.BatchNorm1d(256)
        
        # Layer 6 (fully connected), constituent parts
        self.fc6 = nn.Linear(256, 256, bias=True)
        self.batchnorm6 = nn.BatchNorm1d(256)
        
        # Output layer (fully connected), constituent parts
        self.fc7 = nn.Linear(256, output_dim)
        
    def get_which_conv_layers_transferred(self):
        layers = []
        if self.init_conv1_filters is not None:
            layers.append(1)
        if self.init_conv2_filters is not None:
            layers.append(2)
        if self.init_conv3_filters is not None:
            layers.append(3)
        if self.init_conv4_filters is not None:
            layers.append(4)
        return layers
    
    def forward(self, x):

        cnn = torch.conv1d(x, self.conv1_filters, stride=1, padding="same")
        cnn = self.batchnorm1(cnn)
        cnn = self.activation1(cnn)
        cnn = self.maxpool1(cnn)
        
        # Layer 2
        cnn = torch.conv1d(cnn, self.conv2_filters, stride=1, padding="same")
        cnn = self.batchnorm2(cnn)
        cnn = self.activation(cnn)
        cnn = self.maxpool2(cnn)
        
        # Layer 3
        cnn = torch.conv1d(cnn, self.conv3_filters, stride=1, padding="same")
        cnn = self.batchnorm3(cnn)
        cnn = self.activation(cnn)
        cnn = self.maxpool3(cnn)
        
        # Layer 4
        cnn = torch.conv1d(cnn, self.conv4_filters, stride=1, padding="same")
        cnn = self.batchnorm4(cnn)
        cnn = self.activation(cnn)
        cnn = self.maxpool4(cnn)
        
        # Layer 5
        cnn = self.flatten(cnn)
        cnn = self.fc5(cnn)
        cnn = self.batchnorm5(cnn)
        cnn = self.activation(cnn)
        cnn = self.dropout4(cnn)
        
        # Layer 6
        cnn = self.fc6(cnn)
        cnn = self.batchnorm6(cnn)
        cnn = self.activation(cnn)
        cnn = self.dropout4(cnn)
        
        # Output layer
        y_pred = self.fc7(cnn) 
        
        return y_pred


# =============================================================================
# PyTorch Lightning Integration
# =============================================================================

def get_github_main_directory(reponame='DALdna'):
    """Get the main directory path for the given repository name."""
    currdir = os.getcwd()
    dir = ''
    for dirname in currdir.split('/'):
        dir += dirname + '/'
        if dirname == reponame:
            break
    return dir


def key_with_low(key_list, low):
    """Find key in list that matches lowercase string."""
    the_key = ''
    for key in key_list:
        if key.lower() == low:
            the_key = key
    return the_key 

class PL_DeepSTARR(pl.LightningModule):
    """PyTorch Lightning wrapper for DeepSTARR model.
    
    This class provides training, validation, and testing functionality
    for the DeepSTARR oracle model used in D3 evaluations.
    """
    
    def __init__(self,
                 batch_size: int = 128,
                 train_max_epochs: int = 100,
                 patience: int = 10,
                 min_delta: float = 0.001,
                 input_h5_file: str = 'DeepSTARR_data.h5',
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
        self.model = DeepSTARR(output_dim=2)
        self.name = 'DeepSTARR'
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
        data = h5py.File(input_h5_file, 'r')
        if initial_ds:
            self.X_train = torch.tensor(np.array(data['X_train'])) #(402278, 4, 249)
            self.y_train = torch.tensor(np.array(data['Y_train']))#.unsqueeze(1)
            self.X_test = torch.tensor(np.array(data['X_test']))
            self.y_test = torch.tensor(np.array(data['Y_test']))#.unsqueeze(1)
            self.X_valid = torch.tensor(np.array(data['X_valid']))
            self.y_valid = torch.tensor(np.array(data['Y_valid']))#.unsqueeze(1)                                                     #DSRR
            self.X_test2 = self.X_test
            self.y_test2 = self.y_test
        else:
            self.X_train=data['X_train']
            self.y_train=data['Y_train']
            self.X_test=data['X_test']
            self.y_test=data['Y_test']
            self.X_test2=data['X_test2']
            self.y_test2=data['Y_test2']
            self.X_valid=data['X_valid']
            self.y_valid=data['Y_valid']

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
        dataloader = torch.utils.data.DataLoader(
            X, batch_size=self.batch_size, shuffle=False
        )
        preds = torch.empty(0)
        
        if keepgrad:
            preds = preds.to(self.device)
        else:
            preds = preds.cpu()
        
        for x in tqdm.tqdm(dataloader, total=len(dataloader)):
            pred = self.model(x)
            if not keepgrad:
                pred = pred.detach().cpu()
            preds = torch.cat((preds, pred), axis=0)
            
        return preds

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
            pred = self.model(x)
            if not keepgrad:
                pred = pred.detach().cpu()
            preds = torch.cat((preds, pred), axis=0)
        
        return preds


# 2025-08-28: removing evoaug integration as RobustModel - now using RobustLoader 
# implementation in the dataloader/datamodule creation (data.py)


# =============================================================================
# Training Functions
# =============================================================================


def training_with_PL(chosen_model: str, chosen_dataset: str,
                     initial_test: bool = False, 
                     mcdropout_test: bool = False, 
                     verbose: bool = False, 
                     wanted_wandb: bool = False,
                     seed: int = 41) -> Dict[str, np.ndarray]:
    """Train DeepSTARR model using PyTorch Lightning.
    
    Args:
        chosen_model: Name of the model ('DeepSTARR')
        chosen_dataset: Name of the dataset ('DeepSTARR_data')
        initial_test: Whether to test before training
        mcdropout_test: Whether to test with Monte Carlo dropout
        verbose: Whether to print verbose output
        wanted_wandb: Whether to use Weights & Biases logging
        
    Returns:
        Dictionary containing evaluation metrics
    """

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

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
    
    # Initialize model
    model = eval(f"PL_{chosen_model}(input_h5_file='./{chosen_dataset}.h5', initial_ds=True)")

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
    """Main execution for DeepSTARR oracle model inference."""
    
    # Configuration
    chosen_model = 'DeepSTARR'
    chosen_dataset = 'DeepSTARR_data'
    data_path = f'./{chosen_dataset}.h5'
    checkpoint_path = 'oracle_models/oracle_DeepSTARR_DeepSTARR_data.ckpt'
    
    print("DeepSTARR Oracle Model Inference")
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
                model = PL_DeepSTARR(input_h5_file=data_path, initial_ds=True)
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
                
                # Print individual task correlations
                task_names = ['Developmental', 'Housekeeping']
                print(f"\nPer-task correlations:")
                for i, task in enumerate(task_names):
                    print(f"  {task}:")
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
        print("1. Ensure DeepSTARR_data.h5 is in the current directory")
        print("2. Either train a model first or provide a pre-trained checkpoint")
        print("3. Run inference to get Pearson and Spearman correlations")