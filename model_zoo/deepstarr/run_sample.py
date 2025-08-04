"""
DeepSTARR Sampling Script

This script generates DNA sequences using a trained diffusion model and evaluates them
against the DeepSTARR oracle model to compute MSE scores.
"""

import argparse
import os
import math
from typing import Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import sys

# Project imports
from model import rotary
from model.ema import ExponentialMovingAverage
from model.fused_add_dropout_scale import (
    bias_dropout_add_scale_fused_train, 
    bias_dropout_add_scale_fused_inference, 
    modulate_fused,
)
from model_zoo.deepstarr.deepstarr import PL_DeepSTARR
from scripts import sampling
from utils import graph_lib, noise_lib


def modulate(x, shift, scale):
    """Apply adaptive layer norm modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LayerNorm(nn.Module):
    """Custom LayerNorm implementation."""
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim
        
    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations with dropout for classifier-free guidance."""
    
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """Drops labels to enable classifier-free guidance."""
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class DDiTBlock(nn.Module):
    """Diffusion Transformer Block with adaptive layer norm."""
    
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout

        # Attention layers
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        # MLP layers
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True)
        )

        # Adaptive layer norm modulation
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )

    def forward(self, x, rotary_cos_sin, c, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        )

        # Attention operation
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
        
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = rotary.apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
            )
        
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        if seqlens is None:
            cu_seqlens = torch.arange(
                0, (batch_size + 1) * seq_len, step=seq_len,
                dtype=torch.int32, device=qkv.device
            )
        else:
            cu_seqlens = seqlens.cumsum(-1)
            
        x = flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, seq_len, 0., causal=False
        )
        x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
        x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)

        # MLP operation
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)), 
            None, gate_mlp, x, self.dropout
        )
        return x


class EmbeddingLayer(nn.Module):
    """Token embedding layer with signal embedding."""
    
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        self.signal_embedding = nn.Linear(2, dim)
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x, y):
        vocab_embed = self.embedding[x]
        signal_embed = self.signal_embedding(y.to(torch.float32))
        return torch.add(vocab_embed, signal_embed[:, None, :])


class DDitFinalLayer(nn.Module):
    """Final output layer with adaptive layer norm."""
    
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SEDD(nn.Module):
    """Score-based Discrete Diffusion Model."""
    
    def __init__(self, config):
        super().__init__()

        if type(config) == dict:
            config = OmegaConf.create(config)

        self.config = config
        self.absorb = config.graph.type == "absorb"
        vocab_size = config.tokens + (1 if self.absorb else 0)
        num_classes = 4
        class_dropout_prob = 0.1

        # Model components
        self.vocab_embed = EmbeddingLayer(config.model.hidden_size, vocab_size)
        self.sigma_map = TimestepEmbedder(config.model.cond_dim)
        self.label_embed = LabelEmbedder(num_classes, config.model.cond_dim, class_dropout_prob)
        self.rotary_emb = rotary.Rotary(config.model.hidden_size // config.model.n_heads)

        self.blocks = nn.ModuleList([
            DDiTBlock(
                config.model.hidden_size, 
                config.model.n_heads, 
                config.model.cond_dim, 
                dropout=config.model.dropout
            ) for _ in range(config.model.n_blocks)
        ])

        self.output_layer = DDitFinalLayer(
            config.model.hidden_size, vocab_size, config.model.cond_dim
        )
        self.scale_by_sigma = config.model.scale_by_sigma

    def forward(self, indices, labels, train, sigma):
        x = self.vocab_embed(indices, labels)
        c = F.silu(self.sigma_map(sigma))
        rotary_cos_sin = self.rotary_emb(x)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, rotary_cos_sin, c, seqlens=None)
            x = self.output_layer(x, c)

        # Zero out logits for current token positions
        x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))
        return x


def load_model_local(ckpt_path: str, device: torch.device) -> Tuple[SEDD, object, object]:
    """Load model from local checkpoint."""
    cfg = OmegaConf.load("configs/transformer.yaml")
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    
    # Initialize model
    score_model = SEDD(cfg).to(device)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.training.ema)

    # Load checkpoint
    loaded_state = torch.load(ckpt_path, map_location=device)
    score_model.load_state_dict(loaded_state['model'])
    ema.load_state_dict(loaded_state['ema'])

    # # Load checkpoint
    # loaded_state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # # Check if checkpoint has expected structure
    # if isinstance(loaded_state, dict) and 'model' in loaded_state:
    #     # Standard format with 'model' and 'ema' keys
    #     score_model.load_state_dict(loaded_state['model'])
    #     if 'ema' in loaded_state:
    #         ema.load_state_dict(loaded_state['ema'])
    #     else:
    #         print("Warning: No EMA state found in checkpoint")
    # else:
    #     # Direct state_dict format (converted from Lightning without proper structure)
    #     print("Checkpoint appears to be in direct state_dict format")
    #     print(f"Available keys: {list(loaded_state.keys()) if isinstance(loaded_state, dict) else 'Not a dict'}")
    #     raise ValueError("Checkpoint format not supported. Please use convert_sedd_checkpoint.py to convert properly.")

    # Apply EMA weights
    ema.store(score_model.parameters())
    ema.copy_to(score_model.parameters())
    
    return score_model, graph, noise


def load_deepstarr_data(filepath: str, batch_size: int) -> Tuple[DataLoader, torch.Tensor]:
    """Load and prepare DeepSTARR test data."""
    data = h5py.File(filepath, 'r')
    X_test = torch.tensor(np.array(data['X_test']))
    y_test = torch.tensor(np.array(data['Y_test']))
    X_test = torch.argmax(X_test, dim=1)
    
    testing_ds = TensorDataset(X_test, y_test)
    test_loader = DataLoader(
        testing_ds, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    return test_loader, y_test


def generate_samples(model, graph, noise, test_loader, batch_size, steps, device):
    """Generate samples using the diffusion model."""
    val_pred_seq = []
    
    # Initialize sampling function
    sampling_fn = sampling.get_pc_sampler(
        graph, noise, (batch_size, 249), 'analytic', steps, device=device
    )

    for _, (batch, val_target) in enumerate(tqdm(test_loader, desc="Sampling")):
    # for _, (batch, val_target) in enumerate(test_loader):
        # Handle last batch with different size
        if batch.shape[0] != batch_size:
            sampling_fn = sampling.get_pc_sampler(
                graph, noise, (batch.shape[0], 249), 'analytic', steps, device=device
            )
        
        # Generate sample
        sample = sampling_fn(model, val_target.to(device))
        seq_pred_one_hot = F.one_hot(sample, num_classes=4).float()
        val_pred_seq.append(seq_pred_one_hot)

    return torch.cat(val_pred_seq, dim=0)


def evaluate_samples(deepstarr, val_pred_seqs, device):
    """Evaluate generated samples against DeepSTARR oracle."""
    val_score = deepstarr.predict_custom(deepstarr.X_test.to(device))
    val_pred_score = deepstarr.predict_custom(val_pred_seqs.permute(0, 2, 1).to(device))
    
    sp_mse = (val_score - val_pred_score) ** 2
    mean_sp_mse = torch.mean(sp_mse).cpu()
    
    return mean_sp_mse

def main():
    """Main function to run the sampling and evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Generate DNA sequences using diffusion model")
    parser.add_argument("--model_path", required=True, type=str, 
                       help="Path to the trained model checkpoint")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size for sampling")
    parser.add_argument("--steps", type=int, default=249, 
                       help="Number of diffusion sampling steps")
    parser.add_argument("--data_path", type=str, default="DeepSTARR_data.h5",
                       help="Path to DeepSTARR data file")
    parser.add_argument("--oracle_path", type=str, 
                       default="oracle_models/oracle_DeepSTARR_DeepSTARR_data.ckpt",
                       help="Path to DeepSTARR oracle model")
    args = parser.parse_args()

    # Setup device and load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading diffusion model...")
    model, graph, noise = load_model_local(args.model_path, device)
    
    # Load data and oracle model
    print("Loading DeepSTARR data...")
    test_loader, y_test = load_deepstarr_data(args.data_path, args.batch_size)
    
    print("Loading DeepSTARR oracle model...")
    deepstarr = PL_DeepSTARR.load_from_checkpoint(
        args.oracle_path, input_h5_file=args.data_path
    ).eval()

    # Generate samples
    print("Generating samples...")
    val_pred_seqs = generate_samples(
        model, graph, noise, test_loader, args.batch_size, args.steps, device
    )
    
    # Evaluate samples
    print("Evaluating samples...")
    mean_sp_mse = evaluate_samples(deepstarr, val_pred_seqs, device)
    
    # Save results
    output_dir = os.path.dirname(args.model_path)
    output_path = os.path.join(output_dir, "sample.npz")
    np.savez(output_path, sequences=val_pred_seqs.cpu().numpy())
    
    # Print results
    print(f"Test MSE: {mean_sp_mse:.6f}")
    print(f"Generated {val_pred_seqs.shape[0]} sequences")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()