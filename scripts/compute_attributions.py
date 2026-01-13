import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
from model_zoo.deepstarr.deepstarr import PL_DeepSTARR
import h5py
import argparse
from tangermeme.deep_lift_shap import deep_lift_shap 


def load_deepstarr(ckpt, h5_config, device):
    # always load ckpt on cpu
    model = PL_DeepSTARR.load_from_checkpoint(ckpt, input_h5_file=h5_config, map_location='cpu', weights_only=False) # required for older checkpoints
    model.to(device) # then switch it to gpu
    model.eval()

    return model, device


def load_samples(h5_samples, device) -> np.ndarray:
    with h5py.File(h5_samples, 'r') as f:
        samples = f['sequences_onehot'][:]

    return torch.from_numpy(samples).float().to(device)


def main():
    ap = argparse.ArgumentParser(description='Compute attribution matrix for one-hot encoded samples with deep_lift_shap + gradient correction')
    ap.add_argument('--h5-samples', required=True)
    ap.add_argument('--h5-config', required=True)
    ap.add_argument('--deepstarr-ckpt', required=True)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--output-file', required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model, device = load_deepstarr(args.deepstarr_ckpt, args.h5_config, device)

    samples = load_samples(args.h5_samples, device)
    # samples, seq len, timesteps, alpahbet len
    N, L, T, A = samples.shape
    # reshape for compatibility with tangermeme
    samples = samples.permute(0,3,1,2)

    # pre allocate output
    attributions_hk = []
    attributions_dev = []

    # loop over timesteps and samples
    for t in range(T):
        for n in range(N):   
            # each attribution map for a different (scalar) target
            attr_dev = deep_lift_shap(model, samples[n:n+1, : , :, t], device=str(device), verbose=True, target=0)
            # now gradient correction
            ch_dev = attr_dev.detach().cpu().numpy().astype(np.float32)
            ch_dev = ch_dev - ch_dev.mean(axis=1, keepdims=True)  # subtract mean across bases
            attributions_dev.append(ch_dev)


            attr_hk = deep_lift_shap(model, samples[n:n+1, : , :, t], device=str(device), verbose=True, target=1)
            # now gradient correction
            ch_hk = attr_hk.contiguous().detach().cpu().numpy().astype(np.float32)
            ch_hk = ch_hk - ch_hk.mean(axis=1, keepdims=True)
            attributions_hk.append(ch_hk)

    # stack into final arrays
    attributions_dev = np.concatenate(attributions_dev, axis=0)
    attributions_hk = np.concatenate(attributions_hk, axis=0)

    # save to HDF5
    with h5py.File(args.output_file, 'w') as f:
        f.create_dataset('attributions_dev', data=attributions_dev, compression="gzip")
        f.create_dataset('attributions_hk', data=attributions_hk, compression="gzip")


    print(f"Saved attributions to {args.output_file}")


if __name__ == "__main__":
    main()