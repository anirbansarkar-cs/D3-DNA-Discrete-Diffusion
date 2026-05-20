"""
DeepSTARR attribution template.

Two modes in one file (use this as a starting point for your own runs):

  oracle-check
      Load a DeepSTARR oracle checkpoint and an H5 of inpainted/test
      sequences with motif span annotations, predict on the originals and
      on dinucleotide-shuffled versions of the motif (or flanks), and
      report Spearman / PCC. Useful for sanity-checking the oracle and
      writing dinuc-shuffled one-hots back into the H5 for downstream use.

  attributions
      Compute per-base attributions for one-hot encoded samples with
      DeepLIFT-SHAP and a gradient-correction step (subtract per-position
      mean across bases). Iterates over diffusion timesteps and writes
      `attributions_dev` and `attributions_hk` to an output H5.

------------------------------------------------------------------------
NOTE - TangerMeme dependency
------------------------------------------------------------------------
The `attributions` mode depends on a fix authored by the maintainer of
this repo (@aduranu). The fix is currently an OPEN pull request against
upstream TangerMeme - it is NOT yet merged. Until the PR lands, the
released TangerMeme package on PyPI does not contain this patch and
`deep_lift_shap` will not run correctly here. To use this script you
must install TangerMeme from the PR branch (the local fork lives at
`~/tangermeme-worktrees/add-hook-stacks`), for example:

    pip install -e ~/tangermeme-worktrees/add-hook-stacks

Re-point this note once the upstream PR is merged and released.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import h5py
import numpy as np
import torch
import pytorch_lightning as pl
from scipy import stats

from model_zoo.deepstarr.deepstarr import DeepSTARR, PL_DeepSTARR
from utils.dinuc_shuffle import dinuc_shuffle


# ---------------------------------------------------------------------------
# oracle-check: predict on originals vs dinuc-shuffled segments
# ---------------------------------------------------------------------------

def _dinuc_safe(seq):
    if seq.ndim != 2:
        return seq

    transposed = False
    if seq.shape[0] == 4:
        seq = seq.T
        transposed = True

    if seq.shape[0] > 1:
        seq = dinuc_shuffle(seq)

    if transposed:
        seq = seq.T
    return seq


class DeepSTARROracleCheck(pl.LightningModule):
    def __init__(self, input_h5_file: str, batch_size: int = 128, mode: str = "motif"):
        super().__init__()
        self.batch_size = batch_size
        self.model = DeepSTARR(output_dim=2)
        self.input_h5_file = input_h5_file
        self.mode = mode

        with h5py.File(self.input_h5_file, "r") as f:
            x = f["sequences"][:]
            y = f["Y_target"][:]
            start_dev = f["start_dev"][:]
            end_dev = f["end_dev"][:]
            start_hk = f["start_hk"][:]
            end_hk = f["end_hk"][:]

            if "sequences_onehot" in f:
                onehot = np.squeeze(f["sequences_onehot"][:])
            else:
                x_flat = x.reshape(-1, x.shape[-1])
                onehot = np.eye(4, dtype=np.float32)[x_flat.astype(np.int64)]

        S, I, L = x.shape
        self.num_samples = S
        self.num_iterations = I
        self.seq_length = L

        y = np.repeat(y, repeats=I, axis=0)
        onehot_dinuc = onehot.copy()

        for i in range(onehot.shape[0]):
            sample_idx = i // I
            if not np.isnan(start_dev[sample_idx]):
                s, e = int(start_dev[sample_idx]), int(end_dev[sample_idx])
            elif not np.isnan(start_hk[sample_idx]):
                s, e = int(start_hk[sample_idx]), int(end_hk[sample_idx])
            else:
                continue

            if mode == "motif":
                seg = slice(s, e + 1)
                if seg.stop - seg.start > 1:
                    onehot_dinuc[i, seg] = _dinuc_safe(onehot[i, seg])
            elif mode == "non_motif":
                if s > 1:
                    onehot_dinuc[i, :s] = _dinuc_safe(onehot[i, :s])
                if e + 1 < L - 1:
                    onehot_dinuc[i, e + 1:] = _dinuc_safe(onehot[i, e + 1:])
            else:
                raise ValueError(f"Unknown mode: {mode}")

        with h5py.File(self.input_h5_file, "a") as f:
            if "sequences_onehot" not in f:
                f.create_dataset("sequences_onehot", data=onehot)
            if "sequences_onehot_dinuc" in f:
                del f["sequences_onehot_dinuc"]
            f.create_dataset("sequences_onehot_dinuc", data=onehot_dinuc)

        self.X_original = torch.tensor(onehot, dtype=torch.float32).permute(0, 2, 1)
        self.X_dinuc = torch.tensor(onehot_dinuc, dtype=torch.float32).permute(0, 2, 1)
        self.y_test = torch.tensor(y, dtype=torch.float32)

    @torch.no_grad()
    def predict(self, X):
        self.eval()
        loader = torch.utils.data.DataLoader(X, batch_size=self.batch_size, shuffle=False)
        preds = []
        for xb in loader:
            preds.append(self.model(xb.to(self.device)).cpu())
        return torch.cat(preds, dim=0)

    @staticmethod
    def metrics(y_score, y_true):
        return {
            "Spearman": np.array([stats.spearmanr(y_true[:, i], y_score[:, i])[0]
                                  for i in range(y_score.shape[1])]),
            "PCC": np.array([stats.pearsonr(y_true[:, i], y_score[:, i])[0]
                             for i in range(y_score.shape[1])]),
        }


def run_oracle_check(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepSTARROracleCheck(args.data_path, mode=args.mode).to(device)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)

    y_orig = model.predict(model.X_original)
    y_dinuc = model.predict(model.X_dinuc)

    metrics_orig = model.metrics(y_orig.numpy(), model.y_test.numpy())
    metrics_dinuc = model.metrics(y_dinuc.numpy(), model.y_test.numpy())

    print("Original Mean PCC:", metrics_orig["PCC"].mean())
    print("Dinuc Mean PCC:", metrics_dinuc["PCC"].mean())

    with h5py.File(args.data_path, "a") as f:
        for name, arr in [("y_original_pred", y_orig), ("y_dinuc_pred", y_dinuc)]:
            if name in f:
                del f[name]
            f.create_dataset(
                name,
                data=arr.numpy().reshape(model.num_samples, model.num_iterations, -1),
            )


# ---------------------------------------------------------------------------
# attributions: DeepLIFT-SHAP with gradient correction
# ---------------------------------------------------------------------------

def _load_deepstarr_for_attr(ckpt, h5_config, device):
    model = PL_DeepSTARR.load_from_checkpoint(
        ckpt,
        input_h5_file=h5_config,
        map_location="cpu",
        weights_only=False,  # required for older checkpoints
    )
    model.to(device)
    model.eval()
    return model


def _load_samples(h5_samples, device):
    with h5py.File(h5_samples, "r") as f:
        samples = f["sequences_onehot"][:]
    return torch.from_numpy(samples).float().to(device)


def run_attributions(args):
    # Import here so oracle-check mode doesn't pay the import cost or
    # require TangerMeme to be installed.
    from tangermeme.deep_lift_shap import deep_lift_shap

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_deepstarr_for_attr(args.deepstarr_ckpt, args.h5_config, device)

    samples = _load_samples(args.h5_samples, device)
    # (N, L, T, A) -> (N, A, L, T) for TangerMeme
    N, L, T, A = samples.shape
    samples = samples.permute(0, 3, 1, 2)

    attributions_dev = []
    attributions_hk = []

    for t in range(T):
        for n in range(N):
            attr_dev = deep_lift_shap(
                model, samples[n:n + 1, :, :, t],
                device=str(device), verbose=True, target=0,
            )
            ch_dev = attr_dev.detach().cpu().numpy().astype(np.float32)
            ch_dev = ch_dev - ch_dev.mean(axis=1, keepdims=True)
            attributions_dev.append(ch_dev)

            attr_hk = deep_lift_shap(
                model, samples[n:n + 1, :, :, t],
                device=str(device), verbose=True, target=1,
            )
            ch_hk = attr_hk.contiguous().detach().cpu().numpy().astype(np.float32)
            ch_hk = ch_hk - ch_hk.mean(axis=1, keepdims=True)
            attributions_hk.append(ch_hk)

    attributions_dev = np.concatenate(attributions_dev, axis=0)
    attributions_hk = np.concatenate(attributions_hk, axis=0)

    with h5py.File(args.output_file, "w") as f:
        f.create_dataset("attributions_dev", data=attributions_dev, compression="gzip")
        f.create_dataset("attributions_hk", data=attributions_hk, compression="gzip")

    print(f"Saved attributions to {args.output_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="DeepSTARR attribution template (oracle-check + deep_lift_shap attributions).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    oc = sub.add_parser("oracle-check", help="Predict on originals vs dinuc-shuffled segments.")
    oc.add_argument("--data_path", type=str, required=True)
    oc.add_argument("--ckpt_path", type=str, required=True)
    oc.add_argument("--mode", choices=["motif", "non_motif"], default="non_motif")
    oc.set_defaults(func=run_oracle_check)

    at = sub.add_parser("attributions", help="DeepLIFT-SHAP attributions (requires patched TangerMeme).")
    at.add_argument("--h5-samples", required=True)
    at.add_argument("--h5-config", required=True)
    at.add_argument("--deepstarr-ckpt", required=True)
    at.add_argument("--batch-size", type=int, default=128)
    at.add_argument("--output-file", required=True)
    at.set_defaults(func=run_attributions)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
