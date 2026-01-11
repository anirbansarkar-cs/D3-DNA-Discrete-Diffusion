import os
import h5py
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
from scipy import stats
import tqdm
import pytorch_lightning as pl
from model_zoo.deepstarr.deepstarr import DeepSTARR
import argparse

class PL_DeepSTARR(pl.LightningModule):
    """Inference-only DeepSTARR wrapper."""

    def __init__(self, input_h5_file: str, batch_size: int = 128):
        super().__init__()

        self.batch_size = batch_size
        self.model = DeepSTARR(output_dim=2)
        self.input_h5_file = input_h5_file

        # -------------------------
        # Load + preprocess test data
        # -------------------------
        with h5py.File(self.input_h5_file, "r") as f:
            x = f["sequences"][:]               # (samples,iterations, length)
            y = f["Y_target"][:]

        print("shape before flattening", x.shape)

        # flatten (samples, iterations, length) -> (samples*iterations, length)
        num_samples, num_iterations, seq_length = x.shape
        x = x.reshape(-1, seq_length)
        print("shape after flattening", x.shape)

        # repeat y to match flattened sequences (one label per original sample -> one per iteration)
        if len(y.shape) >= 1 and y.shape[0] == num_samples:
            y = np.repeat(y, repeats=num_iterations, axis=0)

        print("shape of labels after repeat:", y.shape)

        # One-hot encode DNA from index encoding (0=A, 1=C, 2=G, 3=T)
        # x has shape (batch, seq_len) with values in [0, 1, 2, 3]
        onehot = np.eye(4, dtype=np.float32)[x.astype(np.int64)]
        print("shape after one-hot encoding", onehot.shape)

        # tensors - transpose to (batch, 4, seq_len) for DeepSTARR conv1d
        self.X_test = torch.tensor(onehot, dtype=torch.float32).permute(0, 2, 1)
        print("shape after transpose for conv1d", self.X_test.shape)
        self.y_test = torch.tensor(y, dtype=torch.float32)

    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def predict(self):
        self.eval()
        loader = torch.utils.data.DataLoader(
            self.X_test, batch_size=self.batch_size, shuffle=False
        )

        preds = []
        for xb in loader:
            xb = xb.to(self.device)
            preds.append(self.model(xb).cpu())

        return torch.cat(preds, dim=0)

    def metrics(self, y_score, y_true):
        spearman = []
        pearson = []

        for i in range(y_score.shape[1]):
            spearman.append(
                stats.spearmanr(y_true[:, i], y_score[:, i])[0]
            )
            pearson.append(
                stats.pearsonr(y_true[:, i], y_score[:, i])[0]
            )

        return {
            "Spearman": np.array(spearman),
            "PCC": np.array(pearson),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arg parser for fast deepstarr inference")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./DeepSTARR_data.h5",
        help="Path to the HDF5 data file containing sequences and labels"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/grid/koo/home/shared/d3/oracle_weights/deepstarr/oracle_DeepSTARR_DeepSTARR_data.ckpt",
        help="Path to the pre-trained checkpoint (.ckpt)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PL_DeepSTARR(input_h5_file=args.data_path)
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    model = model.to(device)
    model.eval()

    print("Running inference...")
    y_pred = model.predict()
    y_true = model.y_test

    metrics = model.metrics(y_pred.numpy(), y_true.numpy())

    print("Pearson:", metrics["PCC"])
    print("Mean Pearson:", metrics["PCC"].mean())
    print("Spearman:", metrics["Spearman"])
    print("Mean Spearman:", metrics["Spearman"].mean())
