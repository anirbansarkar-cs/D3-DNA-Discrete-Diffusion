import h5py
import torch
import numpy as np
from scipy import stats
import pytorch_lightning as pl
from model_zoo.deepstarr.deepstarr import DeepSTARR
from utils.dinuc_shuffle import dinuc_shuffle
import argparse


class PL_DeepSTARR(pl.LightningModule):
    def __init__(
        self,
        input_h5_file: str,
        batch_size: int = 128,
        mode: str = "motif",  # "motif" | "non_motif"
    ):
        super().__init__()

        self.batch_size = batch_size
        self.model = DeepSTARR(output_dim=2)
        self.input_h5_file = input_h5_file
        self.mode = mode

        with h5py.File(self.input_h5_file, "r") as f:
            x = f["sequences"][:]        # (S, I, L)
            y = f["Y_target"][:]
            start_dev = f["start_dev"][:]
            end_dev = f["end_dev"][:]
            start_hk = f["start_hk"][:]
            end_hk = f["end_hk"][:]

            if "sequences_onehot" in f:
                onehot = f["sequences_onehot"][:]
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
            if not np.isnan(start_dev[i]):
                s, e = int(start_dev[i]), int(end_dev[i])
            elif not np.isnan(start_hk[i]):
                s, e = int(start_hk[i]), int(end_hk[i])
            else:
                continue

            if self.mode == "motif":
                seg = slice(s, e + 1)
            else:
                left = slice(0, s)
                right = slice(e + 1, L)

                if left.stop - left.start > 1:
                    onehot_dinuc[i, left] = dinuc_shuffle(onehot[i, left])
                if right.stop - right.start > 1:
                    onehot_dinuc[i, right] = dinuc_shuffle(onehot[i, right])
                continue

            if seg.stop - seg.start > 1:
                onehot_dinuc[i, seg] = dinuc_shuffle(onehot[i, seg])

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

    def metrics(self, y_score, y_true):
        return {
            "Spearman": np.array([stats.spearmanr(y_true[:, i], y_score[:, i])[0]
                                  for i in range(y_score.shape[1])]),
            "PCC": np.array([stats.pearsonr(y_true[:, i], y_score[:, i])[0]
                             for i in range(y_score.shape[1])]),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--mode", choices=["motif", "non_motif"], default="motif")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PL_DeepSTARR(args.data_path, mode=args.mode).to(device)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)

    y_orig = model.predict(model.X_original)
    y_dinuc = model.predict(model.X_dinuc)

    metrics_orig = model.metrics(y_orig.numpy(), model.y_test.numpy())
    metrics_dinuc = model.metrics(y_dinuc.numpy(), model.y_test.numpy())

    print("Original Mean PCC:", metrics_orig["PCC"].mean())
    print("Dinuc Mean PCC:", metrics_dinuc["PCC"].mean())

    with h5py.File(args.data_path, "a") as f:
        for name, arr in [
            ("y_original_pred", y_orig),
            ("y_dinuc_pred", y_dinuc),
        ]:
            if name in f:
                del f[name]
            f.create_dataset(
                name,
                data=arr.numpy().reshape(
                    model.num_samples, model.num_iterations, -1
                ),
            )
