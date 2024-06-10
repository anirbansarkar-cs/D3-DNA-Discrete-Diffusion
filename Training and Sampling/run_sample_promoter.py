import torch
import argparse
import sys

import data
from load_model import load_model_local
import torch.nn.functional as F
import sampling
import h5py, os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

from sei import Sei
from selene_sdk.utils import NonStrandSpecific

def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="", type=str)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1024) #Sequence length
    args = parser.parse_args()

    device = torch.device('cuda')
    model, graph, noise = load_model_local(args.model_path, device)

    seifeatures = pd.read_csv('promoter_design/target.sei.names', sep='|', header=None)
    # print (seifeatures)
    def get_sei_profile(seq_one_hot, device=torch.device('cpu')):
        B, L, K = seq_one_hot.shape
        seq_one_hot = seq_one_hot.cpu()
        sei_inp = torch.cat([torch.ones((B, 4, 1536)) * 0.25,
                             seq_one_hot.transpose(1, 2),
                             torch.ones((B, 4, 1536)) * 0.25], 2).to(device)  # batchsize x 4 x 4,096
        sei_out = sei(sei_inp).cpu().detach().numpy()  # batchsize x 21,907
        sei_out = sei_out[:, seifeatures[1].str.strip().values == 'H3K4me3']  # batchsize x 2,350
        predh3k4me3 = sei_out.mean(axis=1)  # batchsize
        return predh3k4me3

    # Promoter
    # SEI
    sei = NonStrandSpecific(Sei(4096, 21907))
    sei.load_state_dict(upgrade_state_dict(
        torch.load('promoter_design/best.sei.model.pth.tar', map_location='cpu')['state_dict'],
        prefixes=['module.']))
    sei.to(device)

    test_ds = PromoterDataset(n_tsses=100000, rand_offset=0, split='test')
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                             num_workers=4)

    all_sp_mse = []
    val_pred_seq = []
    sampling_fn = sampling.get_pc_sampler(
        graph, noise, (batch.shape[0], 1024), 'analytic', args.steps, device=device
    )

    for _, (batch) in enumerate(test_loader):
        seq_one_hot = batch[:, :, :4]
        target = batch[:, :, 4:5]
        sei_profile = get_sei_profile(seq_one_hot, device)

        sample = sampling_fn(model, target.to(device))
        seq_pred_one_hot = F.one_hot(sample, num_classes=4).float()
        val_pred_seq.append(seq_pred_one_hot)
        sei_profile_pred = get_sei_profile(seq_pred_one_hot, device)
        sp_mse = (sei_profile - sei_profile_pred) ** 2
        all_sp_mse.append(np.mean(sp_mse))

    val_pred_seqs = torch.cat(val_pred_seq, dim=0)
    mean_sp_mse = np.mean(all_sp_mse)
    print(f"Test-sp-mse {mean_sp_mse}")
    np.savez(os.path.join(args.model_path, f"sample_{rank}.npz", ), val_pred_seqs.cpu())

if __name__ == "__main__":
    main()