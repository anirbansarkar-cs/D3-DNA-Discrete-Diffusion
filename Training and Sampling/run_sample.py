import torch
import argparse
import sys

import data
from load_model import load_model_local
import torch.nn.functional as F
import sampling
import h5py, os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

from PL_DeepSTARR import * #Required for DeepSTARR
from PL_mpra import * #Required for MPRA


def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_path", default="", type=str) #Need to provide model path
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=249) #Sequence length #Change default to 200 for MPRA
    args = parser.parse_args()

    
    device = torch.device('cuda')
    model, graph, noise = load_model_local(args.model_path, device)

    #DeepSTARR. Comment the below lines and uncomment the following ones for MPRA
    ##########
    filepath = os.path.join('DeepSTARR_data.h5') #load DeepSTARR data
    data = h5py.File(filepath, 'r')
    ckpt_aug_path = os.path.join('oracle_DeepSTARR_DeepSTARR_data.ckpt') #Load DeepSTARR oracle model

    #We select test data to calculate MSE and generate samples. Change if required
    X_test = torch.tensor(np.array(data['X_test']))
    y_test = torch.tensor(np.array(data['Y_test']))
    X_test = torch.argmax(X_test, dim=1)
    testing_ds = TensorDataset(X_test, y_test)
    test_ds = torch.utils.data.DataLoader(testing_ds, batch_size=args.batch_size, shuffle=False,
                                             num_workers=4)
    deepstarr = PL_DeepSTARR.load_from_checkpoint(ckpt_aug_path, input_h5_file=filepath).eval()

    val_pred_seq = []
    sampling_fn = sampling.get_pc_sampler(
            graph, noise, (batch.shape[0], 249), 'analytic', args.steps, device=device
        )
    ##########

    #MPRA
    #########
    # filepath = os.path.join('mpra_data.h5') #load DeepSTARR data
    # data = h5py.File(filepath, 'r')
    # ckpt_aug_path = os.path.join('oracle_mpra_mpra_data.ckpt') #Load MPRA oracle model

    # #We select test data to calculate MSE and generate samples. Change if required
    # X_test = torch.tensor(np.array(data['x_test']).astype(np.float32)).permute(0,2,1)
    # y_test = torch.tensor(np.array(data['y_test']).astype(np.float32))
    # X_test = torch.argmax(X_test, dim=1)
    # testing_ds = TensorDataset(X_test, y_test)
    # test_ds = torch.utils.data.DataLoader(testing_ds, batch_size=args.batch_size, shuffle=False,
    #                                          num_workers=4)
    # mpra = PL_mpra.load_from_checkpoint(ckpt_aug_path, input_h5_file=filepath).eval()

    # val_pred_seq = []
    # sampling_fn = sampling.get_pc_sampler(
    #         graph, noise, (batch.shape[0], 200), 'analytic', args.steps, device=device
    #     )
    #########
    
    for _, (batch, val_target) in enumerate(test_ds):
        sample = sampling_fn(model, val_target.to(device))
        seq_pred_one_hot = F.one_hot(sample, num_classes=4).float()
        val_pred_seq.append(seq_pred_one_hot)

    val_pred_seqs = torch.cat(val_pred_seq, dim=0)
    
    #Below two lines are for DeepSTARR
    val_score = deepstarr.predict_custom(deepstarr.X_test.to(device))
    val_pred_score = deepstarr.predict_custom(val_pred_seqs.permute(0, 2, 1).to(device))

    #Below two lines are for MPRA
    # val_score = mpra.predict_custom(mpra.X_test.to(device))
    # val_pred_score = mpra.predict_custom(val_pred_seqs.permute(0, 2, 1).to(device))
    
    sp_mse = (val_score - val_pred_score) ** 2
    mean_sp_mse = torch.mean(sp_mse).cpu()
    print(f"Test-sp-mse {mean_sp_mse}")
    np.savez(os.path.join(args.model_path, f"sample_{rank}.npz", ), val_pred_seqs.cpu())

if __name__=="__main__":
    main()
