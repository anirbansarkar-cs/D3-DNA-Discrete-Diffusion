import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

import os, h5py
import numpy as np
import copy
import re
import tqdm
from scipy import stats
import torch.utils.data as data_utils

class dilated_residual(pl.LightningModule):
    def __init__(self, filter_num, kernel_size, dilation_rate, dropout):
        super().__init__()
        layers = []
        layers.append(nn.Conv1d(filter_num, filter_num, kernel_size, padding='same'))
        layers.append(nn.BatchNorm1d(filter_num))
        for i in range(0, len(dilation_rate)):
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Conv1d(filter_num, filter_num, kernel_size, stride=1,
                                    padding='same', dilation=dilation_rate[i], bias=False))
            layers.append(nn.BatchNorm1d(filter_num))
        self.block = nn.Sequential(*layers)
        self.output_act = nn.ReLU()

    def forward(self, x):
        out = self.block(x)
        residual = torch.add(out, x)
        output = self.output_act(residual)
        return output

class mpra(nn.Module):
    def __init__(self, output_dim): #exp_num, lr,
        super().__init__()
        # self.lr = lr
        self.conv1 = nn.Sequential(*[
            nn.Conv1d(4, 196, 15, padding='same'),
            nn.BatchNorm1d(196),
            nn.ELU(),
            nn.Dropout(0.1)
        ])
        self.res1 = nn.Sequential(*[
            dilated_residual(196, 3, [1, 2, 4, 8], 0.1),
            nn.MaxPool1d(4),
            nn.Dropout(0.2)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv1d(196, 256, 5, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        ])
        self.res2 = nn.Sequential(*[
            dilated_residual(256, 3, [1, 2, 4], 0.1),
            nn.MaxPool1d(4),
            nn.Dropout(0.2)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv1d(256, 256, 3, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        ])
        self.res3 = nn.Sequential(*[
            dilated_residual(256, 3, [1], 0.1),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        ])

        self.activation = nn.ReLU()
        self.dropout4 = nn.Dropout(0.5)
        self.flatten = nn.Flatten()

        # Layer 6 (fully connected), constituent parts
        self.fc4 = nn.LazyLinear(512, bias=True)
        self.batchnorm4 = nn.BatchNorm1d(512)

        # Layer 5 (fully connected), constituent parts
        self.fc5 = nn.LazyLinear(256, bias=True)
        self.batchnorm5 = nn.BatchNorm1d(256)

        # Output layer (fully connected), constituent parts
        self.fc6 = nn.Linear(256, output_dim)

    def forward(self, x):
        nn = self.conv1(x)
        nn = self.res1(nn)

        # Layer 2
        nn = self.conv2(nn)
        nn = self.res2(nn)

        # Layer 3
        nn = self.conv3(nn)
        nn = self.res3(nn)

        # Layer 4
        cnn = self.flatten(nn)
        cnn = self.fc4(cnn)
        cnn = self.batchnorm4(cnn)
        cnn = self.activation(cnn)
        cnn = self.dropout4(cnn)

        # Layer 5
        cnn = self.fc5(cnn)
        cnn = self.batchnorm5(cnn)
        cnn = self.activation(cnn)
        cnn = self.dropout4(cnn)

        # Output layer
        y_pred = self.fc6(cnn)

        return y_pred


#################


from typing import Any
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
# import tqdm
import numpy as np
import random
import h5py
import os
from scipy import stats

import pytorch_lightning as pl
# from lightning.pytorch import loggers as pl_loggers
from pytorch_lightning import loggers as pl_loggers
# import deepstarr_model
# import deepstarr_model_with_init
import tqdm


# import torchsummary

def get_github_main_directory(reponame='DALdna'):
    currdir = os.getcwd()
    dir = ''
    for dirname in currdir.split('/'):
        dir += dirname + '/'
        if dirname == reponame: break
    return dir


def key_with_low(key_list, low):
    the_key = ''
    for key in key_list:
        if key.lower() == low: the_key = key
    return the_key


from filelock import FileLock


class PL_mpra(pl.LightningModule):
    def __init__(self,
                 batch_size=128,  # original: 128, #20, #50, #100, #128,
                 train_max_epochs=100,  # my would-be-choice: 50,
                 patience=10,  # 10, #100, #20, #patience=10,
                 min_delta=0.001,  # min_delta=0.001,
                 input_h5_file='mpra_tewhey.h5',  # Originally created as: cp Orig_DeepSTARR_1dim.h5 DeepSTARRdev.h5
                 lr=0.002,  # most likely: 0.001, #0.002 From Paper
                 initial_ds=True,

                 weight_decay=1e-6,
                 # 1e-6, #1e-6, #0.0, #1e-6, #Stage0 # WEIGHT DECAY: L2 penalty: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
                 min_lr=0.0,
                 # default when not present (configure_optimizer in evoaug_analysis_utils_AC.py)                                                     #DSRR
                 lr_patience=10,  # 1, #2 #,100, #10, #5
                 decay_factor=0.1,  # 0.0 #0.1, #Stage0

                 scale=0.005,  # 0.001 or 0.005 according to Chandana

                 initialization='kaiming_uniform',  # original: 'kaiming_normal', #AC
                 initialize_dense=False,
                 ):
        super().__init__()
        self.scale = scale
        self.model = mpra(output_dim=1)  # , initialization=initialization, initialize_dense=initialize_dense) #.to(device) #goodold
        self.name = 'mpra'
        # self.task_type='single_task_regression'
        self.metric_names = ['PCC', 'Spearman']
        self.initial_ds = initial_ds

        self.batch_size = batch_size
        self.train_max_epochs = train_max_epochs
        self.patience = patience
        self.lr = lr
        self.min_delta = min_delta  # for trainer, but accessible as an attribute if needed                                                     #DSRR
        self.weight_decay = weight_decay

        # ""
        self.min_lr = min_lr
        self.lr_patience = lr_patience
        self.decay_factor = decay_factor
        # ""

        self.input_h5_file = input_h5_file
        data = h5py.File(input_h5_file, 'r')
        if initial_ds:
            self.X_train = torch.tensor(np.array(data['x_train']).astype(np.float32)).permute(0,2,1)  # (402278, 4, 249)
            self.y_train = torch.tensor(np.array(data['y_train']).astype(np.float32))[:, 2].unsqueeze(1)
            self.X_test = torch.tensor(np.array(data['x_test']).astype(np.float32)).permute(0,2,1)
            self.y_test = torch.tensor(np.array(data['y_test']).astype(np.float32))[:, 2].unsqueeze(1)
            self.X_valid = torch.tensor(np.array(data['x_valid']).astype(np.float32)).permute(0,2,1)
            self.y_valid = torch.tensor(np.array(data['y_valid']).astype(np.float32))[:, 2].unsqueeze(1)                                                     #DSRR
            self.X_test2 = self.X_test
            self.y_test2 = self.y_test
        else:
            self.X_train = data['x_train']
            self.y_train = data['y_train']
            self.X_test = data['x_test']
            self.y_test = data['y_test']
            self.X_test2 = data['x_test2']
            self.y_test2 = data['y_test2']
            self.X_valid = data['x_valid']
            self.y_valid = data['y_valid']

    # ""
    def training_step(self, batch, batch_idx):  # QUIQUIURG
        self.model.train()
        inputs, labels = batch
        loss_fn = nn.MSELoss()  # .to(device)
        outputs = self.model(inputs)
        loss = loss_fn(outputs, labels)  # DSRR

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)  # DSRR

        return loss

    # ""

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.lr_patience,
                                                         min_lr=self.min_lr, factor=self.decay_factor)  # DSRR
        # return optimizer
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        inputs, labels = batch
        loss_fn = nn.MSELoss()  # .to(device)
        outputs = self.model(inputs)
        loss = loss_fn(outputs, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)  # DSRR
        out_cpu = outputs.detach().cpu()
        lab_cpu = labels.detach().cpu()
        pcc = torch.tensor(self.metrics(out_cpu, lab_cpu)['PCC'].mean())
        self.log("val_pcc", pcc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # return loss

    def test_step(self, batch, batch_idx):
        self.model.eval()
        inputs, labels = batch
        loss_fn = nn.MSELoss()  # .to(device)
        outputs = self.model(inputs)
        loss = loss_fn(outputs, labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)  # DSRR
        # return loss

    def metrics(self, y_score, y_true):
        vals = []
        for output_index in range(y_score.shape[1]):
            vals.append(stats.spearmanr(y_true[:, output_index], y_score[:, output_index])[0])  # DSRR
        spearmanr_vals = np.array(vals)
        #
        vals = []
        for output_index in range(y_score.shape[-1]):
            vals.append(stats.pearsonr(y_true[:, output_index], y_score[:, output_index])[0])  # DSRR
        pearsonr_vals = np.array(vals)
        metrics = {'Spearman': spearmanr_vals, 'PCC': pearsonr_vals}
        return metrics

    def forward(self, x):  # DSRR
        return self.model(x)

    def predict_custom(self, X, keepgrad=False):
        self.model.eval()
        dataloader = torch.utils.data.DataLoader(X, batch_size=self.batch_size, shuffle=False)  # DSRR
        preds = torch.empty(0)
        if keepgrad:
            preds = preds.to(self.device)
        else:
            preds = preds.cpu()

        for x in tqdm.tqdm(dataloader, total=len(dataloader)):
            pred = self.model(x)
            if not keepgrad: pred = pred.detach().cpu()
            preds = torch.cat((preds, pred), axis=0)
        return preds

    def predict_custom_mcdropout(self, X, seed=41, keepgrad=False):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)  # DSRR
        self.model.eval()
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
        #
        dataloader = torch.utils.data.DataLoader(X, batch_size=self.batch_size, shuffle=False)
        preds = torch.empty(0)
        if keepgrad:
            preds = preds.to(self.device)
        else:
            preds = preds.cpu()

        for x in tqdm.tqdm(dataloader, total=len(dataloader)):
            pred = self.model(x)
            if not keepgrad: pred = pred.detach().cpu()
            preds = torch.cat((preds, pred), axis=0)

        return preds


###########################################


def training_with_PL(chosen_model, chosen_dataset,
                     initial_test=False, mcdropout_test=False, verbose=False, wanted_wandb=False):
    if wanted_wandb:
        import wandb
        from pytorch_lightning.loggers import WandbLogger  # https://docs.wandb.ai/guides/integrations/lightning
        wandb_logger = WandbLogger(log_model="all")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device=} {torch.cuda.is_available()=}")

    currdir = os.popen('pwd').read().replace("\n", "")  # os.getcwd()
    outdir = "../outputs/"  # ../../outputs_DALdna/"
    log_dir = outdir + "lightning_logs_" + chosen_model + "/"
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir)

    if wanted_wandb:
        logger_of_choice = wandb_logger
    else:
        logger_of_choice = tb_logger

    # model=eval("PL_"+chosen_model+"(input_h5_file='../inputs/"+chosen_dataset+".h5', initial_ds=True)") #PERFECT OLD
    model = eval("PL_" + chosen_model + "(input_h5_file='./" + chosen_dataset + ".h5', initial_ds=True)")  # PERFECT OLD

    os.system('date')
    print(model.X_train.shape)
    print(model.y_train.shape)
    train_dataloader = torch.utils.data.DataLoader(list(zip(model.X_train, model.y_train)), batch_size=model.batch_size,
                                                   shuffle=True)
    os.system('date')
    valid_dataloader = torch.utils.data.DataLoader(list(zip(model.X_valid, model.y_valid)), batch_size=model.batch_size,
                                                   shuffle=False)  # True)
    os.system('date')
    test_dataloader = torch.utils.data.DataLoader(list(zip(model.X_test, model.y_test)), batch_size=model.batch_size,
                                                  shuffle=False)  # True)
    os.system('date')

    ckptfile = "oracle_" + model.name + "_" + chosen_dataset + "_2"  # +".ckpt"
    to_monitor = 'val_loss'
    callback_ckpt = pl.callbacks.ModelCheckpoint(
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html
        # gpus=1,
        # auto_select_gpus=True,
        monitor=to_monitor,  # default is None which saves a checkpoint only for the last epoch.
        mode='min',
        save_top_k=1,
        save_weights_only=True,
        dirpath="./",  # "../inputs/", #get_github_main_directory(reponame='DALdna')+"inputs/",
        filename=ckptfile,  # comment out to verify that a different epoch is picked in the name.
    )
    early_stop_callback = pl.callbacks.EarlyStopping(
        # monitor='val_loss',
        monitor=to_monitor,
        min_delta=model.min_delta,
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html
        patience=model.patience,
        verbose=False,
        mode='min'
    )

    if initial_test:
        print('predict_custom')
        y_score = model.predict_custom(model.X_test)
        print('y_score.shape: ', y_score.shape)
        metrics_pretrain = model.metrics(y_score, model.y_test)
        print(f"{metrics_pretrain=}")
        print(f"{model(model.X_test[0:10])=}")

    if mcdropout_test:
        n_mc = 5
        preds_mc = torch.zeros((n_mc, len(model.X_test)))
        for i in range(n_mc):
            preds_mc[i] = model.predict_custom_mcdropout(model.X_test,
                                                         seed=41 + i).squeeze(axis=1).unsqueeze(axis=0)
        print('predict_custom_mcdropout')
        print('y_score.shape: ', preds_mc.shape)
        metrics_pretrain = model.metrics(y_score, model.y_test)
        print(f"{metrics_pretrain=}")
        print(f"{model(model.X_test[0:10])=}")

    print(f"{model.device=}")
    trainer = pl.Trainer(accelerator='cuda', devices=-1, max_epochs=model.train_max_epochs, logger=logger_of_choice,
                         callbacks=[callback_ckpt, early_stop_callback], deterministic=True)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)  # GOODOLD
    # os.system('mv ../inputs/'+ckptfile+'-v1.ckpt ../inputs/'+ckptfile+'.ckpt')
    os.system('mv ./' + ckptfile + '-v1.ckpt ./' + ckptfile + '.ckpt')
    if verbose: os.system('date')
    y_score = model.predict_custom(model.X_test)
    if verbose: os.system('date')
    metrics = model.metrics(y_score, model.y_test)
    print(metrics) #{'Spearman': array([0.75220946, 0.77259962, 0.77856478]), 'PCC': array([0.83569508, 0.83646114, 0.83141418])}
    #{'Spearman': array([0.75037962]), 'PCC': array([0.83507294])} for 0
    #{'Spearman': array([0.77897868]), 'PCC': array([0.8383449])} for 1
    #{'Spearman': array([0.77229597]), 'PCC': array([0.82239165])} for 2
    """
    if wanted_wandb:
        wandb.log(metrics)
    """

    print(ckptfile)
    return metrics


##############################################################


if __name__ == '__main__':

    pairlist = [
        ['mpra', 'mpra_data'],

    ]

    for pair in pairlist:
        chosen_model = pair[0]
        chosen_dataset = pair[1]

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        overall_seed = 41
        myseed = overall_seed
        torch.manual_seed(myseed)
        random.seed(myseed)
        np.random.seed(myseed)

        import logging

        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

        # metrics=training_with_PL(chosen_model, chosen_dataset, initial_test=False, mcdropout_test=False, verbose=False, wanted_wandb=True)
        metrics = training_with_PL(chosen_model, chosen_dataset, initial_test=True, mcdropout_test=False, verbose=False,
                                   wanted_wandb=False)

        print("SCRIPT END")
        print("WARNING: should I do a deep ensemble, and then take the ckpt of the best model?")
