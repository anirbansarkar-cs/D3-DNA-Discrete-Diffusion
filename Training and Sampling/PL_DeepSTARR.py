import torch
import torch.nn as nn
import torch.nn.functional as F

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


#################


from typing import Any
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
#import tqdm
import numpy as np
import random
import h5py
import os
from scipy import stats

import pytorch_lightning as pl
#from lightning.pytorch import loggers as pl_loggers
from pytorch_lightning import loggers as pl_loggers
#import deepstarr_model 
#import deepstarr_model_with_init 
import tqdm
#import torchsummary

def get_github_main_directory(reponame='DALdna'):
    currdir=os.getcwd()
    dir=''
    for dirname in currdir.split('/'):
        dir+=dirname+'/'
        if dirname==reponame: break
    return dir

def key_with_low(key_list,low):
    the_key=''
    for key in key_list:
        if key.lower()==low: the_key=key
    return the_key

from filelock import FileLock 

class PL_DeepSTARR(pl.LightningModule):
    def __init__(self,
                 batch_size=128, #original: 128, #20, #50, #100, #128,
                 train_max_epochs=100, #my would-be-choice: 50,
                 patience=10, #10, #100, #20, #patience=10,
                 min_delta=0.001, #min_delta=0.001,
                 input_h5_file='DeepSTARR_data.h5', #Originally created as: cp Orig_DeepSTARR_1dim.h5 DeepSTARRdev.h5
                 lr=0.002, #most likely: 0.001, #0.002 From Paper
                 initial_ds=True,

                 weight_decay=1e-6, #1e-6, #1e-6, #0.0, #1e-6, #Stage0 # WEIGHT DECAY: L2 penalty: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
                 min_lr=0.0, #default when not present (configure_optimizer in evoaug_analysis_utils_AC.py)                                                     #DSRR
                 lr_patience=10, #1, #2 #,100, #10, #5
                 decay_factor=0.1, #0.0 #0.1, #Stage0

                 scale=0.005, # 0.001 or 0.005 according to Chandana

                 initialization='kaiming_uniform', # original: 'kaiming_normal', #AC 
                 initialize_dense=False, 
                 ):
        super().__init__()
        self.scale=scale
        self.model=DeepSTARR(output_dim=2) #, initialization=initialization, initialize_dense=initialize_dense) #.to(device) #goodold
        self.name='DeepSTARR'
        # self.task_type='single_task_regression'
        self.metric_names=['PCC','Spearman']
        self.initial_ds=initial_ds

        self.batch_size=batch_size
        self.train_max_epochs=train_max_epochs
        self.patience=patience
        self.lr=lr
        self.min_delta=min_delta #for trainer, but accessible as an attribute if needed                                                     #DSRR
        self.weight_decay=weight_decay

        #""
        self.min_lr=min_lr 
        self.lr_patience=lr_patience 
        self.decay_factor=decay_factor 
        #""

        self.input_h5_file=input_h5_file
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

    #""
    def training_step(self, batch, batch_idx): #QUIQUIURG
        self.model.train() 
        inputs, labels = batch 
        loss_fn = nn.MSELoss() #.to(device)
        outputs=self.model(inputs)
        loss = loss_fn(outputs, labels)                                                     #DSRR

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)                                                     #DSRR

        return loss
    #""
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.lr_patience, min_lr=self.min_lr, factor=self.decay_factor)                                                     #DSRR
        #return optimizer
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler":scheduler, "monitor": "val_loss"}} 
    
    
    def validation_step(self, batch, batch_idx): 
        self.model.eval()
        inputs, labels = batch 
        loss_fn = nn.MSELoss() #.to(device)
        outputs=self.model(inputs)
        loss = loss_fn(outputs, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)                                                     #DSRR
        out_cpu=outputs.detach().cpu()
        lab_cpu=labels.detach().cpu()
        pcc=torch.tensor(self.metrics(out_cpu, lab_cpu)['PCC'].mean())
        self.log("val_pcc", pcc, on_step=False, on_epoch=True, prog_bar=True, logger=True)                                                     
        #return loss

    def test_step(self, batch, batch_idx): 
        self.model.eval()
        inputs, labels = batch 
        loss_fn = nn.MSELoss() #.to(device)
        outputs=self.model(inputs)
        loss = loss_fn(outputs, labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)                                                     #DSRR
        #return loss
    
    def metrics(self, y_score, y_true):
        vals = []
        for output_index in range(y_score.shape[1]):
            vals.append(stats.spearmanr(y_true[:,output_index], y_score[:,output_index])[0])                                                         #DSRR
        spearmanr_vals=np.array(vals)
        #
        vals = []
        for output_index in range(y_score.shape[-1]):
            vals.append(stats.pearsonr(y_true[:,output_index], y_score[:,output_index])[0] )                                                         #DSRR
        pearsonr_vals=np.array(vals)
        metrics={'Spearman':spearmanr_vals,'PCC':pearsonr_vals}
        return metrics

    def forward(self, x):                                                     #DSRR
        return self.model(x)

    def predict_custom(self, X, keepgrad=False): 
        self.model.eval()
        dataloader=torch.utils.data.DataLoader(X, batch_size=self.batch_size, shuffle=False)                                                     #DSRR
        preds=torch.empty(0)
        if keepgrad: 
            preds = preds.to(self.device)
        else:
            preds = preds.cpu()
            
        for x in tqdm.tqdm(dataloader, total=len(dataloader)):
            pred=self.model(x)
            if not keepgrad: pred=pred.detach().cpu()
            preds=torch.cat((preds,pred),axis=0)
        return preds

    def predict_custom_mcdropout(self, X,seed=41, keepgrad=False):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)                                                     #DSRR
        self.model.eval()
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'): 
                m.train()
        #
        dataloader=torch.utils.data.DataLoader(X, batch_size=self.batch_size, shuffle=False)
        preds=torch.empty(0)
        if keepgrad: 
            preds = preds.to(self.device)
        else:
            preds = preds.cpu()

        for x in tqdm.tqdm(dataloader, total=len(dataloader)):
            pred=self.model(x)
            if not keepgrad: pred=pred.detach().cpu()
            preds=torch.cat((preds,pred),axis=0)
        
        return preds







###########################################


def training_with_PL(chosen_model, chosen_dataset,
                     initial_test=False, mcdropout_test=False, verbose=False, wanted_wandb=False):

    if wanted_wandb:
        import wandb
        from pytorch_lightning.loggers import WandbLogger # https://docs.wandb.ai/guides/integrations/lightning
        wandb_logger = WandbLogger(log_model="all")

        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device=} {torch.cuda.is_available()=}")

    currdir=os.popen('pwd').read().replace("\n","") #os.getcwd()
    outdir="../outputs/" #../../outputs_DALdna/"
    log_dir=outdir+"lightning_logs_"+chosen_model+"/"
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir)

    if wanted_wandb:
        logger_of_choice=wandb_logger
    else:
        logger_of_choice=tb_logger

    #model=eval("PL_"+chosen_model+"(input_h5_file='../inputs/"+chosen_dataset+".h5', initial_ds=True)") #PERFECT OLD
    model=eval("PL_"+chosen_model+"(input_h5_file='./"+chosen_dataset+".h5', initial_ds=True)") #PERFECT OLD

    os.system('date')
    print (model.X_train.shape)
    print (model.y_train.shape)
    train_dataloader=torch.utils.data.DataLoader(list(zip(model.X_train,model.y_train)), batch_size=model.batch_size, shuffle=True)
    os.system('date')
    valid_dataloader=torch.utils.data.DataLoader(list(zip(model.X_valid,model.y_valid)), batch_size=model.batch_size, shuffle=False) #True)
    os.system('date')
    test_dataloader=torch.utils.data.DataLoader(list(zip(model.X_test,model.y_test)), batch_size=model.batch_size, shuffle=False) #True)
    os.system('date')

    ckptfile="oracle_"+model.name+"_"+chosen_dataset #+".ckpt"
    to_monitor='val_loss' 
    callback_ckpt = pl.callbacks.ModelCheckpoint( # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html
        #gpus=1,
        #auto_select_gpus=True,
        monitor=to_monitor, #default is None which saves a checkpoint only for the last epoch.
        mode='min',
        save_top_k=1,
        save_weights_only=True,
        dirpath="./", #"../inputs/", #get_github_main_directory(reponame='DALdna')+"inputs/", 
        filename=ckptfile, #comment out to verify that a different epoch is picked in the name.
    )
    early_stop_callback = pl.callbacks.EarlyStopping(
                            #monitor='val_loss',
                            monitor=to_monitor,
                            min_delta=model.min_delta, #https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html
                            patience=model.patience,
                            verbose=False,
                            mode='min'
                            )

    if initial_test:
        print('predict_custom')
        y_score=model.predict_custom(model.X_test)
        print('y_score.shape: ',y_score.shape)
        metrics_pretrain=model.metrics(y_score, model.y_test)
        print(f"{metrics_pretrain=}")
        print(f"{model(model.X_test[0:10])=}")
    
    if mcdropout_test:
        n_mc = 5
        preds_mc=torch.zeros((n_mc,len(model.X_test)))
        for i in range(n_mc):
            preds_mc[i] = model.predict_custom_mcdropout(model.X_test,
                                                            seed=41+i).squeeze(axis=1).unsqueeze(axis=0)
        print('predict_custom_mcdropout')
        print('y_score.shape: ',preds_mc.shape)
        metrics_pretrain=model.metrics(y_score, model.y_test)
        print(f"{metrics_pretrain=}")
        print(f"{model(model.X_test[0:10])=}")

    print(f"{model.device=}")
    trainer = pl.Trainer(accelerator='cuda', devices=-1, max_epochs=model.train_max_epochs, logger=logger_of_choice, callbacks=[callback_ckpt,early_stop_callback],deterministic=True) 
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader) #GOODOLD
    #os.system('mv ../inputs/'+ckptfile+'-v1.ckpt ../inputs/'+ckptfile+'.ckpt')
    os.system('mv ./'+ckptfile+'-v1.ckpt ./'+ckptfile+'.ckpt')
    if verbose: os.system('date')
    y_score=model.predict_custom(model.X_test)
    if verbose: os.system('date')
    metrics=model.metrics(y_score, model.y_test)
    print(metrics)

    """
    if wanted_wandb:
        wandb.log(metrics)
    """

    print(ckptfile)
    return metrics




##############################################################





if __name__=='__main__':

    pairlist=[
              ['DeepSTARR', 'DeepSTARR_data'],

    ] 

    for pair in pairlist:
                
        chosen_model=pair[0]
        chosen_dataset=pair[1]

        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        overall_seed=41
        myseed=overall_seed
        torch.manual_seed(myseed)
        random.seed(myseed)
        np.random.seed(myseed)

        import logging
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

        #metrics=training_with_PL(chosen_model, chosen_dataset, initial_test=False, mcdropout_test=False, verbose=False, wanted_wandb=True)
        metrics=training_with_PL(chosen_model, chosen_dataset, initial_test=True, mcdropout_test=False, verbose=False, wanted_wandb=False)

        print("SCRIPT END")
        print("WARNING: should I do a deep ensemble, and then take the ckpt of the best model?")
