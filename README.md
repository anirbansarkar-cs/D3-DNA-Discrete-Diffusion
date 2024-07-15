# Designing DNA With Tunable Regulatory Activity Using Discrete Diffusion

This repo contains a PyTorch implementation for the paper "Designing DNA With Tunable Regulatory Activity Using Discrete Diffusion". The training and sampling part of the code is inspired by [Score entropy discrete diffusion](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion).

## Design Choices

This codebase is built modularly to promote future research (as opposed to a more compact framework, which would be better for applications). The primary files are 

1. ```noise_lib.py```: the noise schedule
2. ```graph_lib```: the forward diffusion process
3. ```sampling.py```: the sampling strategies
4. ```model/```: the model architecture

## Installation

All the training and sampling related codes for D3 are in ```train_n_sample``` folder. Please navigate there and simply run

```
conda env create -f environment.yml
```

which will create a ```d3``` environment with packages installed (please provide your server username in place of ```<username>```). Note that this installs with CUDA 11.8, and different CUDA versions must be installed manually. Activate ```d3 ``` and install torch with below command

```
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
```
Please install other packages as required (may not have installed from ```environment.yml```).

Steps to note before training:
1. Please follow codes from [Dirichlet-flow-matching](https://github.com/HannesStark/dirichlet-flow-matching) and [Dirichlet diffusion score model](https://github.com/jzhoulab/ddsm) for setting up the code to train for Promoter dataset. Then uncomment the line ```from promoter_dataset import PromoterDataset``` in ```data.py```. Ignore this step for other datasets.
2. Comment out all the dataset initialization except for the dataset you want to train D3 in ```data.py```.
3. Make proper changes in ```configs/config.yaml``` for the dataset selected, such as data:train and data:valid. A folder will be created inside ```exp_local``` accordingly. Change other values according to the requirement.
4. Inside ```configs/model/small.yaml```, provide proper length value (promoter -> 1024, deepstarr -> 249, mpra -> 200). Keep cond_dim as 128 for transformer and change it to 256 for convolution architecture.
5. Select proper file for architecture definition through ```model/__init__.py``` according to the selected dataset. (```transformer.py``` file for deepstarr).

Example training command
```
python train.py noise.type=geometric graph.type=uniform model=small model.scale_by_sigma=False
```
This creates a new directory `direc=exp_local/DATE/TIME` with the following structure (compatible with running sampling experiments locally)
```
├── direc
│   ├── hydra
│   │   ├── config.yaml
│   │   ├── ...
│   ├── checkpoints
│   │   ├── checkpoint_*.pth
│   ├── checkpoints-meta
│   │   ├── checkpoint.pth
│   ├── samples
│   │   ├── iter_*
│   │   │   ├── sample_*.txt
│   ├── logs
```
Here, `checkpoints-meta` is used for reloading the run following interruptions, `samples` contains generated images as the run progresses, and `logs` contains the run output. Arguments can be added with `ARG_NAME=ARG_VALUE`, with important ones being:
```
ngpus                     the number of gpus to use in training (using pytorch DDP)
noise.type                geometric
graph.type                uniform
model                     small
model.scale_by_sigma      False
```
### Run Sampling

Steps to note before sampling:
1. If you have trained a model, then you should have a folder saved with run timestamp under ```exp_local/"dataset"/``` which contains configuarion files and different checkpoints that can be used for sampling.
2. If you want to just sample, place the checkpoint file (download links provided below) in ```exp_local/"dataset"/"arch"/checkpoints/``` ("dataset" is either promoter, deepstarr or mpra. "arch" is either Tran or Conv). Please create a folder named ```checkpoints``` under ```exp_local/"dataset"/"arch"/``` and update the file name accordingly in ```load_model.py```(line 26).
3. The configuration files are already provided in the ```exp_local/"dataset"/"arch"/hydra/``` folders which were generated during training and can be used directly for sampling.
4. Please download the oracle models for DeepSTARR, MPRA (download links provided below) to be used for MSE calculation.
5. Please follow codes from [Dirichlet-flow-matching](https://github.com/HannesStark/dirichlet-flow-matching) and [Dirichlet diffusion score model](https://github.com/jzhoulab/ddsm) for downloading SEI features and pretrained models for Promoter dataset.
6. Run specific codes to sample sequences for a specific dataset. (```run_sample.py``` works by default for DeepSTARR, and requires specific changes for MPRA. ```run_sample_promoter.py``` works for Promoter).

We can run sampling using a command 

```
python run_sample.py --model_path MODEL_PATH --steps STEPS
```
The ```model_path``` argument should point to ```exp_local/"dataset"/"arch"/``` folder. If you trained a D3 model, the folder should be ```exp_local/"dataset"/${now:%Y.%m.%d}/${now:%H%M%S}```, which should already be created during training.
In any case, this will generate samples for all the true test activity levels and store them in the model path. Also it will calculate the mse (between true test vs generated) through the oracle predictions. If you face any key mismatch issue with the pretrained D3 models, please consider un/commenting related variables from model architecture details to solve them.

### Datasets and Oracles

We provide preprocessed datasets for [DeepSTARR](https://huggingface.co/datasets/anonymous-3E42/DeepSTARR_preprocessed), [MPRA](https://huggingface.co/datasets/anonymous-3E42/MPRA_preprocessed) and oracle models at [DeepSTARR](https://huggingface.co/anonymous-3E42/DeepSTARR_oracle), [MPRA](https://huggingface.co/anonymous-3E42/MPRA_oracle).

### Pretrained Models

We provide pretrained models for Promoter, DeepSTARR and MPRA datasets below, each with transformer and convolution architectures.

1. [Promoter with transformer](https://huggingface.co/anonymous-3E42/Promoter_D3_Tran_model)
2. [Promoter with convolution](https://huggingface.co/anonymous-3E42/Promoter_D3_Conv_model)
3. [DeepSTARR with transformer](https://huggingface.co/anonymous-3E42/DeepSTARR_D3_Tran_model)
4. [DeepSTARR with convolution](https://huggingface.co/anonymous-3E42/DeepSTARR_D3_Conv_model)
5. [MPRA with transformer](https://huggingface.co/anonymous-3E42/MPRA_D3_Tran_model)
6. [MPRA with convolution](https://huggingface.co/anonymous-3E42/MPRA_D3_Conv_model)

### Sample generated data

We generate data points conditioned on the same activity levels for every dataset, where we only used test splits. Please find below the links to the generated data sets where D3 trained with transformer and convolution architectures.

1.[Promoter generated samples with D3 transformer](https://huggingface.co/datasets/anonymous-3E42/Promoter_sample_generated_D3_Tran)

2.[Promoter generated samples with D3 convolution](https://huggingface.co/datasets/anonymous-3E42/Promoter_sample_generated_D3_Conv)

3.[DeepSTARR generated samples with D3 transformer](https://huggingface.co/datasets/anonymous-3E42/DeepSTARR_sample_generated_D3_Tran)

4.[DeepSTARR generated samples with D3 convolution](https://huggingface.co/datasets/anonymous-3E42/DeepSTARR_sample_generated_D3_Conv)

5.[MPRA generated samples with D3 transformer](https://huggingface.co/datasets/anonymous-3E42/MPRA_sample_generated_D3_Tran)

6.[MPRA generated samples with D3 convolution](https://huggingface.co/datasets/anonymous-3E42/MPRA_sample_generated_D3_Conv)
