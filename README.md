# Designing DNA With Tunable Regulatory Activity Using Discrete Diffusion

This repo contains a PyTorch implementation for the paper "Designing DNA With Tunable Regulatory Activity Using Discrete Diffusion".

## Design Choices

This codebase is built modularly to promote future research (as opposed to a more compact framework, which would be better for applications). The primary files are 

1. ```noise_lib.py```: the noise schedule
2. ```graph_lib```: the forward diffusion process
3. ```sampling.py```: the sampling strategies
4. ```model/```: the model architecture

## Installation

Simply run

```
conda env create -f environment.yml
```

which will create a ```d3``` environment with packages installed. Note that this installs with CUDA 11.8, and different CUDA versions must be installed manually. The biggest factor is making sure that the ```torch``` and ```flash-attn``` packages use the same CUDA version (more found [here](https://github.com/Dao-AILab/flash-attention)).

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

We can run sampling using a command 

```
python run_sample.py --model_path MODEL_PATH --steps STEPS
```

### Datasets

We provide preprocessed datasets for [DeepSTARR](https://huggingface.co/datasets/anonymous-3E42/DeepSTARR_preprocessed) and [MPRA](https://huggingface.co/datasets/anonymous-3E42/MPRA_preprocessed).

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
