# Designing DNA With Tunable Regulatory Activity Using Discrete Diffusion

This repo contains a PyTorch implementation for the paper "Designing DNA With Tunable Regulatory Activity Using Discrete Diffusion" (NeurIPS2024 submission).

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
