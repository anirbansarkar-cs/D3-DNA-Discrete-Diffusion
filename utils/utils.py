import torch
import torch.nn.functional as F
import os
import logging
from omegaconf import OmegaConf, open_dict


# Model utility functions (from model/utils.py)
def get_model_fn(model, train=False):
    """Create a function to give the output of the score-based model.

    Args:
        model: The score model.
        train: `True` for training and `False` for evaluation.
        mlm: If the input model is a mlm and models the base probability 

    Returns:
        A model function.
    """

    def model_fn(x, sigma, labels=None):
        """Compute the output of the score-based model.

        Args:
            x: A mini-batch of input data.
            labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
              for different models.

        Returns:
            A tuple of (model output, new mutable states)
        """
        if train:
            model.train()
        else:
            model.eval()
        
            # otherwise output the raw values (we handle mlm training in losses.py)
        return model(x, labels, train, sigma)

    return model_fn


def get_score_fn(model, train=False, sampling=False):
    if sampling:
        assert not train, "Must sample in eval mode"
    model_fn = get_model_fn(model, train=train)

    def score_fn(x, sigma, labels=None):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sigma = sigma.reshape(-1)
            score = model_fn(x, sigma, labels)
            
            if sampling:
                # when sampling return true score (not log used for training)
                return score.exp()
                
            return score

    return score_fn


# General utility functions (from utils/utils1.py)
def load_hydra_config_from_run(load_dir):
    cfg_path = os.path.join(load_dir, "hydra/config.yaml")
    cfg = OmegaConf.load(cfg_path)
    return cfg


def get_logger(logpath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def restore_checkpoint(ckpt_dir, state, device):
    if not os.path.exists(ckpt_dir):
        os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
        logging.warning(f"No checkpoint found at {ckpt_dir}. Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        state['model'].module.load_state_dict(loaded_state['model'], strict=False)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].module.state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir) 


def update_cfg_with_unknown_args(cfg, unknown_args):
    """Update cfg after loading with OmegaConf to override values in the config file.
    
    Args:
        cfg: The config object to update.
        unknown_args: A list of unknown arguments.
    """
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith('--'):
            key = arg[2:]
            if (i + 1) < len(unknown_args):
                value = unknown_args[i + 1]
                # Try to interpret value as int, float, or bool
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass  # keep as string
                OmegaConf.update(cfg, key, value, merge=True)
                print(f"Updated cfg with {key}={value}")
                i += 2
            else:
                i += 1
        else:
            i += 1