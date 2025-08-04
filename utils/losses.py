import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from utils import graph_lib
from utils.utils import get_score_fn


def get_loss_fn(noise, graph, train, sampling_eps=1e-3, lv=False, sc=False):

    def loss_fn(model, batch, labels=None, cond=None, t=None, perturbed_batch=None):
        """
        Batch shape: [B, L] int. D given from graph
        """

        if t is None:
            if lv:
                raise NotImplementedError("Yeah I gotta do this later")
            else:
                t = (1 - sampling_eps) * torch.rand(batch.shape[0], device=batch.device) + sampling_eps
            
        sigma, dsigma = noise(t)
        
        if perturbed_batch is None:
            perturbed_batch = graph.sample_transition(batch, sigma[:, None])

        log_score_fn = get_score_fn(model, train=train, sampling=False)
        log_score = log_score_fn(perturbed_batch, sigma, labels)

        if sc:
            # Use the complex loss calculation with sampling consistency
            curr_sigma, curr_dsigma = noise(t/2)
            curr_score = log_score_fn(perturbed_batch, curr_sigma, labels)
            t_dsigma = t/2 * curr_dsigma
            rev_rate = t_dsigma[..., None, None] * graph.reverse_rate(perturbed_batch, curr_score)
            x = graph.sample_rate(perturbed_batch, rev_rate)
            
            # Fix: Create sampling_eps tensor on the same device as batch
            sampling_eps_tensor = torch.tensor(sampling_eps, device=batch.device)
            next_sigma, next_dsigma = noise(sampling_eps_tensor)
            next_score = log_score_fn(x, next_sigma, labels)
            t_dsigma_next = sampling_eps_tensor * next_dsigma
            rev_rate_next = t_dsigma_next[..., None, None] * graph.reverse_rate(x, next_score)
            
            x_next = graph.sample_rate(x, rev_rate_next)
            l2_loss = ((batch - x_next)**2)
            mask = torch.rand(batch.shape[0], device=batch.device) < 0.25
            expanded_mask = mask.unsqueeze(-1).expand_as(l2_loss)
            loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)
            loss = (dsigma[:, None] * loss)
            main_loss = loss.clone()
            main_loss[expanded_mask] = loss[expanded_mask] + l2_loss[expanded_mask]
            final_loss = main_loss.sum(dim=-1)
            return final_loss
        else:
            # Use the original simple loss calculation
            loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, batch)
            loss = (dsigma[:, None] * loss).sum(dim=-1)
            return loss

    return loss_fn


def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, 
                    scaler, 
                    params, 
                    step, 
                    lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        scaler.unscale_(optimizer)

        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

    return optimize_fn


def get_step_fn(noise, graph, train, optimize_fn, accum):
    loss_fn = get_loss_fn(noise, graph, train)

    accum_iter = 0
    total_loss = 0

    def step_fn(state, batch, labels=None, cond=None):
        nonlocal accum_iter 
        nonlocal total_loss

        model = state['model']

        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
            loss = loss_fn(model, batch, labels, cond=cond).mean() / accum
            
            scaler.scale(loss).backward()

            accum_iter += 1
            total_loss += loss.detach()
            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())
                optimizer.zero_grad()
                
                loss = total_loss
                total_loss = 0
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch, labels, cond=cond).mean()
                ema.restore(model.parameters())

        return loss

    return step_fn