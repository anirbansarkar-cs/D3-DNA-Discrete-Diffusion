import abc
import torch
import torch.nn.functional as F
from utils.catsample import sample_categorical

from utils.utils import get_score_fn

_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

    
def get_predictor(name):
    return _PREDICTORS[name]



class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, x, labels, t, step_size):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass


@register_predictor(name="euler")
class EulerPredictor(Predictor):
    def update_fn(self, score_fn, x, labels, t, step_size, save_elements=None):
        sigma, dsigma = self.noise(t)
        score = score_fn(x, sigma, labels)

        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        x = self.graph.sample_rate(x, rev_rate)
        
        # Save elements if requested (sequence is saved separately in main loop, only score available for Euler)
        if save_elements is not None:
            if 'score' in save_elements:
                save_elements['score'].append(score.clone())
        
        return x

@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, x, labels, t, step_size):
        return x


@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    def update_fn(self, score_fn, x, labels, t, step_size, save_elements=None):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score = score_fn(x, curr_sigma, labels)

        stag_score = self.graph.staggered_score(score, dsigma)
        # print (stag_score.shape)
        probs = stag_score * self.graph.transp_transition(x, dsigma)
        
        # Save elements if requested (sequence is saved separately in main loop)
        if save_elements is not None:
            if 'score' in save_elements:
                save_elements['score'].append(score.clone())
            if 'stag_score' in save_elements:
                save_elements['stag_score'].append(stag_score.clone())
            if 'prob' in save_elements:
                save_elements['prob'].append(probs.clone())
        
        return sample_categorical(probs)

    
class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, x, labels, t, save_elements=None):
        sigma = self.noise(t)[0]

        score = score_fn(x, sigma, labels)
        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        # truncate probabilities
        if self.graph.absorb:
            probs = probs[..., :-1]
        
        # Save elements if requested (sequence is saved separately in main loop)
        if save_elements is not None:
            if 'score' in save_elements:
                save_elements['score'].append(score.clone())
            if 'stag_score' in save_elements:
                save_elements['stag_score'].append(stag_score.clone())
            if 'prob' in save_elements:
                save_elements['prob'].append(probs.clone())
        
        #return probs.argmax(dim=-1)
        return sample_categorical(probs)
                       

def get_sampling_fn(config, graph, noise, batch_dims, eps, device):
    
    sampling_fn = get_pc_sampler(graph=graph,
                                 noise=noise,
                                 batch_dims=batch_dims,
                                 predictor=config.sampling.predictor,
                                 steps=config.sampling.steps,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=device)
    
    return sampling_fn
    

def get_pc_sampler(graph, noise, batch_dims, predictor, steps, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x, save_elements_list=None):
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model, labels):
        sampling_score_fn = get_score_fn(model, train=False, sampling=True)
        x = graph.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        # Initialize element storage if requested
        saved_elements = {}
        if save_elements_list:
            for elem in save_elements_list:
                saved_elements[elem] = []

        # Save initial state
        if saved_elements and 'sequence' in saved_elements:
            saved_elements['sequence'].append(x.clone())

        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            # Create save_elements dict excluding 'sequence' (we'll save it separately after update)
            predictor_save_elements = {}
            if saved_elements:
                for key in saved_elements:
                    if key != 'sequence':
                        predictor_save_elements[key] = saved_elements[key]
            
            x = predictor.update_fn(sampling_score_fn, x, labels, t, dt, 
                                  save_elements=predictor_save_elements if predictor_save_elements else None)
            
            # Save sequence state after predictor update
            if saved_elements and 'sequence' in saved_elements:
                saved_elements['sequence'].append(x.clone())

        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            # Create save_elements dict excluding 'sequence' (we'll save it separately after update)
            denoiser_save_elements = {}
            if saved_elements:
                for key in saved_elements:
                    if key != 'sequence':
                        denoiser_save_elements[key] = saved_elements[key]
            
            x = denoiser.update_fn(sampling_score_fn, x, labels, t, save_elements=denoiser_save_elements if denoiser_save_elements else None)
            
            # Save final sequence state after denoiser
            if saved_elements and 'sequence' in saved_elements:
                saved_elements['sequence'].append(x.clone())
            
        if saved_elements:
            return x, saved_elements
        return x
    
    return pc_sampler



