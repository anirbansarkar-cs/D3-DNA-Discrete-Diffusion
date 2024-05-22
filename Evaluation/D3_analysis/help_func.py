import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
import matplotlib.cm as cm
import torch
from captum.attr import GradientShap
import logomaker
import matplotlib.pyplot as plt
import h5py

def plot_attribution_map(saliency_df, ax=None, figsize=(20,1)):
  """plot an attribution map using logomaker"""

  logomaker.Logo(saliency_df, figsize=figsize, ax=ax)
  if ax is None:
    ax = plt.gca()
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.yaxis.set_ticks_position('none')
  ax.xaxis.set_ticks_position('none')
  plt.xticks([])
  plt.yticks([])
    
def grad_times_input_to_df(x, grad, alphabet='ACGT'):
  """generate pandas dataframe for saliency plot
     based on grad x inputs """

  x_index = np.argmax(np.squeeze(x), axis=1)
  grad = np.squeeze(grad)
  L, A = grad.shape

  seq = ''
  saliency = np.zeros((L))
  for i in range(L):
    seq += alphabet[x_index[i]]
    saliency[i] = grad[i,x_index[i]]

  # create saliency matrix
  saliency_df = logomaker.saliency_to_matrix(seq=seq, values=saliency)
  return saliency_df

def gradient_shap(act_x,act_idx,num_plot,model,class_index,save_score=None,title=None,save_dir=None):
    if save_score == None:
        fig = plt.figure(figsize=(20,2*num_plot))
    N,A,L = act_x.shape
    score_cache = []
    for i,x in enumerate(act_x):
        # process sequences so that they are right shape (based on insertions)
        x = np.expand_dims(x, axis=0)
        x_tensor = torch.tensor(x, requires_grad=True, dtype=torch.float32)
        x_tensor = model._pad_end(x_tensor)
        x = x_tensor.detach().numpy()

        # get predictions of models
        output = model(x_tensor)

        # random background
        num_background = 1000
        null_index = np.random.randint(0,3, size=(num_background,L))
        x_null = np.zeros((num_background,A,L))
        for n in range(num_background):
            for l in range(L):
               x_null[n,null_index[n,l],l] = 1.0
        x_null_tensor = torch.tensor(x_null, requires_grad=True, dtype=torch.float32)
        x_null_tensor = model._pad_end(x_null_tensor)

        # calculate gradient shap
        gradient_shap = GradientShap(model)
        grad = gradient_shap.attribute(x_tensor,
                                      n_samples=100,
                                      stdevs=0.1,
                                      baselines=x_null_tensor,
                                      target=class_index)
        grad = grad.data.cpu().numpy()

        # process gradients with gradient correction (Majdandzic et al. 2022)
        grad -= np.mean(grad, axis=1, keepdims=True)
        
        if save_score:
            score_cache.append(grad)
            continue
            
        # plot sequence logo of grad times input
        ax = plt.subplot(num_plot,1,i+1)
        saliency_df = grad_times_input_to_df(x.transpose([0,2,1]), grad.transpose([0,2,1]))
        plot_attribution_map(saliency_df, ax, figsize=(20,1))
        plt.ylabel(act_idx[i])
        
    if save_score:
        score_cache = np.squeeze(np.array(score_cache))
        if len(score_cache.shape)<3:
            score_cache=np.expand_dims(score_cache,axis=0)
        np.savez(save_score,score_cache[:,:,:-20])
        return 
    
    if title:
        fig.suptitle(title)
    if save_dir:
        fig.savefig(save_dir, format='pdf', dpi=200, bbox_inches='tight')     
      
            
 # taken from https://github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py
def dinuc_shuffle(seq, num_shufs=None, rng=None):
    """Creates shuffles of the given sequence, in which dinucleotide frequencies
    are preserved.

    Parameters
    ----------
    seq : str or ndarray
        either a string of length L, or an L x D NumPy array of one-hot encodings
    num_shufs : int
        the number of shuffles to create, N; if unspecified, only one shuffle will be created
        `rng`: a NumPy RandomState object, to use for performing shuffles

    Returns
    -------
    list (if 'seq' is string)
        List of N strings of length L, each one being a shuffled version of 'seq'
        
    ndarray (if 'seq' is ndarray)
        ndarray of shuffled versions of 'seq' (shape=(N,L,D)), also one-hot encoded
        If 'num_shufs' is not specified, then the first dimension of N will not be present
        (i.e. a single string will be returned, or an LxD array).
    """
    def string_to_char_array(seq):
        """
        Converts an ASCII string to a NumPy array of byte-long ASCII codes.
        e.g. "ACGT" becomes [65, 67, 71, 84].
        """
        return np.frombuffer(bytearray(seq, "utf8"), dtype=np.int8)


    def char_array_to_string(arr):
        """
        Converts a NumPy array of byte-long ASCII codes into an ASCII string.
        e.g. [65, 67, 71, 84] becomes "ACGT".
        """
        return arr.tostring().decode("ascii")


    def one_hot_to_tokens(one_hot):
        """
        Converts an L x D one-hot encoding into an L-vector of integers in the range
        [0, D], where the token D is used when the one-hot encoding is all 0. This
        assumes that the one-hot encoding is well-formed, with at most one 1 in each
        column (and 0s elsewhere).
        """
        tokens = np.tile(one_hot.shape[1], one_hot.shape[0])  # Vector of all D
        seq_inds, dim_inds = np.where(one_hot)
        tokens[seq_inds] = dim_inds
        return tokens


    def tokens_to_one_hot(tokens, one_hot_dim):
        """
        Converts an L-vector of integers in the range [0, D] to an L x D one-hot
        encoding. The value `D` must be provided as `one_hot_dim`. A token of D
        means the one-hot encoding is all 0s.
        """
        identity = np.identity(one_hot_dim + 1)[:, :-1]  # Last row is all 0s
        return identity[tokens]

    if type(seq) is str:
        arr = string_to_char_array(seq)
    elif type(seq) is np.ndarray and len(seq.shape) == 2:
        seq_len, one_hot_dim = seq.shape
        arr = one_hot_to_tokens(seq)
    else:
        raise ValueError("Expected string or one-hot encoded array")


    if not rng:
        rng = np.random.RandomState()
   
    # Get the set of all characters, and a mapping of which positions have which
    # characters; use `tokens`, which are integer representations of the
    # original characters
    chars, tokens = np.unique(arr, return_inverse=True)

    # For each token, get a list of indices of all the tokens that come after it
    shuf_next_inds = []
    for t in range(len(chars)):
        mask = tokens[:-1] == t  # Excluding last char
        inds = np.where(mask)[0]
        shuf_next_inds.append(inds + 1)  # Add 1 for next token
 
    if type(seq) is str:
        all_results = []
    else:
        all_results = np.empty(
            (num_shufs if num_shufs else 1, seq_len, one_hot_dim),
            dtype=seq.dtype
        )

    for i in range(num_shufs if num_shufs else 1):
        # Shuffle the next indices
        for t in range(len(chars)):
            inds = np.arange(len(shuf_next_inds[t]))
            inds[:-1] = rng.permutation(len(inds) - 1)  # Keep last index same
            shuf_next_inds[t] = shuf_next_inds[t][inds]

        counters = [0] * len(chars)
       
        # Build the resulting array
        ind = 0
        result = np.empty_like(tokens)
        result[0] = tokens[ind]
        for j in range(1, len(tokens)):
            t = tokens[ind]
            ind = shuf_next_inds[t][counters[t]]
            counters[t] += 1
            result[j] = tokens[ind]

        if type(seq) is str:
            all_results.append(char_array_to_string(chars[result]))
        else:
            all_results[i] = tokens_to_one_hot(chars[result], one_hot_dim)
    return all_results if num_shufs else all_results[0]       