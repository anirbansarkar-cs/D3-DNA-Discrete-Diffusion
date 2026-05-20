"""Dinucleotide-preserving shuffle.

Originally from https://github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py
"""

from typing import List

import numpy as np


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
    def string_to_char_array(seq_str):
        return np.frombuffer(bytearray(seq_str, "utf8"), dtype=np.int8)

    def char_array_to_string(arr):
        return arr.tobytes().decode("ascii")

    def one_hot_to_tokens(one_hot):
        tokens = np.tile(one_hot.shape[1], one_hot.shape[0])  # Vector of all D
        seq_inds, dim_inds = np.where(one_hot)
        tokens[seq_inds] = dim_inds
        return tokens

    def tokens_to_one_hot(tokens, one_hot_dim, dtype):
        identity = np.identity(one_hot_dim + 1, dtype=dtype)[:, :-1]  # Last row is all 0s
        return identity[tokens]

    if not rng:
        rng = np.random.RandomState()

    # Branch 1: string input
    if isinstance(seq, str):
        arr = string_to_char_array(seq)
        chars, tokens = np.unique(arr, return_inverse=True)
        shuf_next_inds = []
        for t in range(len(chars)):
            mask = tokens[:-1] == t
            inds = np.where(mask)[0]
            shuf_next_inds.append(inds + 1)

        results: List[str] = []
        N = num_shufs if num_shufs else 1
        for _ in range(N):
            for t in range(len(chars)):
                inds = np.arange(len(shuf_next_inds[t]))
                if len(inds) > 1:
                    inds[:-1] = rng.permutation(len(inds) - 1)
                shuf_next_inds[t] = shuf_next_inds[t][inds]
            counters = [0] * len(chars)
            ind = 0
            result = np.empty_like(tokens)
            result[0] = tokens[ind]
            for j in range(1, len(tokens)):
                t = tokens[ind]
                ind = shuf_next_inds[t][counters[t]]
                counters[t] += 1
                result[j] = tokens[ind]
            results.append(char_array_to_string(chars[result]))
        return results if num_shufs else results[0]

    # Branch 2: ndarray input (L, D)
    if isinstance(seq, np.ndarray) and len(seq.shape) == 2:
        seq_len, one_hot_dim = seq.shape
        arr = one_hot_to_tokens(seq)
        chars, tokens = np.unique(arr, return_inverse=True)
        shuf_next_inds = []
        for t in range(len(chars)):
            mask = tokens[:-1] == t
            inds = np.where(mask)[0]
            shuf_next_inds.append(inds + 1)

        N = num_shufs if num_shufs else 1
        results_np = np.empty((N, seq_len, one_hot_dim), dtype=seq.dtype)
        for i in range(N):
            for t in range(len(chars)):
                inds = np.arange(len(shuf_next_inds[t]))
                if len(inds) > 1:
                    inds[:-1] = rng.permutation(len(inds) - 1)
                shuf_next_inds[t] = shuf_next_inds[t][inds]
            counters = [0] * len(chars)
            ind = 0
            result = np.empty_like(tokens)
            result[0] = tokens[ind]
            for j in range(1, len(tokens)):
                t = tokens[ind]
                ind = shuf_next_inds[t][counters[t]]
                counters[t] += 1
                result[j] = tokens[ind]
            results_np[i] = tokens_to_one_hot(chars[result], one_hot_dim, dtype=seq.dtype)
        return results_np if num_shufs else results_np[0]

    raise ValueError("Expected 'seq' to be a string or a (L, D) one-hot ndarray")
