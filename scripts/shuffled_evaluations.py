import numpy as np
import random
import h5py
import argparse
import os
from typing import Tuple, List, Any

import torch
from omegaconf import OmegaConf, DictConfig

# Add compatibility for both 'lightning' and 'pytorch_lightning'
try:
	import pytorch_lightning as pl
	from pytorch_lightning import loggers as pl_loggers
	LIGHTNING_AVAILABLE = True
except ImportError:
	try:
		import lightning.pytorch as pl
		from lightning.pytorch import loggers as pl_loggers
		LIGHTNING_AVAILABLE = True
	except ImportError:
		LIGHTNING_AVAILABLE = False
		# Create dummy pl object for when lightning is not available
		class DummyLightningModule:
			pass
		pl = type('pl', (), {'LightningModule': DummyLightningModule, 'loggers': type('loggers', (), {})()})
		pl_loggers = pl.loggers
deepstarr_data = "/grid/koo/home/shared/d3/data/deepstarr/DeepSTARR_data.h5"
#onehot_test_shuffled (samples, 230, 4)
lentimpra_cond_shuffled_k562 = "/grid/koo/home/shared/d3/data/lentimpra/lenti_MPRA_K562_data_shuffled_dinuc_test.h5"
lentimpra_cond_hepg2 = "/grid/koo/home/shared/d3/data/lentimpra/lenti_MPRA_HepG2_data.h5"

# create a function that
# loads up the h5 file
# prints and checks the shape of the dataset - which should be good for dinuc_shuffle()
# For each sequence in the test set, generate one or more shuffled versions
# then,
# oracle predict activity (y) on test set of iriginal sequence is alrady done in the mse evaluation scripts
# new: oracle predict activity (y) on test set of shuffled sequences 
# get the mse, adapt from the evaluation script

# script taken from https://github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py
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


# -------------------------------
# New CLI for shuffled evaluation
# -------------------------------

def _load_sequences_from_h5(h5_path: str, dataset: str) -> np.ndarray:
	"""Load test sequences from H5 and return as (N, L, 4) one-hot."""
	with h5py.File(h5_path, 'r') as f:
		keys = list(f.keys())
		# Try common keys per dataset
		if dataset.lower() in ["lentimpra", "mpra"]:
			# Expected key
			if 'onehot_test_shuffled' in f:
				arr = np.array(f['onehot_test_shuffled'])  # (N, L, 4)
			else:
				# Fallback to first 3D dataset
				arr = None
				for k in keys:
					obj = f[k]
					if isinstance(obj, h5py.Dataset) and obj.ndim == 3 and obj.shape[-1] == 4:
						arr = np.array(obj)
						print(f"Loaded sequences: shape={arr.shape} from {h5_path}")
						break
				if arr is None:
					raise KeyError(f"No suitable (N,L,4) dataset found in {h5_path}. Keys: {keys}")
			return arr
		elif dataset.lower() in ["deepstarr"]:
			key_candidates = ['X_test', 'X_valid', 'X_val']
			arr = None
			for k in key_candidates:
				if k in f:
					obj = f[k]
					if isinstance(obj, h5py.Dataset):
						arr = np.array(obj)  # often (N, 4, L)
						break
			if arr is None:
				# Fallback: any 3D dataset
				for k in keys:
					obj = f[k]
					if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
						arr = np.array(obj)
						break
			if arr is None:
				raise KeyError(f"No suitable dataset found in {h5_path}. Keys: {keys}")
			# If channels-first, transpose to (N, L, 4)
			if arr.ndim == 3 and arr.shape[1] == 4:
				arr = np.transpose(arr, (0, 2, 1))
			return arr
		else:
			# Generic: prefer (N, L, 4); if (N, 4, L), transpose
			arr = None
			for k in keys:
				obj = f[k]
				if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
					arr = np.array(obj)
					break
			if arr is None:
				raise KeyError(f"No 3D dataset found in {h5_path}. Keys: {keys}")
			if arr.shape[1] == 4:
				arr = np.transpose(arr, (0, 2, 1))
			return arr


def _generate_shuffles(onehot: np.ndarray, num_shuffles: int, batch: int = 512) -> np.ndarray:
	"""Generate dinucleotide-preserving shuffles for each sequence.
	Returns array of shape (N*num_shuffles, L, 4).
	"""
	N, L, C = onehot.shape
	assert C == 4, f"Expected last dim 4, got {C}"
	out_list: List[np.ndarray] = []
	for start in range(0, N, batch):
		end = min(start + batch, N)
		for i in range(start, end):
			shufs = dinuc_shuffle(onehot[i], num_shufs=num_shuffles)
			# shufs: (num_shuffles, L, 4)
			out_list.append(np.asarray(shufs))
	return np.concatenate(out_list, axis=0)


def _save_h5(output_h5: str, dataset_name: str, array: np.ndarray):
	os.makedirs(os.path.dirname(output_h5), exist_ok=True)
	with h5py.File(output_h5, 'w') as f:
		f.create_dataset(dataset_name, data=array, compression="gzip")


def _create_sp_mse_callback(cfg: Any, dataset: str):
	ds = dataset.lower()
	if ds == 'lentimpra' or ds == 'mpra':
		from model_zoo.lentimpra.sp_mse_callback import create_lentimpra_sp_mse_callback
		return create_lentimpra_sp_mse_callback(cfg, dataset_name=ds)
	if ds == 'deepstarr':
		from model_zoo.deepstarr.sp_mse_callback import create_deepstarr_sp_mse_callback
		return create_deepstarr_sp_mse_callback(cfg, dataset_name=ds)
	if ds == 'promoter':
		from model_zoo.promoter.sp_mse_callback import create_promoter_sp_mse_callback
		return create_promoter_sp_mse_callback(cfg, dataset_name=ds)
	if ds == 'atacseq':
		from model_zoo.atacseq.sp_mse_callback import create_atacseq_sp_mse_callback
		return create_atacseq_sp_mse_callback(cfg, dataset_name=ds)
	raise ValueError(f"Unsupported dataset for SP-MSE callback: {dataset}")


def _compute_mse_with_callback(
	callback,
	original_sequences: np.ndarray,
	shuffled_sequences: np.ndarray,
	device: torch.device,
	batch_size: int = 512,
) -> float:
	"""Compute MSE between oracle predictions on originals vs. shuffled.
	original_sequences: (N, L, 4)
	shuffled_sequences: (N*K, L, 4) with K shuffles per original
	"""
	# Load oracle once
	if getattr(callback, 'oracle_model', None) is None:
		callback.oracle_model = callback.load_oracle_model()
		if callback.oracle_model is None:
			raise RuntimeError("Failed to load oracle model via SP-MSE callback")
	callback.oracle_model = callback.oracle_model.to(device).eval()

	# Predict on originals in batches
	N = original_sequences.shape[0]
	with torch.no_grad():
		orig_preds: List[torch.Tensor] = []
		for start in range(0, N, batch_size):
			end = min(start + batch_size, N)
			batch_np = original_sequences[start:end]
			batch_t = torch.from_numpy(batch_np).float()
			preds = callback.get_oracle_predictions(batch_t, device)
			orig_preds.append(preds.detach().to('cpu'))
		orig_preds_t = torch.cat(orig_preds, dim=0)  # (N, ...)

		# Predict on shuffled in batches
		NS = shuffled_sequences.shape[0]
		shuf_preds: List[torch.Tensor] = []
		for start in range(0, NS, batch_size):
			end = min(start + batch_size, NS)
			batch_np = shuffled_sequences[start:end]
			batch_t = torch.from_numpy(batch_np).float()
			preds = callback.get_oracle_predictions(batch_t, device)
			shuf_preds.append(preds.detach().to('cpu'))
		shuf_preds_t = torch.cat(shuf_preds, dim=0)  # (N*K, ...)

	# Align shapes for MSE: repeat original predictions K times
	K = NS // N
	if K * N != NS:
		raise ValueError("shuffled_sequences must be multiple of original_sequences")
	orig_repeated = orig_preds_t.repeat_interleave(K, dim=0)

	# Compute MSE
	mse = torch.mean((orig_repeated - shuf_preds_t) ** 2).item()
	return float(mse)


def main():
	parser = argparse.ArgumentParser(description='Generate dinuc-shuffled sequences and compute oracle MSE')
	parser.add_argument('--dataset', required=True, choices=['lentimpra', 'deepstarr', 'promoter', 'mpra', 'atacseq'])
	parser.add_argument('--input_h5', required=True, help='Path to input H5 with test sequences')
	parser.add_argument('--output_h5', required=False, help='Optional path to output H5 for shuffled sequences')
	parser.add_argument('--output_txt', required=False, help='Optional path to write MSE metric as a text file')
	parser.add_argument('--num_shuffles', type=int, default=1, help='Number of shuffles per sequence')
	parser.add_argument('--oracle_checkpoint', required=False, help='Path to oracle checkpoint (overrides config)')
	parser.add_argument('--data_path', required=False, help='Data file path for oracle if needed (e.g., DeepSTARR)')
	parser.add_argument('--config', required=False, help='Optional config file to source oracle paths')
	parser.add_argument('--batch_size', type=int, default=512, help='Batch size for oracle predictions')
	parser.add_argument('--dataset_key', required=False, help='Override dataset key in H5 if needed')
	args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Load sequences as (N, L, 4)
	sequences = _load_sequences_from_h5(args.input_h5, args.dataset)
	print(f"Loaded sequences: shape={sequences.shape} from {args.input_h5}")

	# Generate shuffles
	print(f"Generating {args.num_shuffles} dinuc shuffles per sequence...")
	shuffled = _generate_shuffles(sequences, args.num_shuffles)
	print(f"Generated shuffled sequences: shape={shuffled.shape}")

	# Save shuffled dataset (if a proper H5 path is provided)
	if args.output_h5:
		if args.output_h5.lower().endswith('.txt'):
			print(f"Note: --output_h5 ends with .txt, will write metric there instead of H5.")
		else:
			# Use a conventional dataset name
			dset_name = 'onehot_test_shuffled'
			_save_h5(args.output_h5, dset_name, shuffled)
			print(f"✓ Saved shuffled sequences to {args.output_h5} as '{dset_name}'")

	# Build cfg for SP-MSE callback (as DictConfig)
	overlay_dict = {
		'sp_mse_validation': {
			'enabled': True,
		},
		'paths': {}
	}
	if args.oracle_checkpoint:
		overlay_dict['sp_mse_validation']['oracle_path'] = args.oracle_checkpoint
		overlay_dict['paths']['oracle_model'] = args.oracle_checkpoint
	if args.data_path:
		overlay_dict['sp_mse_validation']['data_path'] = args.data_path
		overlay_dict['paths']['data_file'] = args.data_path

	if args.config:
		base_cfg = OmegaConf.load(args.config)
		cfg = OmegaConf.merge(base_cfg, OmegaConf.create(overlay_dict))
	else:
		cfg = OmegaConf.create(overlay_dict)

	# Create dataset-specific SP-MSE callback and compute MSE
	callback = _create_sp_mse_callback(cfg, args.dataset)
	mse_value = _compute_mse_with_callback(callback, sequences, shuffled, device, batch_size=args.batch_size)

	print(f"MSE between original and dinuc-shuffled oracle predictions: {mse_value:.6f}")

	# Write metric to txt path if provided
	metric_path = args.output_txt
	if not metric_path and args.output_h5 and args.output_h5.lower().endswith('.txt'):
		metric_path = args.output_h5
	if metric_path:
		os.makedirs(os.path.dirname(metric_path) or '.', exist_ok=True)
		with open(metric_path, 'w') as f:
			f.write(f"{mse_value:.6f}\n")
		print(f"✓ Wrote MSE to {metric_path}")

	# Also store MSE in the H5 as an attribute if we actually wrote an H5
	try:
		if args.output_h5 and not args.output_h5.lower().endswith('.txt'):
			with h5py.File(args.output_h5, 'a') as f:
				f.attrs['oracle_mse'] = mse_value
				f.attrs['num_shuffles'] = args.num_shuffles
				f.attrs['dataset'] = args.dataset
			print("✓ Stored oracle_mse as attribute in output H5")
	except Exception as e:
		print(f"Warning: failed to write attributes to H5: {e}")


if __name__ == '__main__':
	main()


