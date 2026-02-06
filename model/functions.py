import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
import torch.nn as nn
from torch.nn import functional as F


def number_of_params(model, embed=False, gen=False):
	return sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and ('wpe' not in n or embed) and ('wte' not in n or embed) and ('lm_head' not in n or gen))


def remap_to_indices(arr):
	unique_vals, inverse_indices = np.unique(arr, return_inverse=True)
	return inverse_indices


def nested_dict_from_state_dict(state_dict):
	nested = {}
	for key, value in state_dict.items():
		parts = key.split(".")
		d = nested
		for part in parts[:-1]:
			if part not in d:
				d[part] = {}
			d = d[part]
		d[parts[-1]] = value
	return nested


def fibration_linear(weights, in_clusters, threshold=None, n_clusters = None, first_layer = False, bias=None):
	dim_out, dim_in  = weights.shape
	in_clusters = remap_to_indices(in_clusters)

	if first_layer:
		collapse_weights = weights
	else:	
		num_in_clusters  = len(np.unique(in_clusters))
		collapse_weights = torch.zeros((dim_out, num_in_clusters))

		for color in in_clusters: 
			indices_k = np.where(in_clusters == color)[0]
			collapse_weights[:, color] = weights[:, indices_k].sum(axis=1)

	if bias is not None:
		collapse_weights = torch.cat((collapse_weights, bias.unsqueeze(1)), dim=1)

	if dim_out == 1:
		return np.array([0])
	if dim_in == 1:
		return np.zeros(dim_out)

	collapse_weights_norm = torch.nn.functional.normalize(collapse_weights, dim=1)
	distance = 1 - torch.mm(collapse_weights_norm, collapse_weights_norm.T)
	distance = distance.cpu().numpy()

	if n_clusters is not None:
		clustering = AgglomerativeClustering(
			n_clusters=n_clusters,
			linkage='average',
			metric='precomputed')
	else:
		clustering = AgglomerativeClustering(
			n_clusters=None,
			distance_threshold=threshold,
			linkage='average',
			metric='precomputed')
 
	clusters = clustering.fit_predict(distance)

	return clusters


class TiedLinear(nn.Module):
	"""
	Linear-compatible layer with row tying driven by `key`.

	- out_features = len(key)
	- Rows i,j are tied if key[i] == key[j].
	- Exposes `.weight` (out_features, in_features) and `.bias` (out_features,)
	  so legacy code that expects nn.Linear-like attributes will still work.

	Only the unique group parameters are registered as Parameters:
	  - weight_base: (num_groups, in_features)
	  - bias_base:   (num_groups,) if tie_bias=True
	  - or untied _bias: (out_features,) if tie_bias=False and bias=True
	"""

	def __init__(
		self,
		in_features: int,
		key: torch.Tensor,                 # 1D tensor of length out_features
		bias: bool = True,
		tie_bias: bool = True,
		init_weight: torch.Tensor = None,  # (out_features, in_features) optional
		init_bias: torch.Tensor = None,    # (out_features,) optional
		init_mode: str = "first",          # "first" or "mean"
	):
		super().__init__()
		assert key.dim() == 1, "key must be 1D"
		self.in_features  = int(in_features)
		self.out_features = int(key.numel())
		self.tie_bias     = bool(bias and tie_bias)

		# Build group map: index_map[out_idx] -> group_id in [0..G-1]
		uniq, inverse = torch.unique(key, return_inverse=True)
		self.register_buffer("index_map", inverse.to(torch.long))
		G = int(uniq.numel())

		# ---- parameters (only the unique groups are learnable) ----
		self.weight_base = nn.Parameter(torch.empty(G, self.in_features))
		if self.tie_bias:
			self.bias_base = nn.Parameter(torch.empty(G))
			self._bias = None
		else:
			self.bias_base = None
			self._bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None

		# ---- initialization ----
		with torch.no_grad():
			if init_weight is not None:
				assert init_weight.shape == (self.out_features, self.in_features)
				if init_mode == "first":
					for g in range(G):
						idx = (inverse == g).nonzero(as_tuple=True)[0][0]
						self.weight_base[g].copy_(init_weight[idx])
				elif init_mode == "mean":
					for g in range(G):
						idxs = (inverse == g).nonzero(as_tuple=True)[0]
						self.weight_base[g].copy_(init_weight[idxs].mean(0))
				else:
					raise ValueError("init_mode must be 'first' or 'mean'")
			else:
				nn.init.kaiming_uniform_(self.weight_base, a=5**0.5)

			if self.tie_bias:
				if init_bias is not None:
					assert init_bias.shape == (self.out_features,)
					if init_mode == "first":
						for g in range(G):
							idx = (inverse == g).nonzero(as_tuple=True)[0][0]
							self.bias_base[g].copy_(init_bias[idx])
					else:
						for g in range(G):
							idxs = (inverse == g).nonzero(as_tuple=True)[0]
							self.bias_base[g].copy_(init_bias[idxs].mean())
				else:
					nn.init.zeros_(self.bias_base)
			else:
				if self._bias is not None and init_bias is not None:
					assert init_bias.shape == (self.out_features,)
					self._bias.copy_(init_bias)

	# --- Linear-compatible attributes ---
	@property
	def weight(self) -> torch.Tensor:
		# Shape: (out_features, in_features)
		return self.weight_base[self.index_map]

	@property
	def bias(self) -> torch.Tensor | None:
		if self.tie_bias:
			return self.bias_base[self.index_map]
		return self._bias

	# --- Forward like nn.Linear ---
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (batch, in_features)
		return F.linear(x, self.weight, self.bias)

	def extra_repr(self) -> str:
		g = self.weight_base.size(0)
		return (f"in_features={self.weight_base.size(1)}, out_features={self.out_features}, "
				f"groups={g}, tie_bias={self.tie_bias}")
		
	def merge_groups(self, new_key: torch.Tensor, mode: str = "first"):
		"""
			Merge rows of weight_base (and bias_base if tied) according to new_key.
			
			- new_key: 1D tensor of length == old num_groups (self.weight_base.size(0)).
			Each entry assigns an old group to a new group id.
			- mode: "first" or "mean", decides how to merge weights/bias.
		"""
		assert new_key.dim() == 1, "new_key must be 1D"
		G_old = self.weight_base.size(0)
		assert new_key.numel() == G_old, "new_key must match old num_groups"

		uniq, inverse = torch.unique(new_key, return_inverse=True)
		G_new = uniq.numel()

		# new weight_base
		new_weight = torch.empty(G_new, self.weight_base.size(1), device=self.weight_base.device)
		with torch.no_grad():
			for g in range(G_new):
				idxs = (inverse == g).nonzero(as_tuple=True)[0]
				if mode == "first":
					new_weight[g].copy_(self.weight_base[idxs[0]])
				elif mode == "mean":
					new_weight[g].copy_(self.weight_base[idxs].mean(0))
				else:
					raise ValueError("mode must be 'first' or 'mean'")
		self.weight_base = nn.Parameter(new_weight)

		# new bias_base if tied
		if self.tie_bias:
			new_bias = torch.empty(G_new, device=self.bias_base.device)
			with torch.no_grad():
				for g in range(G_new):
					idxs = (inverse == g).nonzero(as_tuple=True)[0]
					if mode == "first":
						new_bias[g].copy_(self.bias_base[idxs[0]])
					else:
						new_bias[g].copy_(self.bias_base[idxs].mean())
			self.bias_base = nn.Parameter(new_bias)

		# update index_map (out_features stays same)
		self.index_map = inverse[self.index_map]
  