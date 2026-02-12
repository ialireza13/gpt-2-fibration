import torch
import argparse
import os
import tiktoken

from manager.device_manager import DeviceManager
from model.model import GPTSeparateAttention
from config import Config

from data.dataloader import DataLoaderLite
from config import Config

from model.functions import number_of_params


def evaluate_validation(model, val_loss_steps = 160):
	model.eval()
	val_loader.reset()
	with torch.no_grad():
		val_loss_accum = 0.0
		for _ in range(val_loss_steps):
			x, y = val_loader.next_batch()
			x, y = x.to(dm.device), y.to(dm.device)
			with torch.autocast(device_type=dm.device, dtype=torch.bfloat16):
				_, loss = model(x, y)
			loss /= val_loss_steps
			val_loss_accum += loss.detach()
			
	return val_loss_accum.item()

def multiply_thresholds(d, factor):
	new_d = {}
	for k, v in d.items():
		if isinstance(v, dict):
			new_d[k] = multiply_thresholds(v, factor)
		elif isinstance(v, list):
			new_d[k] = [x * factor for x in v]
		elif isinstance(v, (int, float)):
			new_d[k] = v * factor
		else:
			new_d[k] = v  # leave other types unchanged
	return new_d


def find_max_eps(model, thresholds, path, Lw, device, eps_low=0.0, eps_high=2.0, tol=1e-3, max_iter=40, loss_tolerance=1.01):
	"""
	Bisection method to find maximum epsilon that does not increase loss.

	Args:
		model: your model object with .coloring() and .collapse()
		Lw: baseline loss to compare against
		device: torch device
		eps_low, eps_high: search interval for epsilon
		tol: tolerance for convergence
		max_iter: maximum iterations

	Returns:
		best_eps: highest epsilon found with no loss increase
	"""
 
	best_eps = eps_low
	model.eval()

	parts = path.split('.')
	val = thresholds
	# Traverse down to the parent of the target value
	for p in parts[:-1]:
		try:
			p = int(p)
		except ValueError:
			pass
		val = val[p]

	# Handle the final key/index
	last = parts[-1]
	try:
		last = int(last)
	except ValueError:
		pass
 
	for _ in range(max_iter):
		eps_mid = 0.5 * (eps_low + eps_high)

		# Modify in place
		val[last] = eps_mid

		# Apply coloring and collapse
		model.coloring(thresholds=thresholds)
		model_coll = model.collapse(device).to(device)
		# Compute loss
		Lcoll = evaluate_validation(model_coll)

		if Lcoll > Lw*loss_tolerance:
			# Too large, reduce search range
			eps_high = eps_mid
		else:
			# Valid, update best and try larger eps
			best_eps = eps_mid
			eps_low = eps_mid

		# Stop if interval is small enough
		if eps_high - eps_low < tol:
			break
	
	return best_eps

def get_thresholds(model, device, loss_tolerance=1.01):
	thresholds_s = model.n_nodes(return_shape_only=True)

	model = model.to(device)
	model.eval()
 
	Lw = evaluate_validation(model)
 
	model = model.cpu()

	ps = ['q','k','v','p','l1','l2']
	for l in range(model.config.n_layer):
		for p in ps:
			eps = find_max_eps(model, thresholds_s, '.'.join([p, str(l)]), Lw, eps_low=0.0, eps_high=2, device=device, loss_tolerance=loss_tolerance, max_iter=50)
			# print(f'Layer {l}, Module {p}, Threshold {eps}', flush=True)
			thresholds_s[p][l] = eps

		# Lc = evaluate_validation(model_collapsed)
		# print(f'Layer {l} done, non-embed, non-gen compression to {1.0*number_of_params(model_collapsed)/n_params}', flush=True)
		
	# eps = find_max_eps(model, thresholds_s, 'g', Lw, eps_low=0.0, eps_high=2, device=DEVICE, loss_tolerance=loss_tolerance, max_iter=50)
	# thresholds_s['g'] = eps

	return thresholds_s


dm = DeviceManager()

# Load tiktoken encoder
enc = tiktoken.get_encoding("gpt2")

import os
model_dirs = [os.path.join('logs', f) for f in os.listdir('logs') if f.endswith('.pt')]
model_dirs.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
print(model_dirs)

for model_dir in model_dirs:
	# Load the model
	print(model_dir)
	checkpoint = torch.load(model_dir, map_location=dm.device, weights_only=False)
	model_config = checkpoint["config"]
	separate_model = GPTSeparateAttention(model_config)
	separate_model.load_state_dict(checkpoint["model"])

	separate_model.to(dm.device)
	separate_model.eval()

	init_params = number_of_params(separate_model)
	config = Config(data_manager=dm)
	config.batch_size = 8

	val_loader = DataLoaderLite(
			device_manager=dm,
			config=config,
			split="val",
			encoder=enc,
		)

	separate_model.to(dm.device)
	separate_model.eval()
	init_loss = evaluate_validation(separate_model)
	print(f'Model initial loss = {init_loss}', flush=True)

	for budget in [1.05, 1.15, 1.25]:
		thresholds = get_thresholds(separate_model, dm.device, budget)

		separate_model.coloring(thresholds)
		model_collapsed = separate_model.collapse(dm.device).to(dm.device)
		Lc = evaluate_validation(model_collapsed)
		print(f'\tReduction = {1.0*number_of_params(model_collapsed)/init_params}, Lc = {Lc}, (Lc-Lw)/Lw = {1.0*(Lc-init_loss)/init_loss}', flush=True)

	# print([(k, v.shape) for k, v in model_collapsed.state_dict().items() if ('weight' in k) or ('bias' in k)], flush=True)