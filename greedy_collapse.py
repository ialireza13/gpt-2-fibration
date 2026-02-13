import torch
import os
import tiktoken
import time
from manager.device_manager import DeviceManager
from model.model import GPTSeparateAttention
from config import Config
from tqdm import tqdm
from data.dataloader import DataLoaderLite
from config import Config
from data.hellaswag import iterate_examples, render_example, get_most_likely_row
from model.functions import number_of_params
import os

def evaluate_benchmark(model, data_manager):
    num_correct_norm = 0
    num_total = 0
    for i, example in enumerate(iterate_examples("val")):
        if i % data_manager.ddp_world_size != data_manager.ddp_rank:
            continue
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(data_manager.device)
        mask = mask.to(data_manager.device)
        with torch.no_grad():
            with torch.autocast(device_type=data_manager.device, dtype=torch.bfloat16):
                logits, loss = model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)

    acc_norm = num_correct_norm / num_total
    return acc_norm


def evaluate_validation(model, val_loss_steps = 30):
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


def find_max_eps(
	model,
	thresholds,
	path,
	Lw,
	device,
	eps_low=0.0,
	eps_high=2.0,
	tol=1e-2,
	max_iter=50,
	loss_tolerance=1.01,
	coarse_val_steps=5,
	fine_val_steps=10,
):
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
 
	# Track best epsilon found so far
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
 
	# ------------------------------
	# Phase 1: coarse search (fewer batches)
	# ------------------------------
	coarse_iters = max_iter // 2
	low_c, high_c = eps_low, eps_high
	best_c = best_eps

	for _ in range(coarse_iters):
		eps_mid = 0.5 * (low_c + high_c)

		# Modify in place
		val[last] = eps_mid

		# Apply coloring and collapse
		model.coloring(thresholds=thresholds)
		model_coll = model.collapse(device).to(device)
		Lcoll = evaluate_validation(model_coll, val_loss_steps=coarse_val_steps)
		# print(f'Coarse Layer {last}, Threshold {eps_mid}, Lcoll = {Lcoll}', flush=True)
		if Lcoll > Lw * loss_tolerance:
			high_c = eps_mid
		else:
			best_c = eps_mid
			low_c = eps_mid

		if high_c - low_c < 1e-1:
			break

	# ------------------------------
	# Phase 2: fine search (full 160 batches)
	# ------------------------------
	fine_iters = max_iter - coarse_iters
	low_f, high_f = low_c, high_c
	best_f = best_c

	for _ in range(fine_iters):
		eps_mid = 0.5 * (low_f + high_f)

		val[last] = eps_mid

		model.coloring(thresholds=thresholds)
		model_coll = model.collapse(device).to(device)
		Lcoll = evaluate_validation(model_coll, val_loss_steps=fine_val_steps)
		# print(f'Fine Layer {last}, Threshold {eps_mid}, Lcoll = {Lcoll}', flush=True)
		if Lcoll > Lw * loss_tolerance:
			high_f = eps_mid
		else:
			best_f = eps_mid
			low_f = eps_mid

		if high_f - low_f < tol:
			break

	best_eps = best_f

	return best_eps

def get_thresholds(model, device, loss_tolerance=1.01, silent=True):
	thresholds_s = model.n_nodes(return_shape_only=True)

	model = model.to(device)
	model.eval()
 
	Lw = evaluate_validation(model)
 
	model = model.cpu()

	ps = ['q','k','v','p','l1','l2']
	for l in tqdm(range(model.config.n_layer), disable=silent):
		for p in ps:
			eps = find_max_eps(model, thresholds_s, '.'.join([p, str(l)]), Lw, eps_low=0.0, eps_high=2, device=device, loss_tolerance=loss_tolerance)
			# print(f'Layer {l}, Module {p}, Threshold {eps}', flush=True)
			thresholds_s[p][l] = eps

		# Lc = evaluate_validation(model_collapsed)
		# print(f'Layer {l} done, non-embed, non-gen compression to {1.0*number_of_params(model_collapsed)/n_params}', flush=True)
		
	# eps = find_max_eps(model, thresholds_s, 'g', Lw, eps_low=0.0, eps_high=2, device=DEVICE, loss_tolerance=loss_tolerance, max_iter=50)
	# thresholds_s['g'] = eps

	return thresholds_s


dm = DeviceManager()
config = Config(data_manager=dm)
config.batch_size = 32

# Load tiktoken encoder
enc = tiktoken.get_encoding("gpt2")
val_loader = DataLoaderLite(
			device_manager=dm,
			config=config,
			split="val",
			encoder=enc,
		)
 
 
total_batches = val_loader.num_batches()  # total batches in one full pass (this process)

model_dirs = [os.path.join('logs', f) for f in os.listdir('logs') if f.endswith('.pt') and f.startswith('model_1B_')]
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
	
	separate_model.to(dm.device)
	separate_model.eval()
	t1 = time.time()
	init_loss = evaluate_validation(separate_model)
	init_loss_benchmark = evaluate_benchmark(separate_model, dm)
	print(f'Model initial loss = {init_loss}, benchmark = {init_loss_benchmark}, time = {time.time() - t1}', flush=True)

	for budget in [1.1]:
		t1 = time.time()
		thresholds = get_thresholds(separate_model, dm.device, budget, silent=True)

		separate_model.coloring(thresholds)
		model_collapsed = separate_model.collapse(dm.device).to(dm.device)
		Lc = evaluate_validation(model_collapsed)
		Lc_benchmark = evaluate_benchmark(model_collapsed, dm)
		print(f'\tReduction = {1.0*number_of_params(model_collapsed)/init_params}, Lc = {Lc}, benchmark = {Lc_benchmark}, (Lc-Lw)/Lw = {1.0*(Lc-init_loss)/init_loss}, time = {time.time() - t1}', flush=True)

	# print([(k, v.shape) for k, v in model_collapsed.state_dict().items() if ('weight' in k) or ('bias' in k)], flush=True)