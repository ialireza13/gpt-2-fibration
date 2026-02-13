import torch
import time
import math
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from data.hellaswag import evaluate_benchmark
from model.model import GPTConfig, GPTSeparateAttention
from manager.device_manager import DeviceManager
from config import Config
from manager.log_manager import LogManager
from data.dataloader import DataLoaderLite
import torch.distributed as dist
import tiktoken
from tqdm import tqdm
from model.functions import number_of_params


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
	coarse_val_steps=10,
	fine_val_steps=20,
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
		Lcoll = evaluate_validation_greedy(model_coll, val_loss_steps=coarse_val_steps)
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
		Lcoll = evaluate_validation_greedy(model_coll, val_loss_steps=fine_val_steps)
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
 
	Lw = evaluate_validation_greedy(model)
 
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


def evaluate_validation_greedy(model, val_loss_steps = 20):
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

def get_lr(it, config):
	if it < config.warmup_steps:
		return config.max_lr * (it + 1) / config.warmup_steps
	if it > config.max_steps:
		return config.min_lr

	decay_ratio = (it - config.warmup_steps) / (config.max_steps - config.warmup_steps)
	assert 0 <= decay_ratio <= 1, "decay ratio out of bounds: %f" % decay_ratio
	coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
	return config.min_lr + coeff * (config.max_lr - config.min_lr)


def evaluate_validation(step, model, data_manager, log_manager):
	model.eval()
	val_loader.reset()
	with torch.no_grad():
		val_loss_accum = 0.0
		val_loss_steps = 20
		for _ in range(val_loss_steps):
			x, y = val_loader.next_batch()
			x, y = x.to(dm.device), y.to(dm.device)
			with torch.autocast(device_type=dm.device, dtype=torch.bfloat16):
				_, loss = model(x, y)
			loss /= val_loss_steps
			val_loss_accum += loss.detach()
	if data_manager.ddp:
		dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
	if data_manager.master_process:
		print(f"validation loss at step {step}: {val_loss_accum.item():.4f}")
		log_manager.to_file(step, "val", val_loss_accum)


def save_model(model, optimizer, log_manager, step, filename):
	checkpoint_path = os.path.join(log_manager.dir, f"{filename}.pt")
	checkpoint = {
		"model": model.state_dict(),
		"config": model.config,
		"step": step,
		"optimizer": optimizer.state_dict(),
		"rng_state": torch.get_rng_state(),
	}
	if torch.cuda.is_available():
		checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()
	torch.save(checkpoint, checkpoint_path)


if __name__ == "__main__":
	# we set TF32
	torch.set_float32_matmul_precision("high")

	dm = DeviceManager()

	torch.manual_seed(1337)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(1337)

	# this is a trick to be able to use the same batch size as the original GPT-2
	# without having to load huge batches into memory
	config = Config(data_manager=dm)

	enc = tiktoken.get_encoding("gpt2")
	train_loader = DataLoaderLite(
		device_manager=dm,
		config=config,
		split="train",
		encoder=enc,
	)
	val_loader = DataLoaderLite(
		device_manager=dm,
		config=config,
		split="val",
		encoder=enc,
	)

	# we overwrite the vocab size (froom 50257 to 50304) to make the number "nice"
	# (it can be divided by many powers of 2)
	# model = GPT(GPTConfig(vocab_size=50304))
	model = GPTSeparateAttention(GPTConfig(vocab_size=50304))
	
	model_dir = "log/model_1907"
	resume_start_step = int(model_dir.split("_")[-1])
	resume_checkpoint = torch.load(model_dir, map_location=dm.device, weights_only=False)
	model_config = resume_checkpoint["config"]
	model = GPTSeparateAttention(model_config)
	model.load_state_dict(resume_checkpoint["model"])

	model.to(dm.device)
	
	model.eval()
	t1 = time.time()
	init_loss = evaluate_validation_greedy(model)
	init_params = number_of_params(model)
	print(f'Model initial loss = {init_loss}, time = {time.time() - t1}', flush=True)
	thresholds = get_thresholds(model, dm.device, 1.15, silent=True)
	t1 = time.time()
	model.coloring(thresholds)
	model = model.collapse(dm.device).to(dm.device)
	Lc = evaluate_validation_greedy(model)
	print(f'\tReduction = {1.0*number_of_params(model)/init_params}, Lc = {Lc}, (Lc-Lw)/Lw = {1.0*(Lc-init_loss)/init_loss}, time = {time.time() - t1}', flush=True)

	use_compile = False
	if torch.cuda.is_available() and use_compile:
		torch.compile(model)
	if dm.ddp:
		model = DDP(model, device_ids=[dm.ddp_local_rank])
	model = model.module if dm.ddp else model

	optimizer = model.configure_optimizers(
		weight_decay=0.1, learning_rate=3e-4, device=dm.device
	)

	# object use to print to file metrics (train, validation, benchmark performance)
	lm = LogManager(dir="log")
	
	if config.continue_from:
		lr = get_lr(resume_start_step, config)
		for param_group in optimizer.param_groups:
			param_group["lr"] = lr

	start_step = resume_start_step + 1
	for step in range(start_step, config.max_steps):
		t0 = time.time()
		last_step = step == config.max_steps - 1

		if (step % config.log_step == 0 or last_step) and (not use_compile):
			prompt = "Hello, I'm a language model,"
			model.sample_sequence(
				prompt,
				dm,
				enc,
				config.num_return_sequences_sample_training,
				config.max_length_sample_training,
				top_priority=50,
				stream=False,
			)
			if config.do_evaluate_benchmark:
				evaluate_benchmark(step, model, dm, lm)
			evaluate_validation(step, model, dm, lm)

		if step % config.model_output_step == 0 or last_step:
			save_model(model, optimizer, lm, step, f"model_{step}")

		model.train()
		optimizer.zero_grad()
		loss_accum = 0.0
		for micro_step in range(config.grad_accum_steps):
			x, y = train_loader.next_batch()
			# useful line to use float16 precision for training
			if torch.cuda.is_available():
				with torch.autocast(device_type=dm.device, dtype=torch.bfloat16):
					logits, loss = model(x.to(dm.device), y.to(dm.device))
			# on MPS, autocast to bfloat16 is slower, so we use float32
			else:
				logits, loss = model(x.to(dm.device), y.to(dm.device))
			# we need to divide the loss by the number of steps to make sure that we get
			# the same loss as if we had not accumulated gradients
			loss /= config.grad_accum_steps
			loss_accum += loss.detach()  # accumulate loss for logging
			# this is to ensure that the communication of gradients only happens at the
			# very last step to avoid unnecessary communication overhead
			if dm.ddp:
				model.require_backward_grad_sync = (
					micro_step == config.grad_accum_steps - 1
				)
			# here we accumulate the steps
			loss.backward()
		# this is to ensure that the loss is averaged across all processes
		if dm.ddp:
			dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
		loss_accum = loss_accum.item()  # convert to python float for logging
		# clip gradients to avoid exploding gradients
		norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

		# update learning rate
		lr = get_lr(step, config)
		for param_group in optimizer.param_groups:
			param_group["lr"] = lr

		# optimizer step
		optimizer.step()

		# synchronize across GPUS
		if torch.cuda.is_available():
			torch.cuda.synchronize()

		# compute times
		t1 = time.time()
		dt = (t1 - t0) * 1000
		tokens_per_second = (
			(x.size(0) * x.size(1) * config.grad_accum_steps)
			/ (dt / 1000.0)
			* dm.ddp_world_size
		)
		if dm.master_process:
			print(
				f"step {step} | loss: {loss_accum:.4f} | lr: {lr:.3e} | norm: {norm:.4f} | time: {dt:.2f} ms | tokens/sec: {tokens_per_second:.2f}", flush=True
			)
			lm.to_file(step, "train", loss_accum)

	dm.terminate()
