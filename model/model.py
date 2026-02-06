import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import inspect
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
import copy
from .functions import TiedLinear, fibration_linear, remap_to_indices, nested_dict_from_state_dict

@dataclass
class GPTConfig:
	block_size: int = 1024
	vocab_size: int = 50257
	n_layer: int = 12
	n_head: int = 12
	n_embd: int = 768


class GPT(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config

		self.transformer = nn.ModuleDict(
			dict(
				wte=nn.Embedding(config.vocab_size, config.n_embd),
				wpe=nn.Embedding(config.block_size, config.n_embd),
				h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
				ln_f=nn.LayerNorm(config.n_embd),
			)
		)
		self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

		# weight sharing scheme
		self.transformer.wte.weight = self.lm_head.weight

		self.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			std = 0.02
			# in the original paper they scale the weights in the last projection
			# by the number of layers (sqrt) to control the variance of the output
			if hasattr(module, "dummy_flag"):
				std *= (2 * self.config.n_layer) ** -0.5
			torch.nn.init.normal_(module.weight, mean=0.0, std=std)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

	def configure_optimizers(self, weight_decay=0.1, learning_rate=3e-4, device="cpu"):
		# start with all the candidate parameters
		param_dict = {pn: p for pn, p in self.named_parameters()}
		param_dict = {pn: p for pn, p in param_dict.items()}
		# create optim groups. Any parameters that is 2D will be weight decayed,
		# otherwise not
		decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
		nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
		optim_groups = [
			{"params": decay_params, "weight_decay": weight_decay},
			{"params": nodecay_params, "weight_decay": 0.0},
		]
		num_decay_params = sum(p.numel() for p in decay_params)
		num_nodecay_params = sum(p.numel() for p in nodecay_params)
		print(f"[GPT] Number of parameters with weight decay: {num_decay_params:,}")
		print(
			f"[GPT] Number of parameters without weight decay: {num_nodecay_params:,}"
		)
		fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
		use_fused = fused_available and "cuda" in device
		# this avoids iterating over the parameters if using GPU and if option is
		# available
		print(f"\tUsing fused AdamW: {use_fused}")
		optimizer = torch.optim.AdamW(
			optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
		)
		return optimizer

	def forward(self, idx, targets=None):
		# idx is of shape (B, T)
		B, T = idx.size()

		assert (
			T <= self.config.block_size
		), "Cannot forward sequence of length %d, block size is only %d" % (
			T,
			self.config.block_size,
		)

		# forward the token and position embeddings
		pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
		pos_emb = self.transformer.wpe(pos)[None, :, :]  # (1, T, n_embd)
		tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
		x = tok_emb + pos_emb

		for block in self.transformer.h:
			x = block(x)

		x = self.transformer.ln_f(x)  # (B, T, n_embd)
		logits = self.lm_head(x)  # (B, T, vocab_size)
		loss = None
		if targets is not None:
			loss = F.cross_entropy(
				logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
			)
		return logits, loss

	def sample_sequence(
		self,
		prompt,
		data_manger,
		encoder,
		num_return_sequences,
		max_length,
		top_priority=50,
		stream=True,
	):
		self.eval()

		# encode prompt and reshape it into the desired shape
		tokens = encoder.encode(prompt)
		tokens = torch.tensor(tokens, dtype=torch.long)
		tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)

		xgen = tokens.to(data_manger.device)

		# create generator to sample from output
		sample_rng = torch.Generator(device=data_manger.device)
		sample_rng.manual_seed(42 + data_manger.ddp_rank)
		if stream:
			for seq_idx in range(num_return_sequences):
				current_xgen = tokens[seq_idx : seq_idx + 1].to(
					data_manger.device
				)  # Start with one sequence

				# Decode and print initial prompt
				initial_decoded_prompt = encoder.decode(current_xgen[0].tolist())
				print(
					initial_decoded_prompt, end="", flush=True
				)  # Print without newline and flush

				while current_xgen.size(1) < max_length:
					with torch.no_grad():
						logits, _ = self.forward(current_xgen)
						# only keep the last prediction
						logits = logits[:, -1, :]
						probs = F.softmax(logits, dim=-1)
						topk_probs, topk_indices = torch.topk(
							probs, top_priority, dim=-1
						)
						ix = torch.multinomial(
							topk_probs, num_samples=1, generator=sample_rng
						)
						xcol = torch.gather(topk_indices, dim=-1, index=ix)
						current_xgen = torch.cat((current_xgen, xcol), dim=1)

						# Decode and print the newly generated token
						new_token = xcol[0].item()  # Get the single new token
						decoded_token = encoder.decode([new_token])  # Decode it
						print(decoded_token, end="", flush=True)  # Print and flush
				print("\n")  # Add a newline after each sequence is complete.
		else:
			while xgen.size(1) < max_length:
				with torch.no_grad():
					logits, _ = self.forward(xgen)
					# only keep the last prediction
					logits = logits[:, -1, :]
					probs = F.softmax(logits, dim=-1)
					topk_probs, topk_indices = torch.topk(probs, top_priority, dim=-1)
					ix = torch.multinomial(
						topk_probs, num_samples=1, generator=sample_rng
					)
					xcol = torch.gather(topk_indices, dim=-1, index=ix)
					xgen = torch.cat((xgen, xcol), dim=1)

			for i in range(num_return_sequences):
				tokens = xgen[i, :max_length].tolist()
				# decode tokens back to vocabulary
				decoded = encoder.decode(tokens)
				print(f"rank {data_manger.ddp_rank} sample {i}: {decoded}")

	@classmethod
	def from_pretrained(cls, model_type):
		"""Loads pretrained GPT-2 model weights from huggingface"""
		assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

		from transformers import GPT2LMHeadModel

		print("loading weights from pretrained gpt: %s" % model_type)

		# n_layer, n_head and n_embd are determined from model_type
		config_args = {
			"gpt2": dict(n_layer=12, n_head=12, n_embd=768),
			"gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
			"gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
			"gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
		}[model_type]
		config_args["vocab_size"] = 50257
		config_args["block_size"] = 1024
		config = GPTConfig(**config_args)
		model = GPT(config)
		sd = model.state_dict()
		sd_keys = sd.keys()
		sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

		# init a huggingface/transformers model
		model_hf = GPT2LMHeadModel.from_pretrained(model_type)
		sd_hf = model_hf.state_dict()

		sd_keys_hf = sd_hf.keys()
		sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
		sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
		transposed = [
			"attn.c_attn.weight",
			"attn.c_proj.weight",
			"mlp.c_fc.weight",
			"mlp.c_proj.weight",
		]
		assert len(sd_keys_hf) == len(
			sd_keys
		), f"mismatched keys: {len(sd_keys_hf)} vs {len(sd_keys)}"

		for k in sd_keys_hf:
			if any(k.endswith(w) for w in transposed):
				assert sd_hf[k].shape[::-1] == sd[k].shape
				with torch.no_grad():
					sd[k].copy_(sd_hf[k].T)
			else:
				assert sd_hf[k].shape == sd[k].shape
				with torch.no_grad():
					sd[k].copy_(sd_hf[k])

		return model


class CausalSelfAttention(nn.Module):
	def __init__(self, config):
		super().__init__()
		assert config.n_embd % config.n_head == 0
		# key, query, value have n_embd // n_head features each
		self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
		# output projection
		self.c_proj = nn.Linear(config.n_embd, config.n_embd)

		self.c_proj.dummy_flag = 1
		self.n_head = config.n_head
		self.n_embd = config.n_embd
		self.register_buffer(
			"bias",
			torch.tril(torch.ones(config.block_size, config.block_size)).view(
				1, 1, config.block_size, config.block_size
			),
		)

	def forward(self, x):
		B, T, C = x.size()  # batch size, sequence length, embedding dimension
		# calculate query, key, values for all heads in batch
		# nh is "number of heads", hs is "head size", and C (number of channels)
		# is nh * hs
		qkv = self.c_attn(x)
		q, k, v = qkv.split(self.n_embd, dim=2)
		k = k.view(B, T, self.n_head, C // self.n_head).transpose(
			1, 2
		)  # (B, nh, T, hs)
		q = q.view(B, T, self.n_head, C // self.n_head).transpose(
			1, 2
		)  # (B, nh, T, hs)
		v = v.view(B, T, self.n_head, C // self.n_head).transpose(
			1, 2
		)  # (B, nh, T, hs)
		# this is the standard attention mechanism
		# att = (q @ k.transpose(-2, -1)) * (1.0 / (C // self.n_head) ** 0.5)  # (B, nh, T, T)
		# att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
		# att = F.softmax(att, dim=-1)
		# y = att @ v  # (B, nh, T, hs)
		# this uses flash-attention: way faster (especially on GPU)
		y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

		y = y.transpose(1, 2).contiguous().view(B, T, C)
		y = self.c_proj(y)
		return y


class MLP(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
		self.gelu = nn.GELU(approximate="tanh")
		self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

	def forward(self, x):
		x = self.c_fc(x)
		x = self.gelu(x)
		x = self.c_proj(x)
		return x


class Block(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.ln_1 = nn.LayerNorm(config.n_embd)
		self.attn = CausalSelfAttention(config)
		self.ln_2 = nn.LayerNorm(config.n_embd)
		self.mlp = MLP(config)

	def forward(self, x):
		x = x + self.attn(self.ln_1(x))
		x = x + self.mlp(self.ln_2(x))
		return x


class CausalSelfAttentionSeparate(nn.Module):
	"""CausalSelfAttention with separate Q, K, V linear layers for independent pruning."""
	def __init__(self, config):
		super().__init__()
		assert config.n_embd % config.n_head == 0
		# Separate linear layers for query, key, and value
		self.c_q = nn.Linear(config.n_embd, config.n_embd)
		self.c_k = nn.Linear(config.n_embd, config.n_embd)
		self.c_v = nn.Linear(config.n_embd, config.n_embd)
		# output projection
		self.c_proj = nn.Linear(config.n_embd, config.n_embd)

		self.c_proj.dummy_flag = 1
		self.n_head = config.n_head
		self.n_embd = config.n_embd
		self.register_buffer(
			"bias",
			torch.tril(torch.ones(config.block_size, config.block_size)).view(
				1, 1, config.block_size, config.block_size
			),
		)

	def forward(self, x):
		B, T, C = x.size()  # batch size, sequence length, embedding dimension
		# calculate query, key, values separately
		q = self.c_q(x)  # (B, T, n_embd)
		k = self.c_k(x)  # (B, T, n_embd)
		v = self.c_v(x)  # (B, T, n_embd)
		
		# Reshape for multi-head attention
		k = k.view(B, T, self.n_head, C // self.n_head).transpose(
			1, 2
		)  # (B, nh, T, hs)
		q = q.view(B, T, self.n_head, C // self.n_head).transpose(
			1, 2
		)  # (B, nh, T, hs)
		v = v.view(B, T, self.n_head, C // self.n_head).transpose(
			1, 2
		)  # (B, nh, T, hs)
		
		# this uses flash-attention: way faster (especially on GPU)
		y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

		y = y.transpose(1, 2).contiguous().view(B, T, C)
		y = self.c_proj(y)
		return y


class BlockSeparate(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.ln_1 = nn.LayerNorm(config.n_embd)
		self.attn = CausalSelfAttentionSeparate(config)
		self.ln_2 = nn.LayerNorm(config.n_embd)
		self.mlp = MLP(config)

	def forward(self, x):
		x = x + self.attn(self.ln_1(x))
		x = x + self.mlp(self.ln_2(x))
		return x


class GPTSeparateAttention(nn.Module):
	"""GPT model with separate Q, K, V attention layers for independent pruning."""
	def __init__(self, config):
		super().__init__()
		self.config = config

		self.transformer = nn.ModuleDict(
			dict(
				wte=nn.Embedding(config.vocab_size, config.n_embd),
				wpe=nn.Embedding(config.block_size, config.n_embd),
				h=nn.ModuleList([BlockSeparate(config) for _ in range(config.n_layer)]),
				ln_f=nn.LayerNorm(config.n_embd),
			)
		)
		self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

		# weight sharing scheme
		self.transformer.wte.weight = self.lm_head.weight

		self.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			std = 0.02
			# in the original paper they scale the weights in the last projection
			# by the number of layers (sqrt) to control the variance of the output
			if hasattr(module, "dummy_flag"):
				std *= (2 * self.config.n_layer) ** -0.5
			torch.nn.init.normal_(module.weight, mean=0.0, std=std)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

	def configure_optimizers(self, weight_decay=0.1, learning_rate=3e-4, device="cpu"):
		# start with all the candidate parameters
		param_dict = {pn: p for pn, p in self.named_parameters()}
		param_dict = {pn: p for pn, p in param_dict.items()}
		# create optim groups. Any parameters that is 2D will be weight decayed,
		# otherwise not
		decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
		nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
		optim_groups = [
			{"params": decay_params, "weight_decay": weight_decay},
			{"params": nodecay_params, "weight_decay": 0.0},
		]
		num_decay_params = sum(p.numel() for p in decay_params)
		num_nodecay_params = sum(p.numel() for p in nodecay_params)
		print(f"[GPTSeparateAttention] Number of parameters with weight decay: {num_decay_params:,}")
		print(
			f"[GPTSeparateAttention] Number of parameters without weight decay: {num_nodecay_params:,}"
		)
		fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
		use_fused = fused_available and "cuda" in device
		# this avoids iterating over the parameters if using GPU and if option is
		# available
		print(f"\tUsing fused AdamW: {use_fused}")
		optimizer = torch.optim.AdamW(
			optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
		)
		return optimizer

	def forward(self, idx, targets=None):
		# idx is of shape (B, T)
		B, T = idx.size()

		assert (
			T <= self.config.block_size
		), "Cannot forward sequence of length %d, block size is only %d" % (
			T,
			self.config.block_size,
		)

		# forward the token and position embeddings
		pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
		pos_emb = self.transformer.wpe(pos)[None, :, :]  # (1, T, n_embd)
		tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
		x = tok_emb + pos_emb

		for block in self.transformer.h:
			x = block(x)

		x = self.transformer.ln_f(x)  # (B, T, n_embd)
		logits = self.lm_head(x)  # (B, T, vocab_size)
		loss = None
		if targets is not None:
			loss = F.cross_entropy(
				logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
			)
		return logits, loss

	def sample_sequence(
		self,
		prompt,
		data_manger,
		encoder,
		num_return_sequences,
		max_length,
		top_priority=50,
		stream=True,
	):
		self.eval()

		# encode prompt and reshape it into the desired shape
		tokens = encoder.encode(prompt)
		tokens = torch.tensor(tokens, dtype=torch.long)
		tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)

		xgen = tokens.to(data_manger.device)

		# create generator to sample from output
		sample_rng = torch.Generator(device=data_manger.device)
		sample_rng.manual_seed(42 + data_manger.ddp_rank)
		if stream:
			for seq_idx in range(num_return_sequences):
				current_xgen = tokens[seq_idx : seq_idx + 1].to(
					data_manger.device
				)  # Start with one sequence

				# Decode and print initial prompt
				initial_decoded_prompt = encoder.decode(current_xgen[0].tolist())
				print(
					initial_decoded_prompt, end="", flush=True
				)  # Print without newline and flush

				while current_xgen.size(1) < max_length:
					with torch.no_grad():
						logits, _ = self.forward(current_xgen)
						# only keep the last prediction
						logits = logits[:, -1, :]
						probs = F.softmax(logits, dim=-1)
						topk_probs, topk_indices = torch.topk(
							probs, top_priority, dim=-1
						)
						ix = torch.multinomial(
							topk_probs, num_samples=1, generator=sample_rng
						)
						xcol = torch.gather(topk_indices, dim=-1, index=ix)
						current_xgen = torch.cat((current_xgen, xcol), dim=1)

						# Decode and print the newly generated token
						new_token = xcol[0].item()  # Get the single new token
						decoded_token = encoder.decode([new_token])  # Decode it
						print(decoded_token, end="", flush=True)  # Print and flush
				print("\n")  # Add a newline after each sequence is complete.
		else:
			while xgen.size(1) < max_length:
				with torch.no_grad():
					logits, _ = self.forward(xgen)
					# only keep the last prediction
					logits = logits[:, -1, :]
					probs = F.softmax(logits, dim=-1)
					topk_probs, topk_indices = torch.topk(probs, top_priority, dim=-1)
					ix = torch.multinomial(
						topk_probs, num_samples=1, generator=sample_rng
					)
					xcol = torch.gather(topk_indices, dim=-1, index=ix)
					xgen = torch.cat((xgen, xcol), dim=1)

			for i in range(num_return_sequences):
				tokens = xgen[i, :max_length].tolist()
				# decode tokens back to vocabulary
				decoded = encoder.decode(tokens)
				print(f"rank {data_manger.ddp_rank} sample {i}: {decoded}")

	@classmethod
	def from_gpt(cls, gpt_model):
		"""
		Convert a GPT model to GPTSeparateAttention by splitting the QKV attention weights.
		
		Args:
			gpt_model: An instance of GPT class
			
		Returns:
			GPTSeparateAttention model with weights copied from gpt_model
		"""
		config = gpt_model.config
		new_model = cls(config)
		
		# Get state dicts
		old_sd = gpt_model.state_dict()
		new_sd = new_model.state_dict()
		
		# Copy all non-attention weights
		for key in old_sd.keys():
			if "attn.c_attn" not in key:
				if key in new_sd:
					new_sd[key].copy_(old_sd[key])
		
		# Convert attention weights: split c_attn into c_q, c_k, c_v
		for layer_idx in range(config.n_layer):
			# Get the original combined QKV weights and biases
			old_attn_key = f"transformer.h.{layer_idx}.attn.c_attn.weight"
			old_attn_bias_key = f"transformer.h.{layer_idx}.attn.c_attn.bias"
			
			if old_attn_key in old_sd:
				old_weight = old_sd[old_attn_key]  # Shape: (n_embd * 3, n_embd)
				# Split into Q, K, V (each of shape (n_embd, n_embd))
				q_weight = old_weight[0:config.n_embd, :]
				k_weight = old_weight[config.n_embd:2*config.n_embd, :]
				v_weight = old_weight[2*config.n_embd:3*config.n_embd, :]
				
				# Copy to new model
				new_sd[f"transformer.h.{layer_idx}.attn.c_q.weight"].copy_(q_weight)
				new_sd[f"transformer.h.{layer_idx}.attn.c_k.weight"].copy_(k_weight)
				new_sd[f"transformer.h.{layer_idx}.attn.c_v.weight"].copy_(v_weight)
			
			if old_attn_bias_key in old_sd:
				old_bias = old_sd[old_attn_bias_key]  # Shape: (n_embd * 3,)
				# Split into Q, K, V biases (each of shape (n_embd,))
				q_bias = old_bias[0:config.n_embd]
				k_bias = old_bias[config.n_embd:2*config.n_embd]
				v_bias = old_bias[2*config.n_embd:3*config.n_embd]
				
				# Copy to new model
				new_sd[f"transformer.h.{layer_idx}.attn.c_q.bias"].copy_(q_bias)
				new_sd[f"transformer.h.{layer_idx}.attn.c_k.bias"].copy_(k_bias)
				new_sd[f"transformer.h.{layer_idx}.attn.c_v.bias"].copy_(v_bias)
		
		# Load the converted state dict
		new_model.load_state_dict(new_sd)
		
		return new_model
	
	def pick_param(self, d, tp='weight'):
		return d.get(f'{tp}_base', d.get(tp))

	def coloring(self, thresholds=None, n_reduction=None, silent=True):
		assert thresholds is not None or n_reduction is not None, "Please provide either thresholds or n_reduction!"
		assert not (thresholds is not None and n_reduction is not None), "Please provide either thresholds or n_reduction, not both!"
		weights = self.state_dict()
		weights = nested_dict_from_state_dict({k: v.cpu() for k, v in weights.items()})
		emb_size = self.config.n_embd
		src_colors = torch.arange(emb_size)
		in_colors = src_colors
		colors = {}
		colors['emb'] = src_colors
  
		i = 0
		for idx, _layer in tqdm(enumerate(weights['transformer']['h'].items()), disable=silent):
			layer = _layer[1]
   
			wq, wk, wv = self.pick_param(layer['attn']['c_q']), self.pick_param(layer['attn']['c_k']), self.pick_param(layer['attn']['c_v'])
			bq, bk, bv = self.pick_param(layer['attn']['c_q'], 'bias'), self.pick_param(layer['attn']['c_k'], 'bias'), self.pick_param(layer['attn']['c_v'], 'bias')
			wp = self.pick_param(layer['attn']['c_proj'])
			bp = self.pick_param(layer['attn']['c_proj'], 'bias')
   
			w1 = self.pick_param(layer['mlp']['c_fc'])
			b1 = self.pick_param(layer['mlp']['c_fc'], 'bias')
			w2 = self.pick_param(layer['mlp']['c_proj'])
			b2 = self.pick_param(layer['mlp']['c_proj'], 'bias')
   
			if thresholds is not None:
				fib_q_colors = fibration_linear(wq, in_clusters=in_colors, threshold=thresholds['q'][idx], first_layer = False, bias = bq)
				fib_k_colors = fibration_linear(wk, in_clusters=in_colors, threshold=thresholds['k'][idx], first_layer = False, bias = bk)
				fib_v_colors = fibration_linear(wv, in_clusters=in_colors, threshold=thresholds['v'][idx], first_layer = False, bias = bv)
			else:
				fib_q_colors = fibration_linear(wq, in_clusters=in_colors, n_clusters=n_reduction['q'][idx], first_layer = False, bias = bq)
				fib_k_colors = fibration_linear(wk, in_clusters=in_colors, n_clusters=n_reduction['k'][idx], first_layer = False, bias = bk)
				fib_v_colors = fibration_linear(wv, in_clusters=in_colors, n_clusters=n_reduction['v'][idx], first_layer = False, bias = bv)
	
			fib_z_colors = torch.arange(emb_size).numpy()
			if thresholds is not None:
				fib_p_colors = fibration_linear(wp, in_clusters=fib_z_colors, threshold=thresholds['p'][idx], first_layer = False, bias = bp)
			else:
				fib_p_colors = fibration_linear(wp, in_clusters=fib_z_colors, n_clusters=n_reduction['p'][idx], first_layer = False, bias = bp)
	
			fib_r1_colors = in_colors
			if len(fib_r1_colors)==len(fib_p_colors):
				_, fib_l0_colors = torch.unique(torch.stack((fib_r1_colors, torch.tensor(fib_p_colors)), dim=0).T, dim=0, return_inverse=True)
			else:		
				_, fib_l0_colors = torch.unique(torch.stack((fib_r1_colors, torch.tensor(fib_p_colors[self.transformer.h[idx].attn.c_proj.index_map.cpu()])), dim=0).T, dim=0, return_inverse=True)
   
			if thresholds is not None:
				fib_l1_colors = fibration_linear(w1, in_clusters=fib_l0_colors, threshold=thresholds['l1'][idx], first_layer = False, bias = b1)
				fib_l2_colors = fibration_linear(w2, in_clusters=fib_l1_colors, threshold=thresholds['l2'][idx], first_layer = False, bias = b2)
			else:
				fib_l1_colors = fibration_linear(w1, in_clusters=fib_l0_colors, n_clusters=n_reduction['l1'][idx], first_layer = False, bias = b1)
				fib_l2_colors = fibration_linear(w2, in_clusters=fib_l1_colors, n_clusters=n_reduction['l2'][idx], first_layer = False, bias = b2)

			fib_r2_colors = fib_l0_colors.numpy()
			_, fib_memory_colors = torch.unique(torch.stack((torch.tensor(fib_r2_colors), torch.tensor(fib_l2_colors)), dim=0).T, dim=0, return_inverse=True)

			colors_layers = {'q': torch.tensor(fib_q_colors), 'k': torch.tensor(fib_k_colors), 'v': torch.tensor(fib_v_colors),
										'z': torch.tensor(fib_z_colors), 'p': torch.tensor(fib_p_colors), 'l0': fib_l0_colors,
										'l1': torch.tensor(fib_l1_colors), 'l2': torch.tensor(fib_l2_colors), 'mem': fib_memory_colors}
   
			colors[f'{idx+1}'] = colors_layers
			in_colors = fib_memory_colors
   
		# wg = weights['lm_head']['weight']
		# bg = None
  
		# if thresholds is not None:
		# 	fib_y_colors = fibration_linear(wg, in_clusters=in_colors, threshold=thresholds['g'], first_layer = False, bias = bg)
		# else:
		# 	fib_y_colors = fibration_linear(wg, in_clusters=in_colors, n_clusters=n_reduction['g'], first_layer = False, bias = bg)

		# colors['lm_head'] = torch.tensor(fib_y_colors)
  
		self.colors = colors
  
	def collapse(self, DEVICE):
		assert hasattr(self, 'colors'), "Please run coloring() first!"
  
		model_collapsed = copy.deepcopy(self)
  
		for idx_l in range(self.config.n_layer):
			colors_q = self.colors[f'{idx_l+1}']['q']
			colors_k = self.colors[f'{idx_l+1}']['k']
			colors_v = self.colors[f'{idx_l+1}']['v']
   
			wq = self.transformer.h[idx_l].attn.c_q.weight.data
			bq = self.transformer.h[idx_l].attn.c_q.bias.data
			model_collapsed.transformer.h[idx_l].attn.c_q = TiedLinear(in_features=wq.shape[1], key=colors_q, bias=True, tie_bias=True, init_weight=wq, init_bias=bq, init_mode='mean')
			
			wk = self.transformer.h[idx_l].attn.c_k.weight.data
			bk = self.transformer.h[idx_l].attn.c_k.bias.data
			model_collapsed.transformer.h[idx_l].attn.c_k = TiedLinear(in_features=wk.shape[1], key=colors_k, bias=True, tie_bias=True, init_weight=wk, init_bias=bk, init_mode='mean')
			
			wv = self.transformer.h[idx_l].attn.c_v.weight.data
			bv = self.transformer.h[idx_l].attn.c_v.bias.data
			model_collapsed.transformer.h[idx_l].attn.c_v = TiedLinear(in_features=wv.shape[1], key=colors_v, bias=True, tie_bias=True, init_weight=wv, init_bias=bv, init_mode='mean')

			colors_p = self.colors[f'{idx_l+1}']['p']
   
			wp = self.transformer.h[idx_l].attn.c_proj.weight.data
			bp = self.transformer.h[idx_l].attn.c_proj.bias.data
			model_collapsed.transformer.h[idx_l].attn.c_proj = TiedLinear(in_features=wp.shape[1], key=colors_p, bias=True, tie_bias=True, init_weight=wp, init_bias=bp, init_mode='mean')

			colors_l1 = self.colors[f'{idx_l+1}']['l1']
			colors_l2 = self.colors[f'{idx_l+1}']['l2']

			unique_p, colors_p = torch.unique(colors_p, dim=0, return_inverse=True)
			unique_l1, colors_l1 = torch.unique(colors_l1, dim=0, return_inverse=True)

			num_colors_l1 = len(unique_l1)
   
			mtx_partition_l1 = torch.zeros(num_colors_l1, model_collapsed.transformer.h[idx_l].mlp.c_fc.out_features).scatter_(0, colors_l1.unsqueeze(0), 1).to(DEVICE)
			sizes_clusters_l1 = torch.mm(mtx_partition_l1, torch.ones(model_collapsed.transformer.h[idx_l].mlp.c_fc.out_features, 1).to(DEVICE))
   
			w1 = self.transformer.h[idx_l].mlp.c_fc.weight.data.to(DEVICE)
			b1 = self.transformer.h[idx_l].mlp.c_fc.bias.data.to(DEVICE)
			w1_coll = (mtx_partition_l1 @ w1) / sizes_clusters_l1.view(-1, 1)
			b1_coll = (mtx_partition_l1 @ b1) / sizes_clusters_l1.view(-1,)
			model_collapsed.transformer.h[idx_l].mlp.c_fc = nn.Linear(w1_coll.shape[1], w1_coll.shape[0], bias=True).to(DEVICE)
			model_collapsed.transformer.h[idx_l].mlp.c_fc.weight.data = w1_coll
			model_collapsed.transformer.h[idx_l].mlp.c_fc.bias.data = b1_coll

			w2 = self.transformer.h[idx_l].mlp.c_proj.weight.data.to(DEVICE)
			b2 = self.transformer.h[idx_l].mlp.c_proj.bias.data.to(DEVICE)
			w2_coll = w2 @ mtx_partition_l1.T
			b2_coll = b2
			model_collapsed.transformer.h[idx_l].mlp.c_proj = nn.Linear(w2_coll.shape[1], w2_coll.shape[0], bias=True).to(DEVICE)
			model_collapsed.transformer.h[idx_l].mlp.c_proj.weight.data = w2_coll
			model_collapsed.transformer.h[idx_l].mlp.c_proj.bias.data = b2_coll
   
			model_collapsed.transformer.h[idx_l].mlp.c_proj = TiedLinear(key=colors_l2, in_features=w2_coll.shape[1], init_weight=w2_coll, init_bias=b2_coll, init_mode='mean').to(DEVICE)


		return model_collapsed

	def n_nodes(self, return_shape_only=False):
		weights = nested_dict_from_state_dict({k: v.cpu() for k, v in self.state_dict().items()})
  
		n_nodes = {
			'emb': 0,
			'q': [0] * self.config.n_layer,
			'k': [0] * self.config.n_layer,
			'v': [0] * self.config.n_layer,
			'p': [0] * self.config.n_layer,
			'l1': [0] * self.config.n_layer,
			'l2': [0] * self.config.n_layer,
			'g': 0
		}

		if return_shape_only:
			return n_nodes

		n_nodes['emb'] = weights['transformer']['wte']['weight'].shape[1]

		for idx, _layer in enumerate(weights['transformer']['h'].items()):
			layer = _layer[1]
			if hasattr(self.transformer.encoder.layers[idx].self_attn, 'mha'):
				n_nodes['q'][idx] = layer['attn']['q_weight_proto'].shape[0]
				n_nodes['k'][idx] = layer['self_attn']['k_weight_proto'].shape[0]
				n_nodes['v'][idx] = layer['self_attn']['v_weight_proto'].shape[0]
				n_nodes['p'][idx] = layer['self_attn']['mha']['out_proj']['weight_base'].shape[0]

			else:
				wqkv 		= layer['self_attn']['in_proj_weight'].shape[0]
				n_nodes['encoder']['q'][idx], n_nodes['encoder']['k'][idx], n_nodes['encoder']['v'][idx] = wqkv // 3, wqkv // 3, wqkv // 3
				n_nodes['encoder']['p'][idx] = layer['self_attn']['out_proj']['weight'].shape[0]

			if hasattr(self.transformer.encoder.layers[idx].linear1, 'weight_base'):
				n_nodes['encoder']['l1'][idx] = layer['linear1']['weight_base'].shape[0]
			else:
				n_nodes['encoder']['l1'][idx] = layer['linear1']['weight'].shape[0]

			if hasattr(self.transformer.encoder.layers[idx].linear2, 'weight_base'):
				n_nodes['encoder']['l2'][idx] = layer['linear2']['weight_base'].shape[0]
			else:
				n_nodes['encoder']['l2'][idx] = layer['linear2']['weight'].shape[0]

		for idx, _layer in enumerate(weights['transformer']['decoder']['layers'].items()):
			layer = _layer[1]

			if hasattr(self.transformer.decoder.layers[idx].multihead_attn, 'mha'):
				n_nodes['decoder']['q_m'][idx] = layer['multihead_attn']['q_weight_proto'].shape[0]
				n_nodes['decoder']['k_m'][idx] = layer['multihead_attn']['k_weight_proto'].shape[0]
				n_nodes['decoder']['v_m'][idx] = layer['multihead_attn']['v_weight_proto'].shape[0]
				n_nodes['decoder']['p_m'][idx] = layer['multihead_attn']['mha']['out_proj']['weight_base'].shape[0]
			else:
				wqkv 		= layer['multihead_attn']['in_proj_weight'].shape[0]
				n_nodes['decoder']['q_m'][idx], n_nodes['decoder']['k_m'][idx], n_nodes['decoder']['v_m'][idx] = wqkv // 3, wqkv // 3, wqkv // 3
				n_nodes['decoder']['p_m'][idx] = layer['multihead_attn']['out_proj']['weight'].shape[0]

			if hasattr(self.transformer.decoder.layers[idx].self_attn, 'mha'):
				n_nodes['decoder']['q'][idx] = layer['self_attn']['q_weight_proto'].shape[0]
				n_nodes['decoder']['k'][idx] = layer['self_attn']['k_weight_proto'].shape[0]
				n_nodes['decoder']['v'][idx] = layer['self_attn']['v_weight_proto'].shape[0]
				n_nodes['decoder']['p'][idx] = layer['self_attn']['mha']['out_proj']['weight_base'].shape[0]
			else:
				wqkv 		= layer['self_attn']['in_proj_weight'].shape[0]
				n_nodes['decoder']['q'][idx], n_nodes['decoder']['k'][idx], n_nodes['decoder']['v'][idx] = wqkv // 3, wqkv // 3, wqkv // 3
				n_nodes['decoder']['p'][idx] = layer['self_attn']['out_proj']['weight'].shape[0]

			if hasattr(self.transformer.decoder.layers[idx].linear1, 'weight_base'):
				n_nodes['decoder']['l1'][idx] = layer['linear1']['weight_base'].shape[0]
			else:
				n_nodes['decoder']['l1'][idx] = layer['linear1']['weight'].shape[0]

			if hasattr(self.transformer.decoder.layers[idx].linear2, 'weight_base'):
				n_nodes['decoder']['l2'][idx] = layer['linear2']['weight_base'].shape[0]
			else:
				n_nodes['decoder']['l2'][idx] = layer['linear2']['weight'].shape[0]

		if hasattr(self.generator, 'weight_base'):
			n_nodes['g'] = weights['generator']['weight_base'].shape[0]
		else:
			n_nodes['g'] = weights['generator']['weight'].shape[0]
   
		return n_nodes