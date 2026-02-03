#!/bin/bash
torchrun --standalone --nproc-per-node 2 train_gpt2.py