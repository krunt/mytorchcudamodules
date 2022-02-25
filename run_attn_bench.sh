#!/bin/bash

seq_length=2048
num_seqs_start=8
num_seqs_stop=8
layers=1
hidden_dim=2048
heads=16
attn_type='--ref'

#PYTORCH_NO_CUDA_MEMORY_CACHING=1 nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu --cuda-memory-usage=true --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --cudabacktrace all --force-overwrite true -x true -o myattnmem_profile python3 perf_test_multihead_attn.py --trials 1 --warmup-trials 0 --seq-length=$seq_length --num-seqs-start=$num_seqs_start --num-seqs-stop=$num_seqs_stop --layers=$layers --hidden-dim=$hidden_dim --heads=$heads --norm-add $attn_type
#python3 perf_test_multihead_attn.py --trials 1 --warmup-trials 0 --seq-length=$seq_length --num-seqs-start=$num_seqs_start --num-seqs-stop=$num_seqs_stop --layers=$layers --hidden-dim=$hidden_dim --heads=$heads $attn_type

CUDA_LAUNCH_BLOCKING=1 PYTORCH_NO_CUDA_MEMORY_CACHING=1 nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu --cuda-memory-usage=true --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --cudabacktrace all --force-overwrite true -x true -o myattnmem_profile python3 perf_test_multihead_attn.py --trials 2 --warmup-trials 0 --seq-length=$seq_length --num-seqs-start=$num_seqs_start --num-seqs-stop=$num_seqs_stop --layers=$layers --hidden-dim=$hidden_dim --heads=$heads $attn_type

