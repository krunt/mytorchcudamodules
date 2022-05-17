#!/bin/bash

script=${1:-myattnmem_profile}

#PYTORCH_NO_CUDA_MEMORY_CACHING=1 nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu --cuda-memory-usage=true --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --cudabacktrace all --force-overwrite true -x true -o $script python3 perf_test_multihead_attn.py --trials 1 --warmup-trials 0 --seq-length=$seq_length --num-seqs-start=$num_seqs_start --num-seqs-stop=$num_seqs_stop --layers=$layers --hidden-dim=$hidden_dim --heads=$heads --norm-add $attn_type
#python3 perf_test_multihead_attn.py --trials 1 --warmup-trials 0 --seq-length=$seq_length --num-seqs-start=$num_seqs_start --num-seqs-stop=$num_seqs_stop --layers=$layers --hidden-dim=$hidden_dim --heads=$heads $attn_type

nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu --cuda-memory-usage=true --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --cudabacktrace all --force-overwrite true -x true -o $script python3 my_test_fmha_64_speedtest.py


