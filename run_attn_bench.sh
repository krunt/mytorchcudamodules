#!/bin/bash

#seq_length=1024
#num_seqs_start=10
#num_seqs_stop=10
#layers=4
#hidden_dim=1024
#heads=16

seq_length=2048
num_seqs_start=8
num_seqs_stop=8
layers=1
hidden_dim=2048
heads=16


for attn_type in '--ref' '--native' '--lean' ; do 

  python3 perf_test_multihead_attn.py --seq-length=$seq_length --num-seqs-start=$num_seqs_start --num-seqs-stop=$num_seqs_stop --layers=$layers --hidden-dim=$hidden_dim --heads=$heads --norm-add $attn_type

done
