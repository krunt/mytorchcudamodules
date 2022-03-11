import torch
import torch.nn.functional as F
import argparse

import modules.initialize as minit

minit.initialize(verbose=False)

#from apex.contrib.multihead_attn import EncdecMultiheadAttn
#from apex.contrib.multihead_attn import SelfMultiheadAttn
from modules import SelfMultiheadAttn

import sys
sys.path.append('../petabert/')
from lib.modules.attn import LeanSelfAttention
#from lib.modules.masked_attn import LeanSelfAttention

parser = argparse.ArgumentParser(description='Multihead Attention Standalone Test')
parser.add_argument('--seq-length', default=64, type=int, help='Sequence Length of Input')
parser.add_argument('--num-seqs-start', default=10, type=int, help='Start Range of Number of Sequences')
parser.add_argument('--num-seqs-stop', default=120, type=int, help='Stop Range of Number of Sequences')
parser.add_argument('--num-seqs-inc', default=5, type=int, help='Range Increment of Number of Sequences')
parser.add_argument('--trials', default=20, type=int, help='Number of Trials to Execute')
parser.add_argument('--warmup-trials', default=5, type=int, help='Warmup Trials to discard')
parser.add_argument('--layers', default=18, type=int, help='Attention Layers to Execute to Gain CPU/GPU Time Overlap')
parser.add_argument('--hidden-dim', default=1024, type=int, help='Multihead Attention hidden dimension')
parser.add_argument('--heads', default=16, type=int, help='Number of Multihead Attention heads')
parser.add_argument('--encdec-attn', action='store_true', help='Use Encoder-Decoder Attention instead of Self Attention.')
parser.add_argument('--norm-add', action='store_false', help='Include Layer Norm and Dropout-Add in Multihead Attention block.')
parser.add_argument('--ref', action='store_true', help='Reference implementation in python pytorch.')
parser.add_argument('--native', action='store_true', help='torch.nn.MultitheadAttention Version.')
parser.add_argument('--lean', action='store_true', help='use LeanSelfAttention from petabert.')
parser.add_argument('--fwd', action='store_true', help='Only execute Fwd Pass.')
parser.add_argument('--biases', action='store_true', help='Execute multihead attention with Linear Biases.')

args = parser.parse_args()

if not torch.cuda.is_available():
    raise NotImplementedError('Running on CPU is not supported')
torch.cuda.set_device(0)

torch.manual_seed(111)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(111)

model_attn_type = ''
if args.lean:
    model_attn_type = 'lean'
elif args.ref:
    model_attn_type = 'apex-python'
elif args.native:
    model_attn_type = 'native'
else:
    model_attn_type = 'apex-cpp'

torch.cuda.cudart().cudaProfilerStart()

attn_layers = []
for idx in range(0, args.layers) :
#    if args.encdec_attn:
#        if args.lean:
#            assert 0, "not supported"
#        elif args.ref :
#            attn_layers.append(EncdecMultiheadAttn(args.hidden_dim, args.heads, dropout=0.1, bias=args.biases, include_norm_add=False, impl='default'))
#        else :
#            attn_layers.append(EncdecMultiheadAttn(args.hidden_dim, args.heads, dropout=0.1, bias=args.biases, include_norm_add=args.norm_add, impl='fast'))
#    else :
    if args.lean:
        attn_layers.append(LeanSelfAttention(args.hidden_dim, args.heads, residual=False, checkpoint_attention_core=False))
    elif args.native:
        attn_layers.append(torch.nn.MultiheadAttention(args.hidden_dim, args.heads, bias=args.biases))
    elif args.ref:
        attn_layers.append(SelfMultiheadAttn(args.hidden_dim, args.heads, bias=args.biases, include_norm_add=args.norm_add, impl='default'))
    else :
        attn_layers.append(SelfMultiheadAttn(args.hidden_dim, args.heads, bias=args.biases, include_norm_add=args.norm_add, impl='fast'))
    attn_layers[idx].cuda()
    attn_layers[idx].half()
    if not (args.native or args.lean):
        attn_layers[idx].reset_parameters()

start_evt_fwd = []
start_evt_bwd = []
stop_evt_bwd  = []
for recorded_trial in range(0, args.trials) :
    start_evt_fwd.append(torch.cuda.Event(enable_timing=True))
    start_evt_bwd.append(torch.cuda.Event(enable_timing=True))
    stop_evt_bwd.append(torch.cuda.Event(enable_timing=True))

for sequences in range(args.num_seqs_start, args.num_seqs_stop + args.num_seqs_inc, args.num_seqs_inc) :
    inputs        = torch.randn(args.seq_length, sequences, args.hidden_dim, dtype=torch.float16, device=torch.device("cuda")).requires_grad_(True)
    grads         = torch.randn_like(inputs)
   
    for trial in range(0, args.trials + args.warmup_trials) :
        layer_inputs  = inputs
        evt_idx       = trial - args.warmup_trials
    
        if evt_idx >= 0 :
            start_evt_fwd[evt_idx].record()
    
        torch.cuda.nvtx.range_push("forward")
        for lyr_idx in range(0, args.layers) :
            if args.lean :
                outputs, = attn_layers[lyr_idx].forward(layer_inputs.permute(1, 0, 2))
                outputs = outputs.permute(1, 0, 2)
            elif args.native :
                outputs,_ = attn_layers[lyr_idx].forward(layer_inputs, 
                                                         layer_inputs, 
                                                         layer_inputs, 
                                                         key_padding_mask=None, 
                                                         need_weights=False, 
                                                         attn_mask=None)
            else :
                layer_inputs = layer_inputs.permute(1, 0, 2)
                outputs,_ = attn_layers[lyr_idx].forward(layer_inputs, 
                                                         layer_inputs, 
                                                         layer_inputs,
                                                         key_padding_mask=None, 
                                                         need_weights=False, 
                                                         attn_mask=None,
                                                         is_training=True)
                outputs = outputs.permute(1, 0, 2)
            layer_inputs = outputs
        torch.cuda.nvtx.range_pop()
    
        if evt_idx >= 0 :
            start_evt_bwd[evt_idx].record()

        torch.cuda.nvtx.range_push("backward")
        if not args.fwd :
            layer_inputs.backward(grads)
        torch.cuda.nvtx.range_pop()
    
        if evt_idx >= 0 :
            stop_evt_bwd[evt_idx].record()
   
    torch.cuda.synchronize()
    elapsed_time_fwd = 0.0
    elapsed_time_bwd = 0.0
    for evt_idx in range(0, args.trials) :
        elapsed_time_fwd += start_evt_fwd[evt_idx].elapsed_time(start_evt_bwd[evt_idx])
        elapsed_time_bwd += start_evt_bwd[evt_idx].elapsed_time(stop_evt_bwd[evt_idx])
 
    total_allocated = torch.cuda.max_memory_allocated() / 2**20
   
    print("type={} layers={} seqlen={} seqcnt={} hidden={} fwd_ms={:.3f} bwd_ms={:.3f} total_ms={:.3f} total_mb={:.3f}".format(model_attn_type,
                args.layers, args.seq_length, sequences, args.hidden_dim,
                elapsed_time_fwd / ( args.trials * args.layers ), elapsed_time_bwd / ( args.trials * args.layers ), 
                (elapsed_time_fwd + elapsed_time_bwd) / ( args.trials * args.layers ),
                total_allocated))

torch.cuda.cudart().cudaProfilerStop()
