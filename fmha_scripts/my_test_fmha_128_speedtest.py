import apex.contrib.fmha as fmha

import fmhalib as mha

import torch

import math

import numpy as np

import time


def old_py_mha(qkv, b, s, h, d):

    ss = 512 # s // 512
    ns = s // ss
    #ixs = 1

    qkv = qkv.view(b, s, h, 3, d)

    ctx = None
    for ixs in range(ns):
        q = qkv[:, :, :, 0, :].permute(0,2,1,3) # b h s d
        k = qkv[:, ixs*ss:(ixs+1)*ss, :, 1, :].permute(0,2,1,3) # b h is d
        v = qkv[:, ixs*ss:(ixs+1)*ss, :, 2, :].permute(0,2,1,3) # b h is d
        # b h s d * b h d is -> b h s is
        p = torch.matmul(q.float(), k.permute(0,1,3,2).float())
        p_masked = p / math.sqrt(d) # + (1.0 - amask) * -10000.0
        #s = torch.softmax(p_masked, -1).to(qkv.dtype)
        s = torch.exp(p_masked).to(qkv.dtype)
        # b h s is * b h is d -> b h s d
        if ctx is None:
            ctx = torch.matmul(s, v)
        else:
            ctx += torch.matmul(s, v)
        #ctx = torch.matmul(s, v)

    ctx = ctx.permute(0,2,1,3).contiguous() # b s h d

    ctx.retain_grad()

    return ctx


def py_mha(qkv, b, s, h, d):
    qkv = qkv.view(b, s, h, 3, d)
    q = qkv[:, :, :, 0, :].permute(0,2,1,3)
    k = qkv[:, :, :, 1, :].permute(0,2,1,3)
    v = qkv[:, :, :, 2, :].permute(0,2,1,3)
    p = torch.matmul(q.float(), k.permute(0,1,3,2).float())
    p_masked = p / math.sqrt(d)
    s = torch.softmax(p_masked, -1).to(qkv.dtype)
    ctx = torch.matmul(s, v)
    ctx = ctx.permute(0,2,1,3).contiguous()

    ctx.retain_grad()

    return ctx


def run_test(s, b):
    print(f'Test s={s} b={b}')

    zero_tensors=True

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    
    dtype = torch.float16
    device = torch.device('cuda')

    h = 16 
    d = 128

    slens = [s] * b
    a = torch.tensor(np.array([0] + slens), dtype=torch.int32)
    amask = torch.ones(b,h,s,s, dtype=dtype, device=device)
    seqlens = torch.tensor(slens, dtype=torch.int32, device=device)
    cu_seqlens = torch.cumsum(a, 0).to(dtype=torch.int32, device=device)
    total = cu_seqlens[-1].item()

    qkv = torch.randn((b,s,h,3,d), device=device, dtype=dtype)

    qkv_vs = qkv.permute(0,1,3,2,4).contiguous().view(b*s, 3, h,d)

    qkv.requires_grad = True

    st=time.time()
    iters = 50 # 2000
    for it in range(iters):
        torch.cuda.nvtx.range_push("cuda-forward")
        ctx, cmaxs, csums = mha.fwd(qkv_vs, cu_seqlens, 0.0, s, True, False, zero_tensors, None)
        torch.cuda.synchronize(device)
        torch.cuda.nvtx.range_pop()
    #print('fwd_time',s,(time.time()-st)/iters)
    fwd_time=(time.time()-st)/iters

    print(ctx.shape)

    #import pdb
    #pdb.set_trace()

    ctx = ctx.view(b,s,h,d)

    dw = ctx.permute(0,2,1,3) 

    dw2 = dw.permute(0,2,1,3).clone().detach().contiguous()

    st=time.time()
    iters = 50
    for it in range(iters):
        torch.cuda.nvtx.range_push("cuda-backward")
        dqkv2 = mha.bwd(dw2, qkv_vs, cu_seqlens, cmaxs, csums, 0.0, s, zero_tensors)[0]
        torch.cuda.synchronize(device)
        torch.cuda.nvtx.range_pop()
    bwd_time=(time.time()-st)/iters

    total_allocated = torch.cuda.max_memory_allocated() / 2**20
    print('fmha,%d,%f,%f,%f' % (s,fwd_time*1e3,bwd_time*1e3,total_allocated))

#    ctx_ref = None
#
#    iters2 = 1
#    st = time.time()
#    for it in range(iters2):
#        ctx_ref = py_mha(qkv, b,s,h,d)
#    print('py',(time.time()-st)/iters2)
#
#    print(ctx.shape, ctx_ref.shape)
#
#    assert(torch.allclose(ctx_ref.float(), ctx.float(), atol=0.5))

#    for i in range(s):
#        if not torch.allclose(ctx_ref[:,i,:,:].float(), ctx[:,i,:,:].float(), atol=0.5):
#            print(i)
#            #print(ctx_ref[:,i,:,:])
#            #print(ctx[:,i,:,:])
#
#            for j in range(d):
#                if not torch.allclose(ctx_ref[:,i,:,j].float(), ctx[:,i,:,j].float(), atol=0.5):
#                    print(j)
#                    print(ctx_ref[:,i,:,j].float(), ctx[:,i,:,j].float())
#                    break

#
#    labels = torch.randn_like(ctx_ref)
#    diff = ctx_ref - labels
#    l = (diff * diff).sum() / b
#    l.backward()
#
#    dw = ctx_ref.grad.permute(0,2,1,3) 
#
#    dw2 = dw.permute(0,2,1,3).clone().detach().contiguous()
#
#    iters = 1
#    for it in range(iters):
#        torch.cuda.nvtx.range_push("cuda-backward")
#        dqkv2, _ = mha.bwd(dw2, qkv_vs, S_, cu_seqlens, 0.0, s, zero_tensors)
#        torch.cuda.synchronize(device)
#        torch.cuda.nvtx.range_pop()
#    
#    dqkv2 = dqkv2.permute(0,2,1,3).view(b,s, h,3,d)
#
#    assert(torch.allclose(qkv.grad.float(), dqkv2.float(), atol=1e-1))



torch.cuda.cudart().cudaProfilerStart()
#run_test(2048, 10)
#run_test(384*6, 10)
torch.cuda.cudart().cudaProfilerStop()


#for i in range(1,11):
import sys
run_test(int(sys.argv[1]), 10)
