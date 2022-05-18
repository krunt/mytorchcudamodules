import apex.contrib.fmha as fmha

import fmhalib as mha

import torch

import math

import numpy as np

import time


def nullify_buffer_bylen(buf, slens):
    for i in range(len(slens)):
        buf[i, slens[i]:, ...] = 0
    return buf

def py_mha(qkv, amask, b, s, h, d, slens):
    qkv = qkv.view(b, s, h, 3, d)
    q = qkv[:, :, :, 0, :].permute(0,2,1,3)
    k = qkv[:, :, :, 1, :].permute(0,2,1,3)
    v = qkv[:, :, :, 2, :].permute(0,2,1,3) # b h s d
    p = torch.matmul(q.float(), k.permute(0,1,3,2).float())
    p_masked = p / math.sqrt(d) + (1.0 - amask) * -10000.0
    s = torch.softmax(p_masked, -1).to(qkv.dtype)
    ctx = torch.matmul(s, v) # (b,h,s,s) * (b,h,s,d) -> (b,h,s,d)
    ctx = ctx.permute(0,2,1,3).contiguous() # (b,s,h,d)

    nullify_buffer_bylen(ctx, slens)

    ctx.retain_grad()

    return ctx

def pack_buffer_seq(src, dst, slens):
    it = 0
    for i in range(src.shape[0]):
        for j in range(slens[i]):
            dst[it, ...] = src[i, j, ...]
            it += 1
    return dst

def unpack_buffer_seq(src, dst, slens):
    it = 0
    for i in range(dst.shape[0]):
        for j in range(slens[i]):
            dst[i, j, ...] = src[it, ...]
            it += 1
    return dst


def run_test(s, b, h = 16, d = 64):
    print(f'Test s={s} b={b}')

    zero_tensors=True

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    
    dtype = torch.float16
    device = torch.device('cuda')


    slens = list(np.random.randint(32, s+1, b))
    print(slens)
    #slens = list(np.random.randint(s, s+1, b))
    #slens = [s] * b
    #slens = [s-1,s]

    #print(slens)

    a = torch.tensor(np.array([0] + slens), dtype=torch.int32)
    amask = torch.zeros(b,h,s,s, dtype=dtype, device=device)

    for i in range(b):
        slen = slens[i]
        amask[i, :, :slen, :slen] = 1

    seqlens = torch.tensor(slens, dtype=torch.int32, device=device)
    cu_seqlens = torch.cumsum(a, 0).to(dtype=torch.int32, device=device)
    total = cu_seqlens[-1].item()

    qkv = torch.randn((b,s,h,3,d), device=device, dtype=dtype)

    #with torch.no_grad():
        #qkv[:, :, 1, :, :] = qkv[:, :, 0, :, :]

    #qkv_vs = qkv.permute(0,1,3,2,4).contiguous().view(b*s, 3, h,d)
    #print('qkv_vs', qkv_vs.shape)

    qkv_vs = torch.zeros((total,h,3,d), device=device, dtype=dtype)
    qkv_vs = pack_buffer_seq(qkv, qkv_vs, slens)
    qkv_vs = qkv_vs.permute(0, 2, 1, 3).contiguous() # total, 3, h, d

    qkv.requires_grad = True

    st=time.time()
    iters = 1 # 2000
    for it in range(iters):
        torch.cuda.nvtx.range_push("cuda-forward")
        ctx, cmaxs, csums  = mha.fwd(qkv_vs, cu_seqlens, 0.0, s, True, False, zero_tensors, None)
        torch.cuda.synchronize(device)
        torch.cuda.nvtx.range_pop()
    print('cuda',(time.time()-st)/iters)

    ctx1 = torch.zeros((b,s,h,d), device=device, dtype=dtype)
    ctx = unpack_buffer_seq(ctx, ctx1, slens)
    #ctx = ctx.view(b,s,h,d)

    #nullify_buffer_bylen(ctx, slens)

    ctx_ref = None

    iters2 = 1
    st = time.time()
    for it in range(iters2):
        torch.cuda.nvtx.range_push("py-forward")
        ctx_ref = py_mha(qkv, amask, b,s,h,d, slens)
        torch.cuda.nvtx.range_pop()
    print('py',(time.time()-st)/iters2)

    #assert(torch.allclose(ctx_ref.float(), ctx.float(), atol=0.4)) # 1e-2))
    #print('ctx_ref',ctx_ref)
    #print('ctx',ctx)
    assert(torch.allclose(ctx_ref.float(), ctx.float(), atol=0.4)) # 1e-2))

    labels = torch.randn_like(ctx_ref)
    diff = ctx_ref - labels
    l = (diff * diff).sum() / b
    l.backward()

    dw = ctx_ref.grad.permute(0,2,1,3) 

    dw2 = dw.permute(0,2,1,3).clone().detach().contiguous()

    #print('dw2',dw2.shape)
    #print(dw2[:, :, 0, :20], dw2[:, :, 1, :20])

    dw21 = torch.zeros((total,h,d), device=device, dtype=dtype)
    dw2 = pack_buffer_seq(dw2, dw21, slens)

    iters = 1
    for it in range(iters):
        torch.cuda.nvtx.range_push("cuda-backward")
        dqkv2, = mha.bwd(dw2, qkv_vs, cu_seqlens, cmaxs, csums, 0.0, s, zero_tensors)
        torch.cuda.synchronize(device)
        torch.cuda.nvtx.range_pop()

    #print(dgrad_osums.shape)
    #print('sums',dgrad_osums[0, 0, :10], dgrad_osums[0, 1, :10], np.linalg.norm((dgrad_osums[:, 0, :] -  dgrad_osums[:, 1, :]).cpu().numpy()))

    #print(smax.shape)
    #print('smax',smax[0, 0, 0, :10], smax[0, 1, 0, :10], np.linalg.norm((smax[:, 0, :, :] - smax[:, 1, :, :]).cpu().numpy()))

    dqkv21 = torch.zeros((b,s,3,h,d), device=device, dtype=dtype) # total, 3, h, d
    dqkv2 = unpack_buffer_seq(dqkv2, dqkv21, slens).view(b*s, 3, h, d) # b*s, 3, h, d
    dqkv2 = dqkv2.permute(0,2,1,3).view(b,s, h,3,d) # total, h, 3, d -> b,s,h,3,d

    dgrad = nullify_buffer_bylen(qkv.grad.float(), slens)
    dqkv2 = nullify_buffer_bylen(dqkv2, slens)

    assert(torch.allclose(dgrad, dqkv2.float(), atol=0.25))



torch.cuda.cudart().cudaProfilerStart()
run_test(2*384, 8, h=4, d=64)
#run_test(4*384, 4)
#run_test(512, 128)
#run_test(1024, 32)
torch.cuda.cudart().cudaProfilerStop()
