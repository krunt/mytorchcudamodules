import apex.contrib.fmha as fmha

import fmhalib as mha

import torch

import math

import numpy as np

import time


def py_mha(qkv, amask, b, s, h, d):
    qkv = qkv.view(b, s, h, 3, d)
    q = qkv[:, :, :, 0, :].permute(0,2,1,3)
    k = qkv[:, :, :, 1, :].permute(0,2,1,3)
    v = qkv[:, :, :, 2, :].permute(0,2,1,3)
    p = torch.matmul(q.float(), k.permute(0,1,3,2).float())
    p_masked = p / math.sqrt(d) + (1.0 - amask) * -10000.0
    s = torch.softmax(p_masked, -1).to(qkv.dtype)
    ctx = torch.matmul(s, v)
    ctx = ctx.permute(0,2,1,3).contiguous()

    ctx.retain_grad()

    return ctx


def run_test(s, b, h = 16, d = 128):
    print(f'Test s={s} b={b}')

    zero_tensors=True

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    
    dtype = torch.float16
    device = torch.device('cuda')

    slens = [s] * b
    a = torch.tensor(np.array([0] + slens), dtype=torch.int32)
    amask = torch.ones(b,h,s,s, dtype=dtype, device=device)
    seqlens = torch.tensor(slens, dtype=torch.int32, device=device)
    cu_seqlens = torch.cumsum(a, 0).to(dtype=torch.int32, device=device)
    total = cu_seqlens[-1].item()

    qkv = torch.randn((b,s,h,3,d), device=device, dtype=dtype)

    #with torch.no_grad():
        #qkv[:, :, 1, :, :] = qkv[:, :, 0, :, :]

    qkv_vs = qkv.permute(0,1,3,2,4).contiguous().view(b*s, 3, h,d)
    #print('qkv_vs', qkv_vs.shape)

    qkv.requires_grad = True

    st=time.time()
    iters = 1 # 2000
    for it in range(iters):
        torch.cuda.nvtx.range_push("cuda-forward")
        ctx, cmaxs, csums  = mha.fwd(qkv_vs, cu_seqlens, 0.0, s, True, False, zero_tensors, None)
        torch.cuda.synchronize(device)
        torch.cuda.nvtx.range_pop()
    print('cuda',(time.time()-st)/iters)

    ctx = ctx.view(b,s,h,d)

    ctx_ref = None

    iters2 = 1
    st = time.time()
    for it in range(iters2):
        torch.cuda.nvtx.range_push("py-forward")
        ctx_ref = py_mha(qkv, amask, b,s,h,d)
        torch.cuda.nvtx.range_pop()
    print('py',(time.time()-st)/iters2)

    #assert(torch.allclose(ctx_ref.float(), ctx.float(), atol=0.4)) # 1e-2))
    assert(torch.allclose(ctx_ref.float(), ctx.float(), atol=0.4)) # 1e-2))

    labels = torch.randn_like(ctx_ref)
    #labels[:, :, 1, :] = labels[:, :, 0, :]
    diff = ctx_ref - labels
    l = (diff * diff).sum() / b
    l.backward()

    dw = ctx_ref.grad.permute(0,2,1,3) 

    dw2 = dw.permute(0,2,1,3).clone().detach().contiguous()

    #print('dw2',dw2.shape)
    #print(dw2[:, :, 0, :20], dw2[:, :, 1, :20])

    iters = 1
    for it in range(iters):
        torch.cuda.nvtx.range_push("cuda-backward")
        dqkv2, dgrad_osums, smax = mha.bwd(dw2, qkv_vs, cu_seqlens, cmaxs, csums, 0.0, s, zero_tensors)
        torch.cuda.synchronize(device)
        torch.cuda.nvtx.range_pop()

    #print(dgrad_osums.shape)
    #print('sums',dgrad_osums[0, 0, :10], dgrad_osums[0, 1, :10], np.linalg.norm((dgrad_osums[:, 0, :] -  dgrad_osums[:, 1, :]).cpu().numpy()))

    #print(smax.shape)
    #print('smax',smax[0, 0, 0, :10], smax[0, 1, 0, :10], np.linalg.norm((smax[:, 0, :, :] - smax[:, 1, :, :]).cpu().numpy()))

    dqkv2 = dqkv2.permute(0,2,1,3).view(b,s, h,3,d)

    #print('dqkv2',np.linalg.norm((dqkv2[:, :, 0, 2, :] - dqkv2[:, :, 1, 2, :]).cpu().numpy()))
    #print('dqkv1',np.linalg.norm((dqkv2[:, :, 0, 1, :] - dqkv2[:, :, 1, 1, :]).cpu().numpy()))
    #print('dqkv0',np.linalg.norm((dqkv2[:, :, 0, 0, :] - dqkv2[:, :, 1, 0, :]).cpu().numpy()))

    #print('ddd0',dqkv2[0, 0, 0, 0, :128], dqkv2[0, 0, 1, 0, :128])
    #print('ddd1',dqkv2[0, 1, 0, 0, :128], dqkv2[0, 1, 1, 0, :128])
    #print('ddd129',dqkv2[0, 129, 0, 0, :128], dqkv2[0, 129, 1, 0, :128])
    

    #print(qkv.grad.shape, dqkv2.shape)
    #assert(torch.allclose(qkv.grad.float()[:, :, :, 2, :], dqkv2.float()[:, :, :, 2, :], atol=0.01))
    #assert(torch.allclose(qkv.grad.float()[:, :, :, 1, :], dqkv2.float()[:, :, :, 1, :], atol=0.01))
    #print('01', dqkv2.float()[:, :, 0, 0, :10], dqkv2.float()[:, :, 1, 0, :10])
    #print('23', qkv.grad.float()[:, :, 0, 0, :10], qkv.grad.float()[:, :, 1, 0, :10])

    #print(qkv.grad.float()[:, 0, 0, 0, :20], dqkv2.float()[:, 0, 1, 0, :20])
    #print(qkv.grad.float()[:, 1, 0, 0, :20], dqkv2.float()[:, 1, 1, 0, :20])

    #assert(torch.allclose(qkv.grad.float()[:, :, 0, 0, :64], dqkv2.float()[:, :, 0, 0, :64], atol=0.25))
    #assert(torch.allclose(qkv.grad.float()[:, :, 1, 0, :64], dqkv2.float()[:, :, 1, 0, :64], atol=0.25))

    #assert(torch.allclose(qkv.grad.float()[:, :, 0, 0, :], dqkv2.float()[:, :, 0, 0, :], atol=0.25))
    #assert(torch.allclose(qkv.grad.float()[:, :, 1, 0, :], dqkv2.float()[:, :, 1, 0, :], atol=0.25))
    #assert(torch.allclose(qkv.grad.float()[:, :, 0, 0, 64:], dqkv2.float()[:, :, 0, 0, 64:], atol=0.25))
    #assert(torch.allclose(qkv.grad.float()[:, :, 0, 0, :64], dqkv2.float()[:, :, 0, 0, :64], atol=0.25))
    #assert(torch.allclose(qkv.grad.float(), dqkv2.float(), atol=0.25))
    assert(torch.allclose(qkv.grad.float(), dqkv2.float(), atol=0.25))



torch.cuda.cudart().cudaProfilerStart()
run_test(512, 4, h=4, d=128)
#run_test(4*384, 4)
#run_test(512, 128)
#run_test(1024, 32)
torch.cuda.cudart().cudaProfilerStop()
