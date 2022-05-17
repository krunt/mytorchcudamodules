import apex.contrib.fmha as fmha

import fmhalib as mha

import torch

import math

import numpy as np

import time


def py_mha(qkv, b, s, h, d):
    qkv = qkv.view(b, s, h, 3, d)
    q = qkv[:, :, :, 0, :].permute(0,2,1,3)
    k = qkv[:, :, :, 1, :].permute(0,2,1,3) # b s h d -> b h s d
    v = qkv[:, :, :, 2, :].permute(0,2,1,3)
    p = torch.matmul(q.float(), k.permute(0,1,3,2).float()) # b h s d * b h d s -> b h s s
    p_masked = p / math.sqrt(d)
    maxi=torch.max(p_masked, axis=-1)[0]
    print('py-p',p_masked[0,0,0,:16],maxi[0,0,:17])
    s = torch.softmax(p_masked, -1).to(qkv.dtype)
    print('py',s.shape,s[0,0,0,:17],torch.sum(torch.exp(p_masked-maxi.unsqueeze(-1)), axis=-1)[0,0,:17])
    #print('py',s.shape,s[0,0,0,256:266],torch.sum(torch.exp(p_masked-maxi.unsqueeze(-1)), axis=-1)[0,0,:17])
    ctx = torch.matmul(s, v)
    ctx = ctx.permute(0,2,1,3).contiguous()

    ctx.retain_grad()

    return ctx, s


def run_test(s, b):
    print(f'Test s={s} b={b}')

    zero_tensors=True

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    
    dtype = torch.float16
    device = torch.device('cuda')

    h = 1
    d = 128

    slens = [s] * b
    a = torch.tensor(np.array([0] + slens), dtype=torch.int32)
    amask = torch.ones(b,h,s,s, dtype=dtype, device=device)
    seqlens = torch.tensor(slens, dtype=torch.int32, device=device)
    cu_seqlens = torch.cumsum(a, 0).to(dtype=torch.int32, device=device)
    total = cu_seqlens[-1].item()

    qkv = torch.randn((b,s,h,3,d), device=device, dtype=dtype)

    qkv_vs = qkv.permute(0,1,3,2,4).contiguous().view(b*s, 3, h,d)

    cqkv_vs = qkv_vs.clone()

    qkv.requires_grad = True

    #print('qkv_vs',qkv_vs.shape, qkv_vs[0,0,0,:])

    st=time.time()
    iters = 1 # 2000
    for it in range(iters):
        #torch.cuda.nvtx.range_push("cuda-forward")
        ctx, cmaxs, csums = mha.fwd(qkv_vs, cu_seqlens, 0.0, s, True, False, zero_tensors, None)
        torch.cuda.synchronize(device)
        #print('cu0',smax.shape,smax[0,0,:16,0])
        #torch.cuda.nvtx.range_pop()
    #print('cuda',(time.time()-st)/iters)

    print(ctx.shape)

    #import pdb
    #pdb.set_trace()

    ctx = ctx.view(b,s,h,d)

    ctx_ref = None

    iters2 = 1
    st = time.time()
    for it in range(iters2):
        ctx_ref, ref_sftmax = py_mha(qkv, b,s,h,d)
    #print('py',(time.time()-st)/iters2)

    print(ctx.shape, ctx_ref.shape)

    print('ctx_ref',ctx_ref[0,0,0,:16])
    print('ctx',ctx[0,0,0,:16])
    #print('ctx_ref',ctx_ref[0,17,0,:16])
    #print('ctx',ctx[0,17,0,:16])
    print('ctx_ref',ctx_ref[0,0,0,-16:])
    print('ctx',ctx[0,0,0,-16:])
    #print('ctx_ref',ctx_ref[0,17,0,-16:])
    #print('ctx',ctx[0,17,0,-16:])

    #print('ctx_ref',ctx_ref[0,256,0,-16:])
    #print('ctx',ctx[0,256,0,-16:])

    #assert(torch.allclose(ctx_ref.float(), ctx.float(), atol=0.5))

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


    labels = torch.randn_like(ctx_ref)
    diff = ctx_ref - labels
    l = (diff * diff).sum() / b
    l.backward()

    dw = ctx_ref.grad.permute(0,2,1,3) 

    dw2 = dw.permute(0,2,1,3).clone().detach().contiguous()

    print('cmaxs',cmaxs.shape,cmaxs[0,0,:17])
    print('csums',csums.shape,csums[0,0,:17])
    #print('smax',smax.shape,smax[0,0,0,:17])

    #cmaxs[...] = 0
    #csums[...] = 1

    #print('qkv_vs',qkv_vs.shape, qkv_vs[0,0,0,:])

    iters = 1
    for it in range(iters):
        #torch.cuda.nvtx.range_push("cuda-backward")
        dqkv2, dgrad_osums, smax = mha.bwd(dw2, cqkv_vs, cu_seqlens, cmaxs, csums, 0.0, s, zero_tensors)
        torch.cuda.synchronize(device)
        print('cu','refs20', smax.shape,smax[0,0,0,:20], dgrad_osums.shape, dgrad_osums[0, 0, :20])
        #torch.cuda.nvtx.range_pop()

    #ref_sftmax
    print('dw2',dw2.shape, cqkv_vs.shape, ref_sftmax.shape)
    # dw2: b s h d torch.Size([1, 256, 1, 64])
    # cqkv_vs: s n h d torch.Size([256, 3, 1, 64])
    # ref_sftmax: b h s s torch.Size([1, 1, 256, 256])
    #ref_sftmax.permute(0, 1, 3, 2), 

    dgrado = torch.bmm(dw2[0, :, 0, :].unsqueeze(0), cqkv_vs[:, 2, 0, :].unsqueeze(0).permute(0, 2, 1)).unsqueeze(1)
    print('dgrado',dgrado.shape, dgrado[0,0,0,:20]);
    print('refsftmax',ref_sftmax.shape, ref_sftmax[0,0,0,:20])
    #print('dgrado',dgrado.shape, 'refsftmax',ref_sftmax.shape, ref_sftmax[0,0,32,256:256+16])
    #dodv = torch.bmm(dgrado[0,:,:,:].permute(0,1,2), dw2[0, :, 0, :].unsqueeze(0))
    dodv = torch.bmm(ref_sftmax[0, :, :, :].permute(0,2,1), dw2[0, :, 0, :].unsqueeze(0))
    #print('dv', dodv.shape, dodv[0, 0, :20])
    #print('dv2', dqkv2.shape, dqkv2[0, 2, 0, :20])

    #print('dv--', dodv.shape, dodv[0, 0, 64:64+16])
    #print('dv2--', dqkv2.shape, dqkv2[0, 2, 0, 64:64+16])

    print('dv-1', dodv.shape, dodv[0, 0, :20])
    print('dv2-1', dqkv2.shape, dqkv2[0, 2, 0, :20])

    print('dv-1', dodv.shape, dodv[0, 3, -20:])
    print('dv2-1', dqkv2.shape, dqkv2[3, 2, 0, -20:])

    if s > 256:
        print('dv-2', dodv.shape, dodv[0, 257, :20])
        print('dv2-2', dqkv2.shape, dqkv2[257, 2, 0, :20])

    #dgrado = dgrado * ref_sftmax
    print('dgrado',dgrado.shape, ref_sftmax.shape)
    dgrado_refsums = torch.sum(dgrado * ref_sftmax, axis=-1)
    print('dgrado_refsums', dgrado_refsums.shape, dgrado_refsums[0, 0, :20])

    #softmax_grads = torch._softmax_backward_data(dgrado, ref_sftmax, -1, ref_sftmax.dtype)

    softmax_grads = torch._softmax_backward_data(dgrado[0,:,:,:], ref_sftmax[0,:,:], -1, ref_sftmax.dtype).unsqueeze(0)
    softmax_grads /= torch.tensor(np.sqrt(d))
    #softmax_grads = (dgrado[0,:,:,:] * ref_sftmax)
    #softmax_grads = dgrado

    #print(dgrado_refsums.shape, ref_sftmax.shape)

    #softmax_grads = dgrado * ref_sftmax - dgrado_refsums * ref_sftmax
    #softmax_grads /= torch.tensor(np.sqrt(d))
    #softmax_grads = ref_sftmax
    print('softmax_grads',softmax_grads.shape, 'sgrads20', softmax_grads[0,0,0,:20], 'sgrads-20', softmax_grads[0,0,0,-20:])
    #print('softmax_grads',softmax_grads.shape, softmax_grads[0,0,32,256:256+16])

    #dkdv = torch.bmm(softmax_grads[0, :, :, :].permute(0,2,1), cqkv_vs[:, 0, 0, :].unsqueeze(0))
    dkdv = torch.bmm(softmax_grads[0, :, :, :].permute(0,2,1), cqkv_vs[:, 0, 0, :].unsqueeze(0))

    print('dk', dkdv.shape, dkdv[0, 0, :20])
    print('dk2', dqkv2.shape, dqkv2[0, 1, 0, :20])

    print('dk1', dkdv.shape, dkdv[0, 1, -20:])
    print('dk21', dqkv2.shape, dqkv2[1, 1, 0, -20:])

    #print('dk2', dkdv.shape, dkdv[0, 2, :20])
    #print('dk22', dqkv2.shape, dqkv2[2, 1, 0, :20])

    print('dk-1', dkdv.shape, dkdv[0, 17, :20])
    print('dk2-1', dqkv2.shape, dqkv2[17, 1, 0, :20])

    print('dk-10', dkdv.shape, dkdv[0, 17, -20:])
    print('dk2-10', dqkv2.shape, dqkv2[17, 1, 0, -20:])

    if s > 256:
        print('dk-2', dkdv.shape, dkdv[0, 257, :20])
        print('dk2-2', dqkv2.shape, dqkv2[257, 1, 0, :20])
        print('dk-2', dkdv.shape, dkdv[0, 257, -20:])
        print('dk2-2', dqkv2.shape, dqkv2[257, 1, 0, -20:])

    dqdv = torch.bmm(softmax_grads[0, :, :, :], cqkv_vs[:, 1, 0, :].unsqueeze(0))

    print('dq', dqdv.shape, dqdv[0, 0, :20])
    print('dq2', dqkv2.shape, dqkv2[0, 0, 0, :20])

    print('dq1', dqdv.shape, dqdv[0, 1, -20:])
    print('dq21', dqkv2.shape, dqkv2[1, 0, 0, -20:])

    print('dq2', dqdv.shape, dqdv[0, 17, :20])
    print('dq22', dqkv2.shape, dqkv2[17, 0, 0, :20])
#
#    print('dq-1', dqdv.shape, dqdv[0, 127, :20])
#    print('dq2-1', dqkv2.shape, dqkv2[127, 0, 0, :20])
#
#    print('dq0-2', dqdv.shape, dqdv[0, 127, -20:])
#    print('dq02-2', dqkv2.shape, dqkv2[127, 0, 0, -20:])
#
#    print('dq-12', dqdv.shape, dqdv[0, 128, :20])
#    print('dq2-12', dqkv2.shape, dqkv2[128, 0, 0, :20])
#
#    print('dq-13', dqdv.shape, dqdv[0, 129, :20])
#    print('dq2-13', dqkv2.shape, dqkv2[129, 0, 0, :20])

    if s > 256:
        print('dq-2', dqdv.shape, dqdv[0, 257, :20])
        print('dq2-2', dqkv2.shape, dqkv2[257, 0, 0, :20])
        print('dq-2', dqdv.shape, dqdv[0, 257, -20:])
        print('dq2-2', dqkv2.shape, dqkv2[257, 0, 0, -20:])

    #import pdb
    #pdb.set_trace()
    
#    dqkv2 = dqkv2.permute(0,2,1,3).view(b,s, h,3,d)
#
#    assert(torch.allclose(qkv.grad.float(), dqkv2.float(), atol=1e-1))



#torch.cuda.cudart().cudaProfilerStart()
run_test(256, 1)
#torch.cuda.cudart().cudaProfilerStop()
