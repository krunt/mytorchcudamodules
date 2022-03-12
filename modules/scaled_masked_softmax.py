import torch
import torch.nn as nn
import importlib

global apex_scaled_masked_softmax_cuda
apex_scaled_masked_softmax_cuda = None

class ScaledMaskedSoftmax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, mask, scale):
        scale_t = torch.tensor([scale])

        softmax_results = apex_scaled_masked_softmax_cuda.forward(inputs, mask, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        softmax_results, scale_t = ctx.saved_tensors

        input_grads = apex_scaled_masked_softmax_cuda.backward(output_grads, softmax_results, scale_t[0])
        return input_grads, None, None


def scaled_masked_softmax(inputs, mask, scale):
    # input is 4D tensor (b, np, sq, sk)
    return ScaledMaskedSoftmax.apply(inputs, mask, scale)

def initialize():
    global apex_scaled_masked_softmax_cuda
    apex_scaled_masked_softmax_cuda = importlib.import_module(
          "apex_scaled_masked_softmax_cuda")
