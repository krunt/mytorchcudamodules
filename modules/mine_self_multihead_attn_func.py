
import torch
from torch import nn
import torch.nn.functional as F

from modules import scaled_masked_softmax

class SelfAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        use_time_mask,
        is_training,
        heads,
        scale,
        inputs,
        input_weights,
        output_weights,
        input_biases,
        output_biases,
        mask,
        is_additive_mask,
        dropout_prob,
    ):
        use_biases_t = torch.tensor([input_biases is not None])
        heads_t = torch.tensor([heads])
        scale_t = torch.tensor([scale])
        null_tensor = torch.tensor([])
        head_dim = inputs.size(2) // heads

        if use_biases_t[0]:
            input_lin_results = torch.addmm(
                input_biases,
                inputs.reshape(inputs.size(0) * inputs.size(1), inputs.size(2)), 
                input_weights.transpose(0, 1),
                beta=1.0,
                alpha=1.0
            )
        else:
            input_lin_results = torch.mm(
                inputs.reshape(inputs.size(0) * inputs.size(1), inputs.size(2)), input_weights.transpose(0, 1)
            )
        input_lin_results = input_lin_results.view(inputs.size(0), inputs.size(1), input_weights.size(0))

        #print('a', input_lin_results)

        # [seqs, seql, embed]
        input_lin_results = input_lin_results.view(inputs.size(0), inputs.size(1), 3, inputs.size(2))
        queries = input_lin_results[:, :, 0, :].contiguous().view(inputs.size(0), inputs.size(1), heads, head_dim) \
                  .transpose(1,2).contiguous().view(inputs.size(0) * heads, inputs.size(1), head_dim)
        keys = input_lin_results[:, :, 1, :].contiguous().view(inputs.size(0), inputs.size(1), heads, head_dim) \
                  .transpose(1,2).contiguous().view(inputs.size(0) * heads, inputs.size(1), head_dim)
        # output:               [seql_q, seqs*heads, head_dim]
        values = input_lin_results[:, :, 2, :].contiguous().view(inputs.size(0), inputs.size(1), heads, head_dim) \
                  .transpose(1,2).contiguous().view(inputs.size(0) * heads, inputs.size(1), head_dim)

        # [seqs, 
        #print('aqkv', queries, keys, values)

        matmul1_results = torch.empty(
            (queries.size(0), queries.size(1), keys.size(1)), dtype=queries.dtype, device=inputs.device
        )

        # output:           [seqs*heads, seql_q, seql_k]
        matmul1_results = torch.baddbmm(
            matmul1_results,
            queries,
            keys.transpose(1, 2),
            beta=0.0,
            alpha=1,
        )
        
        # output:           [seqs*heads, seql_q, seql_k]
        #softmax_results = F.softmax(matmul1_results, dim=-1)
        attn_mask = torch.ones((inputs.size(0), 1, inputs.size(1), inputs.size(1)), 
                dtype=torch.uint8, device=inputs.device)

        matmul1_results = matmul1_results.view(inputs.size(0), heads, inputs.size(1), inputs.size(1))

        softmax_results = scaled_masked_softmax.apex_scaled_masked_softmax_cuda.forward(matmul1_results, attn_mask, scale_t[0])

        softmax_results = softmax_results.view(inputs.size(0) * heads, inputs.size(1), inputs.size(1))

        matmul2_results = torch.bmm(softmax_results, values)

        matmul2_results = matmul2_results.reshape(inputs.size(0), heads, inputs.size(1), head_dim) \
                          .transpose(1, 2).reshape(inputs.size(0), inputs.size(1), inputs.size(2))

        # output:               [seqs*heads, seql_q, head_dim]
        if use_biases_t[0]:
            outputs = torch.addmm(
                output_biases,
                matmul2_results.reshape(inputs.size(0) * inputs.size(1), inputs.size(2)), 
                output_weights.transpose(0, 1),
                beta=1.0,
                alpha=1.0
            )
        else:
            outputs = torch.mm(
                matmul2_results.reshape(inputs.size(0) * inputs.size(1), inputs.size(2)), 
                output_weights.transpose(0, 1)
            )
        outputs = outputs.view(inputs.size(0), inputs.size(1), inputs.size(2))

        ctx.save_for_backward(
            use_biases_t,
            heads_t,
            scale_t,
            matmul2_results,
            softmax_results,
            input_lin_results,
            inputs,
            input_weights,
            output_weights,
        )

        return outputs.detach()

    @staticmethod
    def backward(ctx, output_grads):
        (
            use_biases_t,
            heads_t,
            scale_t,
            matmul2_results,
            softmax_results,
            input_lin_results,
            inputs,
            input_weights,
            output_weights
        ) = ctx.saved_tensors

        heads = heads_t[0].item()
        head_dim = inputs.size(2) // heads

        # [seqs, seql, emb_dim]
        # inputs

        input_lin_results = input_lin_results.view(inputs.size(0), inputs.size(1), 3, inputs.size(2))
        queries = input_lin_results[:, :, 0, :].contiguous().view(inputs.size(0), inputs.size(1), heads, head_dim) \
                  .transpose(1,2).contiguous().view(inputs.size(0) * heads, inputs.size(1), head_dim)
        keys = input_lin_results[:, :, 1, :].contiguous().view(inputs.size(0), inputs.size(1), heads, head_dim) \
                  .transpose(1,2).contiguous().view(inputs.size(0) * heads, inputs.size(1), head_dim)
        # output:               [seql_q, seqs*heads, head_dim]
        values = input_lin_results[:, :, 2, :].contiguous().view(inputs.size(0), inputs.size(1), heads, head_dim) \
                  .transpose(1,2).contiguous().view(inputs.size(0) * heads, inputs.size(1), head_dim)

        # [seqs * heads, seql, emb_dim]
        input_lin_results_grads = torch.empty((inputs.size(0) * heads, inputs.size(1), 3, head_dim),
                                    dtype=queries.dtype, device=inputs.device)
        queries_grads = input_lin_results_grads[:, :, 0, :]
        keys_grads = input_lin_results_grads[:, :, 1, :]
        values_grads = input_lin_results_grads[:, :, 2, :]

        # output_grads [seqs, seql, emb_dim]

        output_lin_grads = torch.mm(
            output_grads.reshape(output_grads.size(0) * output_grads.size(1), output_grads.size(2)), output_weights
        )
        output_lin_grads = output_lin_grads.view(output_grads.size(0), output_grads.size(1), output_weights.size(1))

        output_weight_grads = torch.mm(
            output_grads.reshape(output_grads.size(0) * output_grads.size(1), output_grads.size(2)).transpose(0, 1),
            matmul2_results.view(matmul2_results.size(0) * matmul2_results.size(1), matmul2_results.size(2)),
        )

        output_lin_grads = output_lin_grads.view(inputs.size(0), inputs.size(1), heads, head_dim) \
                  .transpose(1,2).contiguous().view(inputs.size(0) * heads, inputs.size(1), head_dim)

        if use_biases_t[0]:
            output_bias_grads = torch.sum(
                output_grads.view(output_grads.size(0) * output_grads.size(1), output_grads.size(2)), 0
            )
        else:
            output_bias_grads = None


        # output_lin_grads  [ seqs * heads, seql, head_dim ]
        # values [ seqs * heads, seql, head_dim ]
        # output: [ seqs * heads, seql, seql ]
        matmul2_dgrad1 = torch.bmm(output_lin_grads, values.transpose(1, 2))

        # softmax_results [ seqs * heads, seql, seql ]
        # output_lin_grads  [ seqs * heads, seql, head_dim ]
        # output: [ seqs * heads, seql, head_dim ]
        values_grads = torch.bmm(softmax_results.transpose(1, 2), output_lin_grads)

        # output: [ seqs * heads, seql, seql ]
        softmax_grads = scaled_masked_softmax.apex_scaled_masked_softmax_cuda.backward(
                matmul2_dgrad1.view(inputs.size(0), heads, inputs.size(1), inputs.size(1)), 
                softmax_results.view(inputs.size(0), heads, inputs.size(1), inputs.size(1)),
                scale_t[0])

        softmax_grads = softmax_grads.view(inputs.size(0) * heads, inputs.size(1), inputs.size(1))

        queries_grads = torch.baddbmm(
            queries_grads,
            softmax_grads,
            keys,
            beta=0.0,
            alpha=scale_t[0],
        )

        keys_grads = torch.baddbmm(
            keys_grads,
            softmax_grads.transpose(1, 2),
            queries,
            beta=0.0,
            alpha=scale_t[0],
        )

        # [ seqs * heads, seql, head_dim ]
        input_lin_results_grads = input_lin_results_grads.view(inputs.size(0), heads, inputs.size(1), 3, head_dim) \
                    .transpose(1, 2).contiguous().view(inputs.size(0) * inputs.size(1), heads * 3 * head_dim)
        input_grads = torch.mm(input_lin_results_grads, input_weights)
        input_grads = input_grads.view(inputs.size(0), inputs.size(1), inputs.size(2))

        input_weight_grads = torch.mm(
            input_lin_results_grads.transpose(0, 1), inputs.reshape(inputs.size(0) * inputs.size(1), inputs.size(2))
        )

        if use_biases_t[0]:
            input_bias_grads = torch.sum(input_lin_results_grads, 0)
        else:
            input_bias_grads = None

        return (
            None, # use_time_mask,
            None, # is_training,
            None, # heads,
            None, # scale,
            input_grads, # inputs,
            input_weight_grads, # input_weights,
            output_weight_grads, # output_weights,
            input_bias_grads, # input_biases,
            output_bias_grads, # output_biases,
            None, # mask,
            None, # is_additive_mask,
            None, # dropout_prob,
        )


self_attn_func = SelfAttnFunc.apply
