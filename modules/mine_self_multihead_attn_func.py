
import torch
from torch import nn
import torch.nn.functional as F

from modules import scaled_masked_softmax

class SelfAttnFunc(nn.Module):
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
        dropout_prob_t = torch.tensor([dropout_prob])
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
        softmax_results = scaled_masked_softmax.scaled_masked_softmax(matmul1_results, attn_mask, scale_t[0])
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

        return outputs

self_attn_func = SelfAttnFunc()
