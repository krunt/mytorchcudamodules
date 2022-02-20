
import torch
from transformers import PyTorchBenchmarkArguments
from benchmark.mybenchmark import MyPyTorchBenchmark

args = PyTorchBenchmarkArguments(
    models=["mine"], batch_sizes=[8], sequence_lengths=[32,64,128])



def construct_func(model_name, batch_size, sequence_length):
    def func():
        totMemMb = batch_size * sequence_length * 2**20
        torch.zeros(totMemMb, dtype=torch.uint8, device=torch.device('cuda'))
    return func


benchmark = MyPyTorchBenchmark(construct_func, construct_func, args, configs=[{}])
results = benchmark.run()

print(results)
