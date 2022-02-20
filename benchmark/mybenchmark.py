from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments, BertConfig
from typing import Callable

class MyPyTorchBenchmark(PyTorchBenchmark):

    def __init__(self, construct_train_func, construct_inference_func,
            args = None, configs = None):
        super().__init__(args, configs)
        self._construct_train_func = construct_train_func
        self._construct_inference_func = construct_inference_func

    def _prepare_train_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:
        return self._construct_train_func(model_name, batch_size, sequence_length)

    def _prepare_inference_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:
        return self._construct_inference_func(model_name, batch_size, sequence_length)

