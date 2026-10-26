"""Removable decoder hooks without retaining every layer's prompt activations."""


class LayerCapture:
    def __init__(self, decoder, layer_numbers):
        if (len(layer_numbers) != 3 or len(set(layer_numbers)) != 3 or
                any(type(n) is not int or not 1 <= n <= len(decoder.layers)
                    for n in layer_numbers)):
            raise ValueError('three distinct decoder block numbers required')
        self.decoder, self.layers = decoder, tuple(layer_numbers)
        self.handles = []
        self.values, self.final = {}, None
        self.prefill = False

    def start(self, prefill):
        if type(prefill) is not bool:
            raise ValueError('prefill must be boolean')
        self.prefill, self.values, self.final = prefill, {}, None

    def _block(self, number):
        def capture(module, inputs, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] < 1:
                raise ValueError('expected batch-one decoder activations')
            # clone prevents a view from retaining the entire prefill allocation.
            self.values[number] = value[0, -1].detach().clone()
        return capture

    def _norm(self, module, inputs, output):
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] < 1:
            raise ValueError('expected batch-one normalized decoder activations')
        self.final = output.detach() if self.prefill else output[:, -1:].detach().clone()

    def result(self):
        if set(self.values) != set(self.layers) or self.final is None:
            raise ValueError('incomplete current-forward hook coverage')
        import torch
        return torch.stack([self.values[n] for n in self.layers]), self.final

    def __enter__(self):
        if self.handles:
            raise RuntimeError('capture context already entered')
        try:
            for number in self.layers:
                self.handles.append(self.decoder.layers[number - 1].register_forward_hook(self._block(number)))
            self.handles.append(self.decoder.norm.register_forward_hook(self._norm))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles, self.values, self.final = [], {}, None
