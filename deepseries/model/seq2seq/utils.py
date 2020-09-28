# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/9/27 15:04
"""
import torch
import torch.nn as nn


class MultiEmbeddings(nn.Module):

    def __init__(self, *variable_params):
        # example: *[(name, num_embeddings, embedding_dim), ... ]
        super().__init__()
        self.params = variable_params
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(s, e) for (name, s, e) in variable_params
        })

    def forward(self, input):
        return torch.cat([self.embeddings[name](input[name]) for (name, _, _) in self.params], dim=2)


class Empty(nn.Module):

    def __init__(self, size):
        self.size = size
        super().__init__()

    def forward(self, x):
        return x

    def extra_repr(self):
        return f"{self.size}"


class Inputs(nn.Module):

    def __init__(self, inputs_config=None):
        super().__init__()
        self.inputs_config = inputs_config
        if inputs_config is not None:
            self.numerical = inputs_config.get("numerical")
            self.categorical = inputs_config.get("categorical")
            self.output_size = 0
            if self.categorical is not None:
                self.categorical_inputs = MultiEmbeddings(*self.categorical)
                self.output_size += sum([i[2] for i in self.categorical])

            if self.numerical is not None:
                self.numerical_inputs = nn.ModuleDict({name: Empty(size) for (name, size) in self.numerical})
                self.output_size += sum([i[1] for i in self.numerical])
        else:
            self.output_size = 0

    def forward(self, feed_dict):
        # batch, seq, N
        if self.inputs_config is not None:
            outputs = []
            if self.categorical is not None:
                outputs.append(self.categorical_inputs(feed_dict))
            if self.numerical is not None:
                for (name, _) in self.numerical:
                    outputs.append(self.numerical_inputs[name](feed_dict[name]))
            return torch.cat(outputs, dim=2)
        else:
            return None
