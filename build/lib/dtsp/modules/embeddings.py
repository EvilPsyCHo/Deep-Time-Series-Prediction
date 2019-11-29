# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2019/11/27 14:29
"""
from torch import nn
import torch


class Embeddings(nn.Module):

    def __init__(self, var):
        """

        Parameters
        ----------
        vars (List of Tuple): [(var_n_unique, var_embed_dim), ...]
        """
        super(Embeddings, self).__init__()
        self.var = var
        self.embeds = nn.ModuleList()
        for n, embed_dim in self.var:
            self.embeds.append(nn.Embedding(n, embed_dim))

    def forward(self, x):
        # x shape: B x S x V
        out = []
        v = x.shape[-1]
        for i in range(v):
            out.append(self.embeds[i](x[:, :, i]))
        out = torch.cat(out, dim=2)
        return out


if __name__ == "__main__":
    test_x = torch.randint(0, 10, (4, 10, 2))
    model = Embeddings([(10, 2), (10, 3)])
    y = model(test_x)
    print(test_x.shape, y.shape)
