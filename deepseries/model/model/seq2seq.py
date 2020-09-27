# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/9/18 10:18
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from deepseries.nn import Attention


class RNNEncoder(nn.Module):

    def __init__(self, input_size, rnn_type, hidden_size, bidirectional, num_layers, dropout):
        super().__init__()
        self.input_size = input_size
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_direction = 2 if self.bidirectional else 1
        self.input_dropout = nn.Dropout(dropout)
        self.rnn = getattr(nn, rnn_type)(input_size=input_size, bidirectional=bidirectional, batch_first=True,
                                         num_layers=num_layers, hidden_size=hidden_size, dropout=dropout)

    def forward(self, input: torch.Tensor):
        batch_size = input.shape[0]
        output, hidden = self.rnn(input)

        def _reshape_hidden(hn):
            hn = hn.view(self.num_layers, 2, batch_size, self.hidden_size). \
                permute(0, 2, 1, 3).reshape(self.num_layers, batch_size, self.num_direction * self.hidden_size)
            return hn

        if self.bidirectional and self.rnn_type != "LSTM":
            hidden = _reshape_hidden(hidden)
        elif self.bidirectional and self.rnn_type == "LSTM":
            h, c = _reshape_hidden(hidden[0]), _reshape_hidden(hidden[1])
            hidden = (h, c)

        return output, hidden


class RNNDecoder(nn.Module):

    def __init__(self, input_size, output_size, rnn_type, hidden_size, num_layers, dropout, attn_head, attn_size):
        super().__init__()
        self.input_size = input_size
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size

        self.input_dropout = nn.Dropout(dropout)
        self.rnn = getattr(nn, rnn_type)(input_size=input_size, batch_first=True,
                                         num_layers=num_layers, hidden_size=hidden_size, dropout=dropout)
        self.attention = Attention(attn_head, attn_size, hidden_size, hidden_size, hidden_size, dropout)
        self.regression = nn.Linear(input_size + attn_size, output_size)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor, encoder_output):
        # single step
        # step input -> (batch, 1, N); previous dec hidden (layer, batch, hidden_size)
        dec_rnn_output, dec_rnn_hidden = self.rnn(input, hidden)
        # attention
        attn_applied, attn_weights = self.attention(dec_rnn_output, encoder_output, encoder_output)
        # predict
        concat = F.tanh(torch.cat([input, attn_applied], dim=2))
        output = self.regression(concat)
        return output, hidden, attn_weights


class Seq2Seq(nn.Module):

    def __init__(self, encoder_inputs, decoder_inputs, target_size, decode_length,
                 rnn_type, hidden_size, num_layers, bidirectional, dropout,
                 attn_heads, attn_size, loss_fn=nn.MSELoss(), share_embeddings=None):
        super().__init__()
        self.encoder_inputs = Inputs(encoder_inputs)
        self.decoder_inputs = Inputs(decoder_inputs)
        self.target_size = target_size
        self.decode_length = decode_length
        self.loss_fn = loss_fn

        if share_embeddings is not None:
            pass

        self.encoder = RNNEncoder(self.encoder_inputs.output_size, rnn_type, hidden_size,
                                  bidirectional, num_layers, dropout)
        num_directional = 2 if bidirectional else 1
        self.decoder = RNNDecoder(self.decoder_inputs.output_size + self.target_size, self.target_size, rnn_type,
                                  hidden_size * num_directional, num_layers, dropout, attn_heads, attn_size)

    def train_batch(self, feed_dict, target, last_target):
        enc_inputs = self.encoder_inputs(feed_dict)
        enc_outputs, enc_hidden = self.encoder(enc_inputs)
        dec_inputs = self.decoder_inputs(feed_dict)
        dec_inputs = torch.cat([dec_inputs, last_target], dim=2)
        pred, hidden, attn_weights = self.decoder(dec_inputs, enc_hidden, enc_outputs)
        loss = self.loss_fn(pred, target)
        return loss


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

    def __init__(self, inputs_config):
        super().__init__()
        self.numerical = inputs_config.get("numerical")
        self.categorical = inputs_config.get("categorical")
        self.output_size = 0
        if self.categorical is not None:
            self.categorical_inputs = MultiEmbeddings(*self.categorical)
            self.output_size += sum([i[2] for i in self.categorical])

        if self.numerical is not None:
            self.numerical_inputs = nn.ModuleDict({name: Empty(size) for (name, size) in self.numerical})
            self.output_size += sum([i[1] for i in self.numerical])

    def forward(self, feed_dict):
        # batch, seq, N
        outputs = []
        if self.categorical is not None:
            outputs.append(self.categorical_inputs(feed_dict))
        if self.numerical is not None:
            for (name, _) in self.numerical:
                outputs.append(self.numerical_inputs[name](feed_dict[name]))
        return torch.cat(outputs, dim=2)


if __name__ == "__main__":
    batch_size = 4
    enc_len = 14
    dec_len = 7

    encode_inputs = {
        "numerical": [("enc_flow", 4)],
        "categorical": [("enc_weekday", 8, 2)],
    }

    decode_inputs = {
        "categorical": [("dec_weekday", 8, 2)],
    }

    batch_feed = {
        "enc_flow": torch.rand(batch_size, enc_len, 4),
        "enc_weekday": torch.randint(0, 3, (batch_size, enc_len)),
        "dec_weekday": torch.randint(0, 3, (batch_size, dec_len)),
        "target": torch.rand(batch_size, dec_len, 1),
        "last_target": torch.rand(batch_size, dec_len, 1)
    }

    model = Seq2Seq(encode_inputs, decode_inputs, 1, dec_len, "GRU", 24, 1, True, 0.1, 3, 12, False)
    optmizer = torch.optim.Adam(model.parameters(), 0.01)
    loss = model.train_batch(batch_feed)
    optmizer.zero_grad()
    loss.backward()
    optmizer.step()
    loss.item()