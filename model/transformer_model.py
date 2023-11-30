
import torch
import torch.nn as nn
import math
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, numClass, d_model, nhead, d_hid, nlayers, resnet, dropout):
        super().__init__()
        self.cnn = resnet
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, numClass)
        self.softmax = nn.LogSoftmax(dim = -1)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask: Tensor = None) -> Tensor:
        batch_size = src.shape[0]
        newX = torch.zeros(src.shape[0], src.shape[1], 512)  # Assuming DEVICE is defined elsewhere
        for i in range(src.size(1)):
            l = self.cnn(torch.unsqueeze(src[:,i,:], 1))
            newX[:,i,:] = l

        src = newX
        src = self.pos_encoder(newX)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        output = self.softmax(output)
        return output
