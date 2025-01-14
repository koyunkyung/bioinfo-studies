import torch
import torch.nn as nn
import torch.optim as optim

class Transformer(nn.Module):
    
    def __init__(self, num_tokens, dim_model, num_heads, num_encoder_layers, num_decoder_layers, dropout_p, ):
        super().__init__()

        self.model_type = "Transformer"
        self.dim_model = dim_model

        # layers

        # [encoder, embedding layers 추가 필요]

        self.transformer = nn.Transformer(
            d_model = dim_model,
            nhead = num_heads,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            dropout = dropout_p,
        )
        self.out = nn.Linear(dim_model, num_tokens)

    def forward(self):
        pass