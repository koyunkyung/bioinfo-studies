import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf

class Transformer(nn.Module):
    
    def __init__(self, num_tokens, dim_model, num_heads, num_encoder_layers, num_decoder_layers, dropout_p, ):
        super().__init__()

        self.model_type = "Transformer"
        self.dim_model = dim_model

        # layers

        # [encoder, embedding layers 추가 필요]
        self.
        self.embedding = 

        self.transformer = nn.Transformer(
            d_model = dim_model,
            nhead = num_heads,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            dropout = dropout_p,
        )
        self.out = nn.Linear(dim_model, num_tokens)

    def forward():
        pass

# 멀티 헤드 어텐션과 피드 포워드 신경망으로 이루어진 인코더 구현하고 층 쌓기
class TransformerEncoder:
    def encoder_layer(dff, d_model, num_heads, dropout, name="encoder_layer"):
        inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
        padding_mask = tf.keras.Input(shape=(None, d_model), name="inputs")

# 