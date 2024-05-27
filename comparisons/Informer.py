import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding,DataEmbedding_wo_pos,DataEmbedding_wo_temp,DataEmbedding_wo_pos_temp

class Config:
    def __init__(self, config_dict):
        self._config_dict = config_dict

    def __getattr__(self, item):
        return self._config_dict[item]

class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, opts):
        super(Model, self).__init__()
        configs = Config(opts)
        self.seq_len = configs.seq_day * configs.cycle
        self.label_len = configs.pred_day * configs.cycle
        self.pred_len = configs.pred_day * configs.cycle

        self.target_dim = (configs.node_feature_dim * configs.num_node) + ( configs.num_node * configs.num_node )
        self.enc_in = self.target_dim
        self.embed_type = 0
        self.d_model = 2
        self.d_ff = 4
        self.embed = 'timeF'
        self.freq = 't'
        self.dropout = 0.5
        
        self.factor = 1
        self.n_heads = 2
        self.moving_avg = 100
        self.activation = 'gelu'
        self.e_layers = 2
        self.c_out = self.target_dim
        self.dec_in = self.target_dim
        self.d_layers = 1
        self.distil = True
        
        self.output_attention = True

        # Embedding
        if self.embed_type == 0:
            self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, self.embed, self.freq,
                                            self.dropout)
            self.dec_embedding = DataEmbedding(self.dec_in, self.d_model, self.embed, self.freq,
                                           self.dropout)
        elif self.embed_type == 1:
            self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, self.embed, self.freq,
                                                    self.dropout)
            self.dec_embedding = DataEmbedding(self.dec_in, self.d_model, self.embed, self.freq,
                                                    self.dropout)
        elif self.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(self.enc_in, self.d_model, self.embed, self.freq,
                                                    self.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(self.dec_in, self.d_model, self.embed, self.freq,
                                                    self.dropout)
        elif self.embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(self.enc_in, self.d_model, self.embed, self.freq,
                                                    self.dropout)
            self.dec_embedding = DataEmbedding_wo_temp(self.dec_in, self.d_model, self.embed, self.freq,
                                                    self.dropout)
        elif self.embed_type == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(self.enc_in, self.d_model, self.embed, self.freq,
                                                    self.dropout)
            self.dec_embedding = DataEmbedding_wo_pos_temp(self.dec_in, self.d_model, self.embed, self.freq,
                                                    self.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=self.output_attention),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            [
                ConvLayer(
                    self.d_model
                ) for l in range(self.e_layers - 1)
            ] if self.distil else None,
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, self.factor, attention_dropout=self.dropout, output_attention=False),
                        self.d_model, self.n_heads),
                    AttentionLayer(
                        ProbAttention(False, self.factor, attention_dropout=self.dropout, output_attention=False),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model),
            projection=nn.Linear(self.d_model, self.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]