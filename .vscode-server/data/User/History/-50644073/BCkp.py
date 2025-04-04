
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch
import torch.nn as nn
import numpy as np

class PatchClassEmbedding(nn.Module):
    def __init__(self, d_model, n_patches, pos_emb=None):
        super(PatchClassEmbedding, self).__init__()
        self.d_model = d_model
        self.n_tot_patches = n_patches + 1
        self.pos_emb = pos_emb

        # Class embedding token
        self.class_embed = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.kaiming_normal_(self.class_embed)

        if self.pos_emb is not None:
            self.pos_emb = torch.tensor(np.load(self.pos_emb), dtype=torch.float32)
            self.lap_position_embedding = nn.Embedding(self.pos_emb.size(0), self.d_model)
        else:
            self.position_embedding = nn.Embedding(self.n_tot_patches, self.d_model)

    def forward(self, inputs):
        batch_size = inputs.size(0)

        # Repeat the class token for each batch
        class_embed = self.class_embed.repeat(batch_size, 1, 1)

        # Concatenate the class token and inputs
        x = torch.cat((class_embed, inputs), dim=1)

        if self.pos_emb is None:
            positions = torch.arange(self.n_tot_patches, device=inputs.device).unsqueeze(0)
            pe = self.position_embedding(positions)
        else:
            pe = self.pos_emb.unsqueeze(0).to(inputs.device)
            pe = self.lap_position_embedding(pe)

        encoded = x + pe
        return encoded


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        # Linear layers for query, key, and value
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        # Final linear layer for the output
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth) and transpose for attention."""
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        # Reshape and concat back
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)

        output = self.dense(concat_attention)
        return output, attention_weights

    def scaled_dot_product_attention(self, q, k, v, mask):
        dk = q.size(-1)
        scaled_qk = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

        if mask is not None:
            scaled_qk = scaled_qk.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.nn.functional.softmax(scaled_qk, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights




class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, activation):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, training=False):
        attn_output, _ = self.mha(x, x, x)
        out1 = self.layernorm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + self.dropout2(ffn_output))
        return out2


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, activation, n_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ACTModel(nn.Module):
    def __init__(self, config):
        super(ACTModel, self).__init__()
        
        self.d_model = 64*4
        self.mlp_head_size = 512

        # Transformer component
        transformer = TransformerEncoder(
            d_model=self.d_model,
            num_heads=4,
            d_ff=self.d_model*4,
            dropout=0.4,
            activation=f.gelu,
            n_layers=6
        )
        
        self.transformer = transformer
        self.dense1 = nn.Linear(config['KEYPOINTS'] * config['CHANNELS'], self.d_model)
        self.patch_embed = PatchClassEmbedding(self.d_model, config['FRAMES'] // config['SUBSAMPLE'])
        self.mlp_head = nn.Linear(self.d_model, self.mlp_head_size)
        self.final_dense = nn.Linear(self.mlp_head_size, config['CLASSES'])

    def forward(self, inputs):
        x = self.dense1(inputs)
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = x[:, 0, :]  # Taking only the class token
        x = self.mlp_head(x)
        x = self.final_dense(x)
        return x
