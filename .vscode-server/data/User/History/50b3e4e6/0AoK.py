import torch
import torch.nn as nn
import torch.nn.functional as f
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
        self.position_embedding = nn.Embedding(770, self.d_model)

    def forward(self, inputs, mask=None):
        batch_size = inputs.size(0)
        batch_size, seq_len, _ = inputs.size()
        # Repeat the class token for each batch
        class_embed = self.class_embed.repeat(batch_size, 1, 1)

        # Concatenate the class token and inputs
        x = torch.cat((class_embed, inputs), dim=1)

        if self.pos_emb is None:
            positions = torch.arange(seq_len + 1, device=inputs.device).unsqueeze(0).expand(batch_size, -1)
            pe = self.position_embedding(positions)
            
            if mask is not None:
                # Expand the mask to match the embedding size (batch_size, seq_len + 1, d_model)
                padding_mask = mask.squeeze(1).squeeze(1)  # Shape: [batch_size, seq_len]
                padding_mask = torch.cat([torch.ones((batch_size, 1), device=inputs.device, dtype=torch.long), padding_mask], dim=1)

                #
                # Expand the mask to match the embedding size (batch_size, seq_len + 1, d_model)
                padding_mask = mask.squeeze(1).squeeze(1)  # Shape: [batch_size, seq_len]
                padding_mask = torch.cat([torch.ones((batch_size, 1), device=inputs.device, dtype=torch.long), padding_mask], dim=1)

                # Expand padding mask to the embedding dimension
                padding_mask = padding_mask.unsqueeze(-1).expand(-1, -1, self.d_model)
                pe = pe.masked_fill(padding_mask == 0, 0)
  

        encoded = x + pe
        return encoded


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.attention_weights = None
        self.outputs = None
        self.attention_weights_grad =  None
        
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

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        
        dk = q.size(-1)
        scaled_qk = q @ k.transpose(-2, -1) / (dk ** 0.5 + 1e-8)

        if mask is not None:
            mask = torch.cat([torch.ones((mask.shape[0], 1, 1, 1), device=mask.device), mask], dim=-1)
            scaled_qk = scaled_qk.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.nn.functional.softmax(scaled_qk, dim=-1)
        
        attention_weights.register_hook(self.save_attention_weights_grad)
        
        # output = (attention_weights.unsqueeze(-1) * v.unsqueeze(-2)).sum(dim=-2)
        output = attention_weights @ v
        self.outputs = output
        
        return output, attention_weights
    
    def save_attention_weights_grad(self, grad):
        self.attention_weights_grad = grad


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

    def forward(self, x, mask, training=False):
        attn_output, _ = self.mha(x, x, x, mask)
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

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


def create_padding_mask(lengths, max_len):
    """Creates a mask for padded elements in the sequence."""
    batch_size = lengths.size(0)
    mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
    return mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, max_len)


class AcT(nn.Module):
    def __init__(self, dropout):
        super(AcT, self).__init__()
        
        self.d_model = 64*4
        self.mlp_head_size = 512

        # Transformer component
        transformer = TransformerEncoder(
            d_model=self.d_model,
            num_heads=4,
            d_ff=self.d_model*4,
            dropout=dropout,
            activation=f.gelu,
            n_layers=6
        )
        
        self.transformer = transformer
        self.dense1 = nn.Linear(52, self.d_model)
        self.patch_embed = PatchClassEmbedding(self.d_model, 30)
        self.mlp_head = nn.Linear(self.d_model, self.mlp_head_size)
        self.final_dense = nn.Linear(self.mlp_head_size, 20)

    def forward(self, inputs, lengths=None):
        x = self.dense1(inputs)
        max_len = x.size(1)  # Sequence length after embedding
        mask = create_padding_mask(lengths, max_len)
        x = self.patch_embed(x, mask.to(x.device))
        x = self.transformer(x, mask.to(x.device))
        x = x[:, 0, :]  # Taking only the class token
        x = self.mlp_head(x)
        x = self.final_dense(x)
        return x
    

