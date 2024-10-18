"""
Lightweight Temporal Attention Encoder module

Modified from the original code by: VSainteuf
code: https://github.com/VSainteuf/lightweight-temporal-attention-pytorch/tree/master
module: https://github.com/VSainteuf/lightweight-temporal-attention-pytorch/blob/master/models/ltae.py
paper: https://arxiv.org/abs/2007.00586

Modifications:
 * enabled sample-wise positions: model expects 0th band to be positions
 * moved decoder/classification head logic into LTAE module (was in PSE-LTAE but we're not using PSE here)

Original file notes:
    Credits:
    The module is heavily inspired by the works of Vaswani et al. on self-attention and their pytorch implementation of
    the Transformer served as code base for the present script.

    paper: https://arxiv.org/abs/1706.03762
    code: github.com/jadore801120/attention-is-all-you-need-pytorch
"""

import torch
import torch.nn as nn
import numpy as np
import copy


class LTAE(nn.Module):
    def __init__(self, in_channels=10, n_head=16, d_k=8, mlp3=[256,128], mlp4=[128, 64, 32], n_classes=20, dropout=0.2, d_model=256,
                 T=1000, len_max_seq=24, return_att=False, attention_type="LTAE"):
        """
        Sequence-to-embedding encoder.
        Args:
            in_channels (int): Number of channels of the input embeddings
            n_head (int): Number of attention heads
            d_k (int): Dimension of the key and query vectors
            mlp3 (list): Defines the dimensions of the successive feature spaces of the MLP that processes
                the concatenated outputs of the attention heads
            dropout (float): dropout
            T (int): Period to use for the positional encoding
            len_max_seq (int, optional): Maximum sequence length, used to pre-compute the positional encoding table
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)
            attention_type (str): Type of attention mechanism to use.
                "LTAE" uses the modified LTAE model mechanism.
                "vanilla" uses the original Vaswani mechanism with a max pooling operation to collapse the time dimension.

        Note: original implementation used index positions or static positions provided at init.
        This version is modified to expect the positions as the 0th band of the input tensor.

        """

        super(LTAE, self).__init__()
        self.in_channels = in_channels
        self.mlp3 = copy.deepcopy(mlp3)
        self.mlp4 = copy.deepcopy(mlp4 + [n_classes])
        self.return_att = return_att
        self.d_model = d_model
        self.attention_type = attention_type

        if d_model is not None:
            self.inconv = nn.Sequential(
                nn.Conv1d(in_channels, d_model, 1),
                #nn.LayerNorm([d_model])
                nn.BatchNorm1d(d_model)  # Use BatchNorm1d instead of LayerNorm for Conv1d output
            )
        else:
            self.inconv = None
            self.d_model = in_channels

        self.inlayernorm = nn.LayerNorm(self.in_channels)
        self.outlayernorm = nn.LayerNorm(self.mlp3[-1])

        if attention_type == "LTAE":
            self.attention_heads = LTAEMultiHeadAttention(
                n_head=n_head, d_k=d_k, d_in=self.d_model)
        elif attention_type == "vanilla":
            self.attention_heads = VanillaMultiHeadAttention(
                n_head=n_head, d_model=self.d_model, d_k=d_k, d_v=d_k, dropout=dropout)
        
        assert (self.mlp3[0] == self.d_model)

        activation = nn.ReLU()

        layers = []
        for i in range(len(self.mlp3) - 1):
            layers.extend([
                nn.Linear(self.mlp3[i], self.mlp3[i + 1]),
                nn.BatchNorm1d(self.mlp3[i + 1]),
                activation
            ])

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.T = T  # For positional encoding

        # Decoder for classification
        self.decoder = get_decoder(self.mlp4)

    def forward(self, x):
        """
        Forward pass for the LTAE module.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, seq_len, in_channels + 1).
                The 0th band (last dimension) is expected to be the positions.

        Returns:
            torch.Tensor: Output embeddings with shape (batch_size, mlp3[-1]).
            (Optional) torch.Tensor: Attention weights.
        """
        # x: (batch_size, seq_len, in_channels + 1)
        sz_b, seq_len, d = x.shape
        positions = x[:, :, 0]  # Extract positions: shape (batch_size, seq_len)
        x = x[:, :, 1:]  # Remove positions from input features: shape (batch_size, seq_len, in_channels)

        x = self.inlayernorm(x)

        if self.inconv is not None:
            # Reshape for Conv1d: (batch_size, in_channels, seq_len)
            x = x.permute(0, 2, 1)
            x = self.inconv(x)  # Apply Conv1d and LayerNorm
            x = x.permute(0, 2, 1)  # Reshape back to (batch_size, seq_len, d_model)

        # Compute positional encodings
        pos_enc = get_sinusoid_encoding(positions, self.d_model, T=self.T)
        enc_output = x + pos_enc  # Add positional encodings

        # Apply attention heads
        enc_output, attn = self.attention_heads(enc_output, enc_output, enc_output)

        if self.attention_type == "LTAE":
            # Concatenate heads
            enc_output = enc_output.permute(1, 0, 2).contiguous().view(sz_b, -1)  # Concatenate heads
        elif self.attention_type == "vanilla":
            # Max pooling over time
            enc_output = enc_output.max(dim=1)[0]

        enc_output = self.outlayernorm(self.dropout(self.mlp(enc_output)))

        enc_output = self.decoder(enc_output)

        if self.return_att:
            return enc_output, attn
        else:
            return enc_output


class LTAEMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = LTAEScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, q, k, v):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = q.size()

        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(-1, d_k)  # (n*b) x d_k

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(n_head * sz_b, seq_len, -1)
        output, attn = self.attention(q, k, v)
        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)

        output = output.view(n_head, sz_b, 1, d_in // n_head)
        output = output.squeeze(dim=2)

        return output, attn


class LTAEScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        return output, attn

class VanillaScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)  # Apply softmax over the last dimension

    def forward(self, q, k, v, mask=None):
        # Ensure that q, k, v have 4 dimensions: (batch_size, n_head, seq_len, d_k)
        # Apply attention: (batch_size, n_head, seq_len, seq_len)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.temperature

        # Apply mask if provided
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # Softmax + dropout
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        # Multiply attention weights by values (output shape: (batch_size, n_head, seq_len, d_v))
        output = torch.matmul(attn, v)

        return output, attn

def get_decoder(mlp3):
    """Returns an MLP with the layer widths specified in mlp3."""
    layers = []
    for i in range(len(mlp3) - 1):
        layers.append(nn.Linear(mlp3[i], mlp3[i + 1]))
        if i < (len(mlp3) - 2):
            layers.extend([
                nn.BatchNorm1d(mlp3[i + 1]),
                nn.ReLU()
            ])
    m = nn.Sequential(*layers)
    return m

def get_sinusoid_encoding(position, d_hid, T=1000):
    """
    Generate sinusoid encoding for a sequence of positions.

    Args:
        position (torch.Tensor): Tensor of positions with shape (batch_size, seq_len).
        d_hid (int): Hidden dimension for positional encoding.
        T (int): Temperature parameter for scaling.

    Returns:
        torch.Tensor: Positional encodings with shape (batch_size, seq_len, d_hid).
    """
    batch_size, seq_len = position.size()
    position = position.unsqueeze(2)  # Shape: (batch_size, seq_len, 1)
    div_term = torch.exp(torch.arange(0, d_hid, 2, dtype=torch.float32, device=position.device) *
                         -(np.log(T) / d_hid))
    div_term = div_term.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, d_hid//2)
    pos_enc = torch.zeros(batch_size, seq_len, d_hid, device=position.device)
    pos_enc[:, :, 0::2] = torch.sin(position * div_term)
    pos_enc[:, :, 1::2] = torch.cos(position * div_term)
    return pos_enc


class VanillaMultiHeadAttention(nn.Module):
    ''' Vanilla Multi-Head Attention module (similar to Vaswani et al.) '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        
        self.attention = VanillaScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        
        self.fc = nn.Linear(n_head * d_v, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, d_model = q.size()
        len_k, len_v = k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n_head * d_k)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention calculation: b x n_head x lq x d_k
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        # Apply attention on all the projected vectors in batch.
        output, attn = self.attention(q, k, v)
        
        # Reshape back to original shape
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # Concatenate heads

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn