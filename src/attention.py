import torch
from torch import nn
import math

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.projection = nn.Linear(config.n_embd, config.n_embd)
        self.dropout_attn = nn.Dropout(config.attn_pdrop)
        self.drop_projection = nn.Dropout(config.resid_pdrop)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Boolean for the mask to save memory -> 1 bit per element instead of 32 bits for a float
        self.mask = torch.triu(torch.ones(config.block_size, config.block_size), diagonal=1).bool()


    def forward(self, x):
        # B = batch size, T = sequence length (nb of tokens/letters)
        # E = embedding dimension
        B, T, E = x.size()


        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Reshape x to (B, T, n_head, E // n_head)
        # to have multiple heads
        # reshape always work but view does not work
        # if the tensor is not contiguous
        # Here we can use view since the tensor is a contiguous
        # output of the linear layer
        # E.g. after transposing the tensor it is not contiguous anymore
        # We transpose to have the matrix multiplication over
        # tokens and small embedding dimension
        Q = Q.view(B, T, self.n_head, E // self.n_head).transpose(1, 2)
        K = K.view(B, T, self.n_head, E // self.n_head).transpose(1, 2)
        V = V.view(B, T, self.n_head, E // self.n_head).transpose(1, 2)

        # attention = softmax(QK/ sqrt(E)) * V
        # Need masking for causal attention
        attention_scores = Q @ K.transpose(-2, -1) * (1.0 / math.sqrt(K.size(-1)))
        attention_scores = torch.masked_fill(attention_scores, self.mask[:T, :T], float('-inf'))
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout_attn(attention_probs)
        attention_output = attention_probs @ V
        # We use contiguous since after the transpose the tensor is not contiguous anymore
        # We could have used reshape instead of view
        # but contiguous + reshape is more explicit
        attention_output = attention_output.transpose(1, 2).contiguous().view(B, T, E)

        # Final projection
        # This is the layer that learn how "shuffle" the attention output
        # So one head output can impact the whole output rather
        # than just the smaller embedding dimension it is trained on
        output = self.projection(attention_output)
        output = self.drop_projection(output)
        return output





