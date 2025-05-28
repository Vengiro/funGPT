import torch
from torch import nn
from attention import CausalSelfAttention





class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Normalize the input text over the embedding dimension
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        # fc layer post attention
        self.fc = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop)
        )




class GPTconfig:
    """ base GPT config """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    rope = False
    bottleneck_dim = None

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        # If extra parameters are provided, set them as attributes
        # E.g. layers=12, heads=12, embd_dim=768 dict
        for k, v in kwargs.items():
            setattr(self, k, v)

class GPT1config(GPTconfig):
    """ GPT-1  network params """
    n_layer = 12
    n_head = 12
    n_embd = 768