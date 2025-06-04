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
            nn.GELU(approximate='tanh'),  # Use tanh approximation for GELU
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop)
        )

    def forward(self, x):
        # Normalize the input text + apply attention and add the result to the input
        x = x + self.attention(self.ln1(x))
        # Normalize the aware (thanks to the attention) input text + apply a FC layer
        # and add the result to the input
        x = x + self.fc(self.ln2(x))
        return x




class GPTconfig:
    """ base GPT config """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1


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


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.block_size = config.block_size
        self.n_layer = config.n_layer
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # input embedding layer
        # all possible tokens are mapped to a vector of size n_embd
        self.embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        # each position in the input sequence (of max length block_size) is mapped to a learnable vector of size n_embd
        self.position_emb = nn.Embedding(config.block_size, config.n_embd)
        # transformer blocks
        # * operator is used to unpack the list of TransformerBlock instances
        # [t1, t2, ..., tn] into arguments for nn.Sequential(t1, t2, ..., tn)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layer)])
        # Final normalization layer
        self.ln_f = nn.LayerNorm(config.n_embd)
        # output layer
        self.output_layer = nn.Linear(config.n_embd, config.vocab_size)

        # Tie the last FC layer with the input embedding layer as in GPT-2
        # Reduce number of parameters and improve performance by decreasing
        # distance (dot product) between the semantically similar tokens
        self.output_layer.weight = self.embeddings.weight

        ## Something went wrong with the initialization of my weights since without this proper initialization the loss
        # start around 780
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std= 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, x):
        # x is a sequence of token indices of shape (batch_size, sequence_length)
        B, T = x.size()
        # Get the token embeddings for each token in the sequence
        token_emb = self.embeddings(x)
        # Get the position indices for each token in the sequence
        # arange creates a tensor with values from 0 to T-1
        position_indices = torch.arange(T, device=x.device)
        # get position embeddings
        position_emb = self.position_emb(position_indices)
        # No need to add dimension to position_indices since it broadcasts automatically
        input_emb = token_emb + position_emb
        # Forward the input through the transformer blocks
        # then the output is passed through the output layer
        # Dim of the input to the blocks is (B, T, n_embd)
        x = self.blocks(input_emb)
        x = self.ln_f(x)  # Final normalization
        x = self.output_layer(x)
        # The output is of shape (B, T, vocab_size)
        return x

    def infer(self, x):
        """
        Predict the next token in the sequence.
        :param x: input sequence of shape (batch_size, sequence_length)
        :return: predicted token probabilities of shape (batch_size, sequence_length, vocab_size)
        """

        # Same thing as forward
        B, T = x.size()
        token_emb = self.embeddings(x)
        position_indices = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        position_emb = self.position_emb(position_indices)
        input_emb = token_emb + position_emb
        x = self.blocks(input_emb)

        # Only feed the last aware token since
        # we want to predict the next token
        x = x[:, -1, :]
        # The output is of shape (B, vocab_size)
        x = self.output_layer(x)
        # Apply softmax to get probabilities
        x = torch.softmax(x, dim=-1)
        return x

