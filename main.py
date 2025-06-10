from src.models import GPT1config, GPT
from src.attention import CausalSelfAttention
from src.trainer import Trainer, TrainerConfig
from src.dataset import Tokenizer, CustomDataLoader
import tiktoken
import os
import torch

def main():

    input = "I am a model that can generate"
    dataset_tokens = True  # Set to True if you want to use only dataset tokens
    flash_attention = True  # Set to True if you want to use flash attention with supporting GPUs

    encoder = tiktoken.get_encoding("gpt2")
    print(f"Total number of tokens in the encoder: {encoder.n_vocab}")
    nb_tokens = encoder.n_vocab
    data_path = 'src/data/shakeSpeare.txt'
    tokens_path = data_path.replace('.txt', '_tokens.npy')

    tokenizer = Tokenizer(data_path, encoder)
    if os.path.exists(tokens_path):
        print(f"Loading tokens from {tokens_path}")
        tokens = tokenizer.load_tokens(tokens_path)
    else:
        print(f"Encoding and saving tokens to {tokens_path}")
        tokens = tokenizer.encode_save(tokens_path)

    decoder = None
    if dataset_tokens:
        print(f"Total number of unique tokens in the dataset: {len(set(tokens))}")
        tokens, nb_tokens, decoder = tokenizer.dataset_vocab(tokens)

    block_size = 32
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    gpt_config = GPT1config(vocab_size=nb_tokens, block_size=block_size, device=device, flash_attention=flash_attention)
    trainer_config = TrainerConfig(batch_size=batch_size)

    model = GPT(gpt_config)
    model = model.to(device)
    model = torch.compile(model, backend="eager")
    data_loader = CustomDataLoader(tokens, block_size, batch_size=batch_size, shuffle=True)
    trainer = Trainer(trainer_config, model, data_loader, None, device)
    trainer.train(tokenizer, input, decoder)

    print("Training completed.")


if __name__ == "__main__":
    main()

