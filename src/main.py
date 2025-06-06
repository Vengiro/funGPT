from models import GPT1config, GPT
from attention import CausalSelfAttention
from trainer import Trainer, TrainerConfig
from dataset import FileTokenizer, CustomDataLoader
import tiktoken
import os
import torch

def main():

    encoder = tiktoken.get_encoding("gpt2")
    print(f"Total number of tokens in the encoder: {encoder.n_vocab}")
    nb_tokens = encoder.n_vocab
    data_path = 'data/shakeSpeare.txt'
    tokens_path = data_path.replace('.txt', '_tokens.npy')

    tokenizer = FileTokenizer(data_path, encoder)
    if os.path.exists(tokens_path):
        print(f"Loading tokens from {tokens_path}")
        tokens = tokenizer.load_tokens(tokens_path)
    else:
        print(f"Encoding and saving tokens to {tokens_path}")
        tokens = tokenizer.encode_save(tokens_path)

    print(f"Number of tokens in the dataset: {len(tokens)}")

    block_size = 1024
    batch_size = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    gpt_config = GPT1config(vocab_size=nb_tokens, block_size=block_size, device=device)
    trainer_config = TrainerConfig(batch_size=batch_size)

    model = GPT(gpt_config)
    model = model.to(device)
    #model = torch.compile(model, backend="inductor", mode="default")
    data_loader = CustomDataLoader(tokens, block_size, batch_size=batch_size, shuffle=True)
    trainer = Trainer(trainer_config, model, data_loader, None, device)
    trainer.train()

    print("Training completed.")


if __name__ == "__main__":
    main()

