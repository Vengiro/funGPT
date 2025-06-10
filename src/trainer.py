import torch
import time
from src.dataset import Tokenizer
class TrainerConfig:
     def __init__(self, learning_rate=0.001, batch_size=32, epochs=10):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs



class Trainer:
    def __init__(self, config, model, data_loader, eval_loader, device):

        self.config = config
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = device
        self.data_loader = data_loader
        self.eval_loader = eval_loader


    def train(self, tokenizer, input=None, decoder=None):
        if input is not None:
            input_tokens = tokenizer.tokenize(input)

        self.model.train()  # Set the model to training mode
        for ep in range(self.config.epochs):
            total_loss = 0.0
            print(f"Number of batches in epoch {ep+1}: {len(self.data_loader)}")
            for batch_idx, (x, y) in enumerate(self.data_loader):
                tstart = time.time()

                x, y = x.to(self.device), y.to(self.device)

                # Forward pass
                outputs = self.model(x)
                # Cross-entropy loss expects the shape [B*T, Vocab] for outputs and [B*T] for targets
                # It compares each vocab_size logits against the target token
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), y.view(-1))

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                tfinish = time.time()
                #print(f"Time for one batch: {tfinish - tstart:.4f} seconds, Batch Loss: {loss.item():.4f}")

            if input is not None:
                for _ in range(3):  # Generate 3 samples
                    print(self.infer(tokenizer, input_tokens, decoder))


            avg_loss = total_loss / len(self.data_loader)
            print(f"Epoch [{ep+1}/{self.config.epochs}], Loss: {avg_loss:.4f}")




    def evaluate(self):
        self.model.eval()


    def infer(self, tokenizer, input_tokens, decoder=None, max_length=10, temp=1.0):
        input_tokens = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(self.device)  # Add batch dimension
        with torch.no_grad():
            for _ in range(max_length):
                # Only feed the input tokens up to the block size
                outputs = self.model.infer(input_tokens[:, -self.model.block_size:], temp)
                idx = outputs.argmax(dim=-1).item()
                if decoder is not None:
                    idx = decoder[idx]
                result_token = torch.tensor([idx], device=self.device, dtype=torch.long).unsqueeze(0)  # Add batch dimension
                input_tokens = torch.cat((input_tokens, result_token), dim=1)

            generated_tokens = input_tokens.squeeze(0).tolist()
            output_sequence = tokenizer.detokenize(generated_tokens)

            return output_sequence
