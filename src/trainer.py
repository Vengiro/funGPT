import torch
import time
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


    def train(self):

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
                print(f"Time for one batch: {tfinish - tstart:.4f} seconds, Batch Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(self.data_loader)
            print(f"Epoch [{ep+1}/{self.config.epochs}], Loss: {avg_loss:.4f}")


    def evaluate(self):
        self.model.eval()