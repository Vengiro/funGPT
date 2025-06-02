import torch

class TrainerConfig:
     def __init__(self, learning_rate=0.001, batch_size=32, epochs=10):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs



class Trainer:
    def __init__(self, config, model, data_loader):

        self.config = config
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.train()
        self.data_loader = data_loader


    def train(self, train_loader):
