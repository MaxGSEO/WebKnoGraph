# File: src/backend/services/graph_training_service.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch_geometric.data import Data
from src.shared.interfaces import ILogger
from src.backend.models.graph_models import GraphSAGEModel


class LinkPredictionTrainer:
    def __init__(self, model: GraphSAGEModel, data: Data, config, logger: ILogger):
        self.model = model
        self.data = data
        self.config = config
        self.logger = logger
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()

    def _get_negative_samples(self):
        return torch.randint(
            0, self.data.num_nodes, (2, self.data.num_edges), dtype=torch.long
        )

    def train(self):
        edge_label_index = torch.cat(
            [self.data.edge_index, self._get_negative_samples()], dim=1
        )
        edge_label = torch.cat(
            [torch.ones(self.data.num_edges), torch.zeros(self.data.num_edges)], dim=0
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.data = self.data.to(device)
        edge_label_index = edge_label_index.to(device)
        edge_label = edge_label.to(device)

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()
            z = self.model(self.data.x, self.data.edge_index)
            out = self.model.predict_link(z, edge_label_index)
            loss = self.criterion(out, edge_label)
            loss.backward()
            self.optimizer.step()
            yield epoch, loss.item()
