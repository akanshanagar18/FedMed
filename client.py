"""Private hospital-side Flower client for BRISC MRI classification."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import torch
import torch.nn as nn

from data import load_data
from model import BrainTumorCNN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HospitalClient(fl.client.NumPyClient):
    def __init__(self, hospital_id: str, trainloader, testloader) -> None:
        self.hospital_id = hospital_id
        self.trainloader = trainloader
        self.testloader = testloader
        self.model = BrainTumorCNN().to(DEVICE)
        self.loss_fn = nn.CrossEntropyLoss()

    def get_parameters(self, config: Dict[str, str]):
        return [value.detach().cpu().numpy() for value in self.model.state_dict().values()]

    def set_parameters(self, parameters: List) -> None:
        state_dict = OrderedDict(
            (key, torch.tensor(value))
            for key, value in zip(self.model.state_dict().keys(), parameters)
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        epochs = int(config.get("local_epochs", 1))
        learning_rate = float(config.get("learning_rate", 0.001))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.train()
        for _ in range(epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                self.loss_fn(self.model(images), labels).backward()
                optimizer.step()
        print(f"Hospital {self.hospital_id}: completed round {config.get('server_round', '?')}")
        return self.get_parameters({}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config) -> Tuple[float, int, Dict[str, float]]:
        self.set_parameters(parameters)
        self.model.eval()
        total_loss = correct = total = 0
        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits = self.model(images)
                total_loss += self.loss_fn(logits, labels).item() * labels.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
        return total_loss / total, total, {"accuracy": correct / total}


def main() -> None:
    parser = argparse.ArgumentParser(description="Start one private hospital Flower client")
    parser.add_argument("--hospital", choices=("A", "B", "C"), required=True)
    parser.add_argument("--server", default="127.0.0.1:8080", help="Flower server host:port")
    parser.add_argument("--data-root", default="brisc2025", help="Hospital-local BRISC root")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--synthetic", action="store_true", help="Use fake data for a quick connectivity test")
    args = parser.parse_args()

    trainloader, testloader = load_data(args.data_root, args.hospital, args.batch_size, args.synthetic)
    client = HospitalClient(args.hospital, trainloader, testloader)
    print(f"Hospital {args.hospital}: connecting to {args.server} on {DEVICE}.")
    fl.client.start_client(server_address=args.server, client=client.to_client())


if __name__ == "__main__":
    main()
