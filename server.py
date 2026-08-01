"""Flower coordinator: broadcasts global weights and performs FedAvg."""

from __future__ import annotations

import argparse

import flwr as fl

from strategy import build_strategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the federated-learning server")
    parser.add_argument("--address", default="0.0.0.0:8080", help="Bind address for gRPC")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated rounds")
    parser.add_argument("--min-clients", type=int, default=3, help="Hospitals required per round")
    args = parser.parse_args()

    print(f"Starting Flower server at {args.address}; waiting for {args.min_clients} hospitals.")
    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=build_strategy(args.min_clients),
    )


if __name__ == "__main__":
    main()
