import flwr as fl


strategy = fl.server.strategy.FedAvg(
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
)


fl.server.start_server(server_address="0.0.0.0:8080", strategy=strategy, config={"num_rounds": 3})