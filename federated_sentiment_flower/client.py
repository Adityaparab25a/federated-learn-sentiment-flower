import flwr as fl
from train_utils import preprocess_data, get_model, train, evaluate
import torch


class SentimentClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        self.model = get_model()
        self.dataset = preprocess_data(f'client_data_{client_id}.csv')


    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]


    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict, strict=True)


    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model = train(self.model, self.dataset)
        return self.get_parameters(config={}), len(self.dataset), {}


    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        acc = evaluate(self.model, self.dataset)
        return float(0.0), len(self.dataset), {"accuracy": acc}


if __name__ == "__main__":
    import sys
    client_id = int(sys.argv[1])
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=SentimentClient(client_id))