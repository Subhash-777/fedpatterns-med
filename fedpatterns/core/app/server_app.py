import logging
import os
import random
import numpy as np
import torch
import mlflow

from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common import Context, ndarrays_to_parameters

from fedpatterns.core.domain.cnn_model import SmallCNN
from fedpatterns.patterns.mediator.federated_mediator import FederatedMediator
from fedpatterns.patterns.strategy.federated_strategy import PatternizedFedAvg

log = logging.getLogger(__name__)

NUM_CLIENTS = 5

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

def server_fn(context: Context) -> ServerAppComponents:
    run_config = context.run_config
    dataset_name = run_config.get("fed_dataset", "PneumoniaMNIST").lower()
    num_classes = 14 if dataset_name == "chestmnist" else 1

    tracking_uri = run_config.get("mlflow-tracking-uri", "")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    run_id = run_config.get("mlflow-run-id", "")
    if run_id:
        mlflow.start_run(run_id=run_id)

    model = SmallCNN(num_classes=num_classes)
    enable_dp = run_config.get("fed_enable_dp", "True").lower() == "true"
    if enable_dp:
        from opacus.validators import ModuleValidator
        if not ModuleValidator.is_valid(model):
            model = ModuleValidator.fix(model)
            
    initial_params_np = [val.cpu().numpy() for val in model.state_dict().values()]
    initial_parameters = ndarrays_to_parameters(initial_params_np)

    inner_strategy = PatternizedFedAvg(
        initial_parameters=initial_parameters,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=lambda rnd: {"server_round": rnd},
        on_evaluate_config_fn=lambda rnd: {"server_round": rnd},
    )

    mediator = FederatedMediator(
        strategy=inner_strategy,
        run_id=run_id
    )

    return ServerAppComponents(strategy=mediator, config=ServerConfig(num_rounds=3))

app = ServerApp(server_fn=server_fn)
