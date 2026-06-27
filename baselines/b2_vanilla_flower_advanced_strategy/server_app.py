import logging
import random
import numpy as np
import torch
import mlflow

from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common import Context, ndarrays_to_parameters
from flwr.server.strategy import FedProx

from fedpatterns.core.domain.cnn_model import SmallCNN

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    accuracy = sum(accuracies) / sum(examples)
    
    # MLflow is set up globally in server_fn, but active run context may be lost.
    # We can retrieve the active run if it exists, or just use mlflow.log_metric
    # if it doesn't fail.
    try:
        mlflow.log_metric("accuracy", accuracy)
    except Exception as e:
        log.warning(f"Failed to log metric: {e}")
        
    return {"accuracy": accuracy}

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
    initial_params_np = [val.cpu().numpy() for val in model.state_dict().values()]
    initial_parameters = ndarrays_to_parameters(initial_params_np)

    strategy = FedProx(
        proximal_mu=0.01,
        initial_parameters=initial_parameters,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=lambda rnd: {"server_round": rnd, "proximal_mu": 0.01},
        on_evaluate_config_fn=lambda rnd: {"server_round": rnd},
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=3))

app = ServerApp(server_fn=server_fn)
