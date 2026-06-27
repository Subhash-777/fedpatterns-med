# M2 Patternized Architecture

This diagram illustrates the implementation of the **FederatedMediator (P1)** and **FederatedStrategy (P2)** patterns in Milestone 2. By introducing the Mediator, we successfully decoupled the MLflow observability and orchestration lifecycle from the pure mathematical aggregation (FedAvg).

```mermaid
classDiagram
    class ServerApp {
        <<Framework>>
    }
    
    class Strategy {
        <<Interface>>
        +aggregate_fit()
        +aggregate_evaluate()
    }
    
    class FederatedStrategy {
        <<Interface / P2>>
        +aggregate_fit()
        +aggregate_evaluate()
    }
    
    class FederatedMediator {
        <<P1>>
        -strategy: FederatedStrategy
        -run_id: str
        +aggregate_fit()
        +aggregate_evaluate()
    }
    
    class PatternizedFedAvg {
        <<Concrete P2>>
    }
    
    class MLflowTracker {
        <<Utility>>
        +log_round_metrics()
    }

    ServerApp --> Strategy : uses
    FederatedMediator ..|> Strategy : implements
    FederatedMediator --> FederatedStrategy : delegates math to
    FederatedMediator --> MLflowTracker : logs metrics via
    PatternizedFedAvg ..|> FederatedStrategy : implements
    
    note for FederatedMediator "Intercepts FL hooks to handle\nside-effects (observability)\nthen delegates aggregation."
    note for PatternizedFedAvg "Pure math. Devoid of side-effects."
```
