import os
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from fedpatterns.core.domain.cnn_model import SmallCNN
from fedpatterns.datasets.medmnist.loader import get_full_dataset

# Constants
NUM_CLASSES = 1
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, scaler: torch.cuda.amp.GradScaler) -> float:
    model.train()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE).float()
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * inputs.size(0)
        
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE).float()
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            total_loss += loss.item() * inputs.size(0)
            preds = (outputs >= 0.0).float()
            correct += (preds == targets).sum().item()
            
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def main():
    # Load all data centrally
    train_dataset = get_full_dataset(split="train")
    val_dataset = get_full_dataset(split="val")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = SmallCNN(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    # MLflow setup
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT}/mlflow.db")
    mlflow.set_experiment("M5-B0-Centralized")
    
    history = []
    
    with mlflow.start_run(tags={"baseline_id": "B0", "milestone": "M5"}) as run:
        set_seed(42)
        print(f"Started Centralized Training. Run ID: {run.info.run_id}")
        
        for epoch in range(1, EPOCHS + 1):
            train_loss = train(model, train_loader, optimizer, scaler)
            val_loss, val_acc = evaluate(model, val_loader)
            
            print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": val_acc
            }, step=epoch)
            
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })
            
    # Save results
    results_dir = PROJECT_ROOT / "output" / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)
    df.to_csv(results_dir / "b0_results.csv", index=False)
    print("Training complete. Results saved.")

if __name__ == "__main__":
    main()
