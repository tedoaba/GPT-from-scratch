import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import optuna
import torch
from optuna.integration import PyTorchLightningPruningCallback
from src.train import main as train_model
from src.config import config
import logging

def objective(trial):
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1)
    n_layer = trial.suggest_int('n_layer', 1, 4)
    n_embed = trial.suggest_categorical('n_embed', [16, 32, 64])
    num_head = trial.suggest_categorical('num_head', [2, 4, 8])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2)

    config.batch_size = batch_size
    config.learning_rate = learning_rate
    config.n_layer = n_layer
    config.n_embed = n_embed
    config.num_head = num_head
    config.dropout = dropout
    config.weight_decay = weight_decay

    train_loss, val_loss = train_model()
    val_loss_tensor = torch.tensor(val_loss)
    if torch.isnan(val_loss_tensor):
        return float('inf')
    
    return val_loss

def tune_hyperparameters():
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=2, n_jobs=4)

    if len(study.trials) == 0:
        raise ValueError("No trials are completed yet.")

    best_params = study.best_params
    print("Best hyperparameters: ", best_params)

    with open('logs/hyperparameters/best_hyperparameters.txt', 'w') as f:
        f.write("Best hyperparameters: \n")
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    tune_hyperparameters()