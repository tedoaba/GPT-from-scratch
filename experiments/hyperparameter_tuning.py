import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import optuna
import torch
from train import main as train_model
from config import config
import logging

def objective(trial):
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1)
    n_layer = trial.suggest_int('n_layer', 1, 4)
    n_embed = trial.suggest_categorical('n_embed', [16, 32, 64])
    num_head = trial.suggest_categorical('num_head', [2, 4, 8])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)

    config.batch_size = batch_size
    config.learning_rate = learning_rate
    config.n_layer = n_layer
    config.n_embed = n_embed
    config.num_head = num_head
    config.dropout = dropout

    train_loss, val_loss = train_model()
    return val_loss

def tune_hyperparameters():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=2)

    best_params = study.best_params
    print("Best hyperparameters: ", best_params)

    with open('logs/hyperparameters/best_hyperparameters.txt', 'w') as f:
        f.write("Best hyperparameters: \n")
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    tune_hyperparameters()