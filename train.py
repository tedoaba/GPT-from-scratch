import torch
import logging
from config import config
from data_loader import prepare_data

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f"Using device: {config.device}")

    torch.manual_seed(config.seed)
    
    logger.info("Loading data...")
    train_data, val_data, vocab_size, decode = prepare_data(config.data_path)
    logger.info("Data loaded successfully.")
