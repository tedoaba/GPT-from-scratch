import torch
import logging
from config import config

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def create_vocab(text):
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos, vocab_size

def train_val_split(data, split_ratio=0.9):
    n = int(split_ratio * len(data))
    return data[:n], data[n:]

def encode_text(text, stoi):
    return [stoi[c] for c in text]

def prepare_data(file_path: str):
    logger = logging.getLogger(__name__)
    logger.info("Loading data from %s", file_path)
    
    text = load_data(file_path)
    stoi, itos, vocab_size = create_vocab(text)
    data = torch.tensor(encode_text(text, stoi), dtype=torch.long)
    train_data, val_data = train_val_split(data, config.split_ratio)

    logger.info("Data loaded: %d training tokens, %d validation tokens", len(train_data), len(val_data))

    decode = lambda l: ''.join([itos[i] for i in l])
    return train_data, val_data, vocab_size, decode
